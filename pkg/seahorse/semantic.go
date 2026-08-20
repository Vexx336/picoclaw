package seahorse

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/sipeed/picoclaw/pkg/embeddings"
	"github.com/sipeed/picoclaw/pkg/logger"
)

// Semantic memory (Tier 3, M2): embedding-based retrieval over the seahorse
// message store. Vectors are produced by a dedicated embedding model served
// by llama.cpp (skynet :18084), stored in the message_embeddings table next
// to the seahorse tables, and searched with an in-memory flat index.
//
// The design is intentionally additive: when enableSemantic is false (the
// default) none of this code runs and behavior is identical to pre-M2.
// Failures degrade to grep-only — they never block message persistence.

const (
	// semanticQueueSize bounds the async embedding queue. Ingests are small
	// (1-3 messages per turn), so a few hundred slots is plenty.
	semanticQueueSize = 512
	// maxEmbedTextLen caps per-message text sent to the embedding model.
	maxEmbedTextLen = 2000
	// defaultSemanticTopK / defaultSemanticMinScore apply when unset.
	defaultSemanticTopK     = 8
	defaultSemanticMinScore = 0.35
)

// SemanticEngine owns the embedding client, vector store, and in-memory
// index for one Engine. All write paths are asynchronous so message
// persistence never blocks on the network; failures are logged and dropped.
type SemanticEngine struct {
	db    *sql.DB
	store *Store // seahorse store for message content/parts

	client *embeddings.Client
	estore *embeddings.Store
	index  *embeddings.Index
	cfg    Config

	indexMu sync.RWMutex // guards index

	// lifecycle guards queue + closed.
	lifecycleMu sync.Mutex
	queue       chan int64
	closed      bool
	wg          sync.WaitGroup
}

// newSemanticEngine builds the semantic engine: client, vector store
// (migrates message_embeddings on the shared DB), and a loaded flat index.
func newSemanticEngine(ctx context.Context, db *sql.DB, store *Store, cfg Config) (*SemanticEngine, error) {
	endpoint := normalizeEmbeddingEndpoint(cfg.EmbeddingEndpoint)
	if endpoint == "" {
		return nil, errors.New("semantic: embeddingEndpoint is required")
	}
	if cfg.EmbeddingDim <= 0 {
		return nil, errors.New("semantic: embeddingDim must be > 0")
	}
	if cfg.EmbeddingModel == "" {
		cfg.EmbeddingModel = "default"
	}

	client := embeddings.NewClient(endpoint, cfg.EmbeddingModel, cfg.EmbeddingDim)
	estore, err := embeddings.NewStore(ctx, db, cfg.EmbeddingModel, cfg.EmbeddingDim)
	if err != nil {
		return nil, fmt.Errorf("semantic: vector store: %w", err)
	}

	se := &SemanticEngine{
		db:     db,
		store:  store,
		client: client,
		estore: estore,
		cfg:    cfg,
		queue:  make(chan int64, semanticQueueSize),
	}

	ix, err := estore.LoadIndex(ctx)
	if err != nil {
		return nil, fmt.Errorf("semantic: load index: %w", err)
	}
	se.index = ix
	logger.InfoCF("seahorse", "semantic index loaded", map[string]any{
		"vectors": ix.Len(),
		"model":   cfg.EmbeddingModel,
		"dim":     cfg.EmbeddingDim,
	})

	// Async write path is the default; it is the only mode that matters for
	// live ingest. AsyncWrite=false falls back to synchronous embedding in
	// Ingest (still non-fatal on error).
	if cfg.AsyncWriteEnabled() {
		se.startWorker()
	}
	return se, nil
}

// normalizeEmbeddingEndpoint accepts either the base URL (http://host:18084/v1)
// or the full endpoint (http://host:18084/v1/embeddings) and returns the base.
func normalizeEmbeddingEndpoint(endpoint string) string {
	e := strings.TrimRight(strings.TrimSpace(endpoint), "/")
	if strings.HasSuffix(e, "/embeddings") {
		e = strings.TrimRight(strings.TrimSuffix(e, "/embeddings"), "/")
	}
	return e
}

// startWorker launches the async embedding consumer.
func (e *SemanticEngine) startWorker() {
	e.wg.Add(1)
	go func() {
		defer e.wg.Done()
		for id := range e.queue {
			ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
			err := e.embedOne(ctx, id)
			cancel()
			if err != nil {
				logger.WarnCF("seahorse", "async embed failed", map[string]any{
					"message_id": id,
					"error":      err.Error(),
				})
			}
		}
	}()
}

// Enqueue adds message IDs to the async embedding queue. Non-blocking; drops
// when the queue is full (memory degrades to grep-only, never breaks).
func (e *SemanticEngine) Enqueue(ids ...int64) {
	e.lifecycleMu.Lock()
	defer e.lifecycleMu.Unlock()
	if e.closed {
		return
	}
	for _, id := range ids {
		select {
		case e.queue <- id:
		default:
			logger.WarnCF("seahorse", "embed queue full; dropping", map[string]any{"message_id": id})
		}
	}
}

// embedOne embeds a single message and updates the index.
func (e *SemanticEngine) embedOne(ctx context.Context, messageID int64) error {
	msg, err := e.store.GetMessageByID(ctx, messageID)
	if err != nil {
		return err
	}
	text := messageEmbedText(msg)
	vec, err := e.client.Embed(ctx, text)
	if err != nil {
		return err
	}
	if err := e.estore.Upsert(ctx, messageID, vec); err != nil {
		return err
	}
	e.addToIndex(messageID, vec)
	return nil
}

// Backfill embeds all messages that lack a vector for the configured model.
// Returns the number embedded; stops early on error or context cancellation
// (callers resume on next startup — PendingIDs only returns un-embedded rows).
func (e *SemanticEngine) Backfill(ctx context.Context, batchSize int) (int, error) {
	if batchSize <= 0 {
		batchSize = 16
	}
	total := 0
	for {
		pending, err := e.estore.PendingIDs(ctx, 0, batchSize)
		if err != nil {
			return total, err
		}
		if len(pending) == 0 {
			return total, nil
		}

		texts := make([]string, 0, len(pending))
		msgByID := make(map[int64]*Message, len(pending))
		for _, id := range pending {
			msg, err := e.store.GetMessageByID(ctx, id)
			if err != nil {
				texts = append(texts, "[missing message]")
				continue
			}
			msgByID[id] = msg
			texts = append(texts, messageEmbedText(msg))
		}

		vecs, err := e.client.EmbedBatch(ctx, texts)
		if err != nil {
			return total, err
		}
		for i, id := range pending {
			if i >= len(vecs) {
				break
			}
			if msgByID[id] == nil {
				continue // message vanished mid-backfill
			}
			if err := e.estore.Upsert(ctx, id, vecs[i]); err != nil {
				logger.WarnCF("seahorse", "backfill upsert failed", map[string]any{"message_id": id, "error": err.Error()})
				continue
			}
			e.addToIndex(id, vecs[i])
		}
		total += len(pending)
		logger.InfoCF("seahorse", "semantic backfill progress", map[string]any{"embedded": total})

		// Pause briefly so live ingest traffic isn't starved.
		select {
		case <-ctx.Done():
			return total, ctx.Err()
		case <-time.After(200 * time.Millisecond):
		}
	}
}

// addToIndex inserts or replaces a vector under the index lock.
func (e *SemanticEngine) addToIndex(messageID int64, vec []float32) {
	e.indexMu.Lock()
	defer e.indexMu.Unlock()
	e.index.Add(messageID, vec)
}

// search returns nearest neighbors for a query vector.
func (e *SemanticEngine) search(vec []float32, topK int, minScore float64) []embeddings.Hit {
	e.indexMu.RLock()
	defer e.indexMu.RUnlock()
	return e.index.Search(vec, topK, minScore)
}

// Close stops the async worker and drains the queue.
func (e *SemanticEngine) Close() {
	e.lifecycleMu.Lock()
	if e.closed {
		e.lifecycleMu.Unlock()
		return
	}
	e.closed = true
	close(e.queue)
	e.lifecycleMu.Unlock()
	e.wg.Wait()
}

// messageEmbedText builds a compact text representation of a message for
// embedding. Prefers top-level content and appends parts (tool names,
// arguments, results, media URIs) so tool-heavy turns stay searchable.
func messageEmbedText(msg *Message) string {
	var b strings.Builder
	b.WriteString(msg.Content)
	for _, p := range msg.Parts {
		switch p.Type {
		case "tool_use":
			b.WriteString(" [tool: ")
			b.WriteString(p.Name)
			if p.Arguments != "" {
				b.WriteString(" ")
				b.WriteString(p.Arguments)
			}
			b.WriteString("]")
		case "tool_result":
			if p.Text != "" {
				b.WriteString(" [result: ")
				b.WriteString(p.Text)
				b.WriteString("]")
			}
		case "media":
			if p.MediaURI != "" {
				b.WriteString(" [media: ")
				b.WriteString(p.MediaURI)
				b.WriteString("]")
			}
		}
	}
	s := strings.TrimSpace(b.String())
	if s == "" {
		return "[empty message]"
	}
	if len(s) > maxEmbedTextLen {
		s = s[:maxEmbedTextLen]
	}
	return s
}

// semanticSnippet renders a compact one-line snippet for search results.
func semanticSnippet(content string) string {
	s := strings.Join(strings.Fields(content), " ")
	if len(s) > 200 {
		s = s[:200] + "…"
	}
	return s
}
