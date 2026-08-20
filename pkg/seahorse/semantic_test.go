package seahorse

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sipeed/picoclaw/pkg/tools"
)

// fakeEmbedServer returns deterministic vectors so tests can verify that
// semantically similar texts land near each other. Each distinct word maps to
// a unique basis vector; the final vector is the sum of the words' vectors.
func fakeEmbedServer(t *testing.T) *httptest.Server {
	t.Helper()

	wordVec := func(word string) []float32 {
		switch strings.ToLower(word) {
		case "gpu", "gpus", "a770", "wedge", "reset", "resets":
			return []float32{1, 0, 0, 0}
		case "audio", "sound", "crackle", "crackling", "rtkit":
			return []float32{0, 1, 0, 0}
		case "backup", "restore", "restic", "gdrive":
			return []float32{0, 0, 1, 0}
		case "pihole", "pi-hole", "dns", "blocklist":
			return []float32{0, 0, 0, 1}
		default:
			return []float32{0, 0, 0, 0}
		}
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/embeddings") {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		var req struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		type item struct {
			Object    string    `json:"object"`
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		}
		data := make([]item, 0, len(req.Input))
		for i, text := range req.Input {
			vec := []float32{0, 0, 0, 0}
			for _, w := range strings.Fields(text) {
				wv := wordVec(strings.Trim(w, ".,!?[]():"))
				for j := range vec {
					vec[j] += wv[j]
				}
			}
			data = append(data, item{Object: "embedding", Index: i, Embedding: vec})
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data":  data,
			"model": "fake-embed",
			"usage": map[string]any{"prompt_tokens": 1, "total_tokens": 1},
		})
	}))
	t.Cleanup(server.Close)
	return server
}

func semanticTestConfig(t *testing.T, endpoint string) Config {
	t.Helper()
	return Config{
		DBPath:            filepath.Join(t.TempDir(), "semantic.db"),
		EnableSemantic:    true,
		EmbeddingEndpoint: endpoint + "/v1/embeddings",
		EmbeddingModel:    "fake-embed",
		EmbeddingDim:      4,
		TopK:              5,
		MinScore:          0.1,
	}
}

func TestValidateSemantic(t *testing.T) {
	good := semanticTestConfig(t, "http://x")
	if err := good.ValidateSemantic(); err != nil {
		t.Errorf("good config should validate: %v", err)
	}
	if err := (Config{EnableSemantic: true}).ValidateSemantic(); err == nil {
		t.Error("missing endpoint should error")
	}
	if err := (Config{EnableSemantic: true, EmbeddingEndpoint: "http://x"}).ValidateSemantic(); err == nil {
		t.Error("missing model should error")
	}
	if err := (Config{EnableSemantic: true, EmbeddingEndpoint: "http://x", EmbeddingModel: "m"}).ValidateSemantic(); err == nil {
		t.Error("missing dim should error")
	}
	if err := (Config{EnableSemantic: false}).ValidateSemantic(); err != nil {
		t.Errorf("disabled config should always validate: %v", err)
	}
}

func TestSemanticDisabledByDefault(t *testing.T) {
	dir := t.TempDir()
	eng, err := NewEngine(Config{DBPath: filepath.Join(dir, "short.db")}, nil)
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	defer eng.Close()

	if eng.SemanticEnabled() {
		t.Error("semantic should be disabled by default")
	}
	_, err = eng.GetRetrieval().Semantic(context.Background(), SemanticInput{Query: "anything"})
	if err == nil {
		t.Error("Semantic should error when disabled")
	}
}

func TestSemanticInvalidConfig(t *testing.T) {
	dir := t.TempDir()
	_, err := NewEngine(Config{
		DBPath:            filepath.Join(dir, "short.db"),
		EnableSemantic:    true,
		EmbeddingEndpoint: "", // missing
	}, nil)
	if err == nil {
		t.Fatal("expected error for incomplete semantic config")
	}
}

// waitForEmbedCount polls until the vector store has the expected rows.
func waitForEmbedCount(t *testing.T, eng *Engine, want int) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		n, err := eng.semantic.estore.CountForModel(context.Background())
		if err == nil && n >= want {
			return
		}
		time.Sleep(25 * time.Millisecond)
	}
	got, _ := eng.semantic.estore.CountForModel(context.Background())
	t.Fatalf("timed out waiting for %d embeddings, got %d", want, got)
}

func TestSemanticIngestAndSearch(t *testing.T) {
	server := fakeEmbedServer(t)
	eng, err := NewEngine(semanticTestConfig(t, server.URL), nil)
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	defer eng.Close()
	if !eng.SemanticEnabled() {
		t.Fatal("expected semantic enabled")
	}

	ctx := context.Background()
	_, err = eng.Ingest(ctx, "agent:sem-test", []Message{
		{Role: "user", Content: "the gpu wedges and resets under compute load", TokenCount: 8},
		{Role: "user", Content: "fix the audio crackle by installing rtkit", TokenCount: 8},
	})
	if err != nil {
		t.Fatalf("Ingest: %v", err)
	}
	waitForEmbedCount(t, eng, 2)

	res, err := eng.GetRetrieval().Semantic(ctx, SemanticInput{Query: "gpu reset problems"})
	if err != nil {
		t.Fatalf("Semantic: %v", err)
	}
	if len(res.Messages) == 0 {
		t.Fatalf("expected semantic hits, got none (hint: %s)", res.Hint)
	}
	if !strings.Contains(res.Messages[0].Snippet, "gpu") {
		t.Errorf("top hit should be the gpu message, got %q (rank %.2f)", res.Messages[0].Snippet, res.Messages[0].Rank)
	}
	if res.Messages[0].Rank <= 0 {
		t.Errorf("expected positive cosine rank, got %f", res.Messages[0].Rank)
	}

	// Query for the audio message should rank it top.
	res2, err := eng.GetRetrieval().Semantic(ctx, SemanticInput{Query: "sound crackling audio issue"})
	if err != nil {
		t.Fatalf("Semantic 2: %v", err)
	}
	if len(res2.Messages) == 0 {
		t.Fatalf("expected hits for audio query")
	}
	if !strings.Contains(res2.Messages[0].Snippet, "audio") {
		t.Errorf("top hit should be the audio message, got %q (rank %.2f)", res2.Messages[0].Snippet, res2.Messages[0].Rank)
	}
}

func TestSemanticToolAndExpand(t *testing.T) {
	server := fakeEmbedServer(t)
	eng, err := NewEngine(semanticTestConfig(t, server.URL), nil)
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	defer eng.Close()

	ctx := context.Background()
	_, err = eng.Ingest(ctx, "agent:sem-tool", []Message{
		{Role: "user", Content: "remember to back up with restic every week", TokenCount: 8},
		{Role: "user", Content: "pihole dns blocklists are on swarm1 and swarm2", TokenCount: 8},
	})
	if err != nil {
		t.Fatalf("Ingest: %v", err)
	}
	waitForEmbedCount(t, eng, 2)

	tool := NewSemanticTool(eng.GetRetrieval())
	if tool.Name() != "short_semantic" {
		t.Errorf("Name = %q, want short_semantic", tool.Name())
	}

	// Missing query -> error result
	bad := tool.Execute(ctx, map[string]any{})
	if bad == nil || bad.IsError == false {
		t.Error("missing query should return error result")
	}

	// Happy path
	res := tool.Execute(ctx, map[string]any{"query": "restic restore backup"})
	if res == nil || res.IsError {
		t.Fatalf("execute failed: %v", res)
	}
	if !strings.Contains(res.ForLLM, "restic") {
		t.Errorf("expected backup message in results, got: %s", res.ForLLM)
	}
	var out struct {
		Messages []GrepMessageResult `json:"messages"`
	}
	if err := json.Unmarshal([]byte(res.ForLLM), &out); err != nil {
		t.Fatalf("unmarshal result: %v", err)
	}
	if len(out.Messages) == 0 {
		t.Fatal("expected messages in tool output")
	}

	// Expand the top hit -> full message text
	expanded, err := eng.GetRetrieval().ExpandMessages(ctx, []int64{out.Messages[0].ID})
	if err != nil {
		t.Fatalf("ExpandMessages: %v", err)
	}
	if len(expanded.Messages) == 0 || !strings.Contains(expanded.Messages[0].Content, "restic") {
		t.Errorf("expanded message should contain restic, got %+v", expanded.Messages)
	}
}

func TestSemanticBackfill(t *testing.T) {
	server := fakeEmbedServer(t)
	eng, err := NewEngine(semanticTestConfig(t, server.URL), nil)
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	defer eng.Close()

	ctx := context.Background()
	conv, err := eng.store.GetOrCreateConversation(ctx, "agent:sem-backfill")
	if err != nil {
		t.Fatalf("GetOrCreateConversation: %v", err)
	}
	// Add messages directly to the store so they bypass the ingest queue.
	var ids []int64
	for _, content := range []string{"backup the whole fleet nightly", "dns pihole blocking ads", "gpu wedge reset storm"} {
		m, err := eng.store.AddMessage(ctx, conv.ConversationID, "user", content, 5)
		if err != nil {
			t.Fatalf("AddMessage: %v", err)
		}
		ids = append(ids, m.ID)
	}

	before, err := eng.semantic.estore.CountForModel(ctx)
	if err != nil {
		t.Fatalf("CountForModel before: %v", err)
	}
	if before != 0 {
		t.Fatalf("expected 0 embeddings before backfill, got %d", before)
	}

	n, err := eng.SemanticBackfill(ctx, 2)
	if err != nil {
		t.Fatalf("SemanticBackfill: %v", err)
	}
	if n != 3 {
		t.Errorf("backfill embedded %d, want 3", n)
	}

	after, _ := eng.semantic.estore.CountForModel(ctx)
	if after != 3 {
		t.Errorf("after backfill count = %d, want 3", after)
	}

	// Second run should be a no-op (all embedded).
	n2, err := eng.SemanticBackfill(ctx, 2)
	if err != nil {
		t.Fatalf("SemanticBackfill 2: %v", err)
	}
	if n2 != 0 {
		t.Errorf("second backfill embedded %d, want 0", n2)
	}

	// Semantic search should now find the backup message.
	res, err := eng.GetRetrieval().Semantic(ctx, SemanticInput{Query: "restic restore backup"})
	if err != nil {
		t.Fatalf("Semantic after backfill: %v", err)
	}
	if len(res.Messages) == 0 || !strings.Contains(res.Messages[0].Snippet, "backup") {
		t.Errorf("expected backup message top, got %+v", res.Messages)
	}
}

func TestMessageEmbedText(t *testing.T) {
	msg := &Message{
		Content: "do the thing",
		Parts: []MessagePart{
			{Type: "tool_use", Name: "exec", Arguments: `{"command":"df -h"}`},
			{Type: "tool_result", ToolCallID: "call_1", Text: "disk is full"},
			{Type: "media", MediaURI: "/tmp/photo.png"},
		},
	}
	text := messageEmbedText(msg)
	for _, want := range []string{"do the thing", "tool: exec", "df -h", "result: disk is full", "media: /tmp/photo.png"} {
		if !strings.Contains(text, want) {
			t.Errorf("embed text missing %q: %s", want, text)
		}
	}

	if got := messageEmbedText(&Message{}); got != "[empty message]" {
		t.Errorf("empty message text = %q, want [empty message]", got)
	}

	long := &Message{Content: strings.Repeat("x", maxEmbedTextLen+100)}
	if len(messageEmbedText(long)) > maxEmbedTextLen {
		t.Error("embed text should be capped at maxEmbedTextLen")
	}
}

func TestSemanticConversationScoping(t *testing.T) {
	server := fakeEmbedServer(t)
	eng, err := NewEngine(semanticTestConfig(t, server.URL), nil)
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	defer eng.Close()

	ctx := context.Background()
	for _, sk := range []string{"agent:scope-a", "agent:scope-b"} {
		_, err := eng.Ingest(ctx, sk, []Message{{Role: "user", Content: "the gpu wedges on the a770", TokenCount: 6}})
		if err != nil {
			t.Fatalf("Ingest %s: %v", sk, err)
		}
	}
	waitForEmbedCount(t, eng, 2)

	convA, err := eng.store.GetConversationBySessionKey(ctx, "agent:scope-a")
	if err != nil || convA == nil {
		t.Fatalf("get conv A: %v", err)
	}

	// Scoped to conversation A -> exactly 1 hit.
	res, err := eng.GetRetrieval().Semantic(ctx, SemanticInput{
		Query:          "gpu reset",
		ConversationID: convA.ConversationID,
	})
	if err != nil {
		t.Fatalf("Semantic scoped: %v", err)
	}
	if len(res.Messages) != 1 {
		t.Fatalf("expected 1 scoped hit, got %d", len(res.Messages))
	}
	if res.Messages[0].ConversationID != convA.ConversationID {
		t.Errorf("hit belongs to wrong conversation: %d", res.Messages[0].ConversationID)
	}

	// All conversations -> 2 hits.
	resAll, err := eng.GetRetrieval().Semantic(ctx, SemanticInput{Query: "gpu reset", AllConversations: true})
	if err != nil {
		t.Fatalf("Semantic all: %v", err)
	}
	if len(resAll.Messages) != 2 {
		t.Fatalf("expected 2 all-conversation hits, got %d", len(resAll.Messages))
	}
}

func TestSemanticToolScopesToSessionWhenRequested(t *testing.T) {
	server := fakeEmbedServer(t)
	eng, err := NewEngine(semanticTestConfig(t, server.URL), nil)
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	defer eng.Close()

	ctx := context.Background()
	for _, sk := range []string{"agent:tool-scope-a", "agent:tool-scope-b"} {
		_, err := eng.Ingest(ctx, sk, []Message{{Role: "user", Content: "pihole dns blocklist", TokenCount: 6}})
		if err != nil {
			t.Fatalf("Ingest %s: %v", sk, err)
		}
	}
	waitForEmbedCount(t, eng, 2)

	// Simulate a turn in conversation A.
	convA, _ := eng.store.GetConversationBySessionKey(ctx, "agent:tool-scope-a")
	ctxWithSession := tools.WithToolSessionContext(ctx, "default", "agent:tool-scope-a", nil)

	tool := NewSemanticTool(eng.GetRetrieval())
	res := tool.Execute(ctxWithSession, map[string]any{"query": "dns", "all_conversations": false})
	if res == nil || res.IsError {
		t.Fatalf("execute failed: %v", res)
	}
	var out struct {
		Messages []GrepMessageResult `json:"messages"`
	}
	if err := json.Unmarshal([]byte(res.ForLLM), &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(out.Messages) != 1 {
		t.Fatalf("expected 1 hit scoped to current session, got %d", len(out.Messages))
	}
	if out.Messages[0].ConversationID != convA.ConversationID {
		t.Errorf("hit conversation %d, want %d", out.Messages[0].ConversationID, convA.ConversationID)
	}

	// Default (all_conversations true) -> both.
	resAll := tool.Execute(ctxWithSession, map[string]any{"query": "dns"})
	if resAll == nil || resAll.IsError {
		t.Fatalf("execute all failed: %v", resAll)
	}
	var outAll struct {
		Messages []GrepMessageResult `json:"messages"`
	}
	if err := json.Unmarshal([]byte(resAll.ForLLM), &outAll); err != nil {
		t.Fatalf("unmarshal all: %v", err)
	}
	if len(outAll.Messages) != 2 {
		t.Fatalf("expected 2 all-conversation hits, got %d", len(outAll.Messages))
	}
}

func TestSemanticNormalizeEndpoint(t *testing.T) {
	cases := []struct{ in, want string }{
		{"http://100.88.1.92:18084/v1/embeddings", "http://100.88.1.92:18084/v1"},
		{"http://100.88.1.92:18084/v1/embeddings/", "http://100.88.1.92:18084/v1"},
		{"http://100.88.1.92:18084/v1", "http://100.88.1.92:18084/v1"},
		{"  http://x:18084/v1/embeddings  ", "http://x:18084/v1"},
		{"", ""},
	}
	for _, c := range cases {
		if got := normalizeEmbeddingEndpoint(c.in); got != c.want {
			t.Errorf("normalizeEmbeddingEndpoint(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestSQLiteDSN(t *testing.T) {
	cases := []struct{ in, want string }{
		{"/tmp/x.db", "/tmp/x.db?_pragma=foreign_keys(1)"},
		{"file:/tmp/x.db?mode=rwc", "file:/tmp/x.db?mode=rwc&_pragma=foreign_keys(1)"},
		{":memory:", ":memory:"},
		{"/tmp/x.db?_pragma=foreign_keys(1)", "/tmp/x.db?_pragma=foreign_keys(1)"},
	}
	for _, c := range cases {
		if got := sqliteDSN(c.in); got != c.want {
			t.Errorf("sqliteDSN(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}
