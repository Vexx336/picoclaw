package seahorse

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

// TestLiveSemanticSmoke exercises the full M2 semantic path against the real
// embedding server on skynet (:18084, jina-embeddings-v3). Skipped unless
// EMB_LIVE=1. Hermetic by default.
//
// Run with:
//
//	EMB_LIVE=1 go test ./pkg/seahorse/ -run TestLiveSemanticSmoke -v
func TestLiveSemanticSmoke(t *testing.T) {
	if os.Getenv("EMB_LIVE") != "1" {
		t.Skip("set EMB_LIVE=1 to run live test against skynet")
	}
	endpoint := os.Getenv("EMB_ENDPOINT")
	if endpoint == "" {
		endpoint = "http://100.88.1.92:18084/v1/embeddings"
	}

	cfg := Config{
		DBPath:            t.TempDir() + "/live-semantic.db",
		EnableSemantic:    true,
		EmbeddingEndpoint: endpoint,
		EmbeddingModel:    "jina-embeddings-v3",
		EmbeddingDim:      1024,
		TopK:              3,
		MinScore:          0.3,
	}
	eng, err := NewEngine(cfg, nil)
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	defer eng.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	_, err = eng.Ingest(ctx, "agent:live-sem-smoke", []Message{
		{Role: "user", Content: "the arc a770 gpu wedges and resets under any compute load on skynet", TokenCount: 15},
		{Role: "user", Content: "pi-hole dns blocklists run on swarm1 and swarm2 with caddy auth", TokenCount: 15},
		{Role: "user", Content: "p70 uses xanmod kernel with nvidia driver 580 for the quadro m3000m", TokenCount: 15},
	})
	if err != nil {
		t.Fatalf("Ingest: %v", err)
	}
	waitForEmbedCount(t, eng, 3)

	queries := []struct {
		q        string
		wantWord string
	}{
		{"gpu keeps resetting under load", "a770"},
		{"dns ad blocking admin password", "pi-hole"},
		{"which nvidia driver for the laptop", "nvidia"},
	}
	for _, q := range queries {
		res, err := eng.GetRetrieval().Semantic(ctx, SemanticInput{Query: q.q})
		if err != nil {
			t.Fatalf("Semantic(%q): %v", q.q, err)
		}
		if len(res.Messages) == 0 {
			t.Fatalf("Semantic(%q): no hits (hint: %s)", q.q, res.Hint)
		}
		t.Logf("query %q -> top: %q (rank %.3f)", q.q, res.Messages[0].Snippet, res.Messages[0].Rank)
		if !strings.Contains(res.Messages[0].Snippet, q.wantWord) {
			t.Errorf("query %q: top hit %q does not contain %q", q.q, res.Messages[0].Snippet, q.wantWord)
		}
	}
}
