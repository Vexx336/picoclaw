package embeddings

import (
	"context"
	"os"
	"testing"
	"time"
)

// TestLiveSkynet is a live smoke test against the real embedding server on
// skynet (:18084, jina-embeddings-v3). It is skipped unless EMB_LIVE=1 is
// set, so `go test ./pkg/embeddings/` stays hermetic.
//
// Run with:
//
//	EMB_LIVE=1 go test ./pkg/embeddings/ -run TestLiveSkynet -v
func TestLiveSkynet(t *testing.T) {
	if os.Getenv("EMB_LIVE") != "1" {
		t.Skip("set EMB_LIVE=1 to run live test against skynet")
	}
	endpoint := os.Getenv("EMB_ENDPOINT")
	if endpoint == "" {
		endpoint = "http://100.88.1.92:18084/v1"
	}

	c := NewClient(endpoint, "jina-embeddings-v3", 1024)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	start := time.Now()
	vecs, err := c.EmbedBatch(ctx, []string{
		"the arc a770 wedges under compute load",
		"pi-hole dns runs on swarm1 and swarm2",
		"the p70 uses nvidia driver 580",
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("batch embed took %s, got %d vectors of dim %d",
		time.Since(start).Round(time.Millisecond), len(vecs), len(vecs[0]))

	if len(vecs) != 3 || len(vecs[0]) != 1024 {
		t.Fatalf("unexpected shapes: %d vectors, dim %d", len(vecs), len(vecs[0]))
	}

	// Search against the batch with a related query; the arc message should win.
	ix := NewIndex(1024)
	for i, v := range vecs {
		ix.Add(int64(i+1), v)
	}
	qvec, err := c.Embed(ctx, "gpu wedges and resets under compute load")
	if err != nil {
		t.Fatal(err)
	}
	hits := ix.Search(qvec, 3, 0)
	if len(hits) == 0 {
		t.Fatal("no hits from live search")
	}
	t.Logf("live hits: %#v", hits)
	if hits[0].MessageID != 1 {
		t.Errorf("expected arc message (id 1) to be top hit, got %d", hits[0].MessageID)
	}
}
