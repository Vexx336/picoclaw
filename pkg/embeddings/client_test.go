package embeddings

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func makeEmbedResponse(t *testing.T, vecs [][]float32, encodeB64 bool) map[string]any {
	t.Helper()
	data := make([]any, 0, len(vecs))
	for i, v := range vecs {
		item := map[string]any{"object": "embedding", "index": i}
		if encodeB64 {
			blob := make([]byte, len(v)*4)
			for j, x := range v {
				binary.LittleEndian.PutUint32(blob[j*4:], math.Float32bits(x))
			}
			item["embedding_base64"] = base64.StdEncoding.EncodeToString(blob)
		} else {
			item["embedding"] = v
		}
		data = append(data, item)
	}
	return map[string]any{
		"data":  data,
		"model": "jina-embeddings-v3",
		"usage": map[string]any{"prompt_tokens": 5, "total_tokens": 5},
	}
}

func TestEmbedFloat(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		if req["encoding_format"] != "float" {
			t.Errorf("expected encoding_format float, got %v", req["encoding_format"])
		}
		input, _ := req["input"].([]any)
		if len(input) == 0 {
			t.Fatalf("expected at least 1 input")
		}
		vecs := make([][]float32, 0, len(input))
		for i := range input {
			vecs = append(vecs, []float32{float32(i) + 0.1, float32(i) + 0.2, float32(i) + 0.3})
		}
		_ = json.NewEncoder(w).Encode(makeEmbedResponse(t, vecs, false))
	}))
	defer srv.Close()

	c := NewClient(srv.URL+"/v1", "jina-embeddings-v3", 3)
	vecs, err := c.EmbedBatch(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatal(err)
	}
	if len(vecs) != 2 || len(vecs[0]) != 3 || vecs[0][0] != 0.1 || vecs[1][2] != 1.3 {
		t.Fatalf("unexpected vectors: %#v", vecs)
	}

	single, err := c.Embed(context.Background(), "hello")
	if err != nil {
		t.Fatal(err)
	}
	if len(single) != 3 || single[0] != 0.1 {
		t.Fatalf("single embed = %#v, want [0.1 0.2 0.3]", single)
	}
}

func TestEmbedBase64(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(makeEmbedResponse(t, [][]float32{{1.5, -2.5, 3.25}}, true))
	}))
	defer srv.Close()

	c := NewClient(srv.URL+"/v1", "m", 3)
	vecs, err := c.EmbedBatch(context.Background(), []string{"hi"})
	if err != nil {
		t.Fatal(err)
	}
	if vecs[0][0] != 1.5 || vecs[0][1] != -2.5 || vecs[0][2] != 3.25 {
		t.Fatalf("unexpected decoded vector: %#v", vecs[0])
	}
}

func TestEmbedSkipsEmptyTexts(t *testing.T) {
	c := NewClient("http://unused", "m", 3)
	if _, err := c.Embed(context.Background(), ""); !errors.Is(err, ErrEmptyInput) {
		t.Fatalf("expected ErrEmptyInput, got %v", err)
	}
	if _, err := c.EmbedBatch(context.Background(), []string{"", "  "}); !errors.Is(err, ErrEmptyInput) {
		t.Fatalf("expected ErrEmptyInput for all-empty batch, got %v", err)
	}
}

func TestEmbedDimMismatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(makeEmbedResponse(t, [][]float32{{0.1, 0.2, 0.3}}, false))
	}))
	defer srv.Close()

	c := NewClient(srv.URL+"/v1", "m", 5)
	_, err := c.Embed(context.Background(), "hi")
	if !errors.Is(err, ErrDimMismatch) {
		t.Fatalf("expected ErrDimMismatch, got %v", err)
	}
}

func TestEmbedRetryOnTransient(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&calls, 1) < 3 {
			http.Error(w, "boom", http.StatusInternalServerError)
			return
		}
		_ = json.NewEncoder(w).Encode(makeEmbedResponse(t, [][]float32{{0.1, 0.2, 0.3}}, false))
	}))
	defer srv.Close()

	c := NewClient(srv.URL+"/v1", "m", 3)
	c.MaxRetries = 3
	c.RetryDelay = time.Millisecond

	if _, err := c.Embed(context.Background(), "hi"); err != nil {
		t.Fatalf("expected success after retries, got %v", err)
	}
	if got := atomic.LoadInt32(&calls); got != 3 {
		t.Fatalf("expected 3 calls, got %d", got)
	}
}

func TestEmbedNoRetryOnBadRequest(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		http.Error(w, "bad input", http.StatusBadRequest)
	}))
	defer srv.Close()

	c := NewClient(srv.URL+"/v1", "m", 3)
	c.MaxRetries = 5
	c.RetryDelay = time.Millisecond

	if _, err := c.Embed(context.Background(), "hi"); err == nil {
		t.Fatal("expected error")
	}
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Fatalf("expected no retry on 400, got %d calls", got)
	}
}

func TestEmbedContextCancel(t *testing.T) {
	done := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-done // hold the connection
	}))
	defer srv.Close()
	defer close(done)

	c := NewClient(srv.URL+"/v1", "m", 3)
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := c.Embed(ctx, "hi")
	if err == nil {
		t.Fatal("expected context error")
	}
	if !strings.Contains(err.Error(), "context") && !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("unexpected error: %v", err)
	}
}
