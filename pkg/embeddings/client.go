// Package embeddings provides a pure-Go client for llama.cpp's
// OpenAI-compatible /v1/embeddings endpoint, plus an in-memory flat index
// and a SQLite-backed vector store. It is used by the seahorse context
// manager to add semantic retrieval over chat history.
//
// No CGO and no external runtime dependencies: the HTTP client uses only
// net/http, and the store rides on the same modernc.org/sqlite driver the
// seahorse engine already uses.
package embeddings

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"time"
)

// Client talks to a llama.cpp llama-server running with --embedding.
//
// The endpoint is OpenAI-compatible:
//
//	POST {BaseURL}/v1/embeddings
//	{"input": ["text"], "model": "jina-embeddings-v3", "encoding_format": "float"}
//
// Response:
//
//	{"data":[{"object":"embedding","index":0,"embedding":[...]}],"model":"...","usage":{"prompt_tokens":N,"total_tokens":N}}
type Client struct {
	BaseURL    string        // e.g. "http://100.88.1.92:18084/v1"
	Model      string        // model name passed to llama.cpp (informational; server may ignore)
	Dim        int           // expected embedding dimension; mismatches are an error
	HTTPClient *http.Client  // optional; defaults to a 60s-timeout client
	MaxRetries int           // additional attempts after the first (default 1)
	RetryDelay time.Duration // delay between retries (default 250ms)
}

// DefaultTimeout is applied to the HTTP client when none is provided.
const DefaultTimeout = 60 * time.Second

// DefaultRetryDelay is the pause between retry attempts.
const DefaultRetryDelay = 250 * time.Millisecond

// ErrDimMismatch is returned when the server returns vectors of a different
// dimension than the configured one (e.g. after a model swap on the server).
var ErrDimMismatch = errors.New("embeddings: dimension mismatch")

// ErrEmptyInput is returned when Embed/EmbedBatch receives no texts.
var ErrEmptyInput = errors.New("embeddings: empty input")

// NewClient builds a client with defaults applied.
func NewClient(baseURL, model string, dim int) *Client {
	return &Client{
		BaseURL:    strings.TrimRight(baseURL, "/"),
		Model:      model,
		Dim:        dim,
		HTTPClient: &http.Client{Timeout: DefaultTimeout},
		MaxRetries: 1,
		RetryDelay: DefaultRetryDelay,
	}
}

// Embed returns the embedding vector for a single text.
func (c *Client) Embed(ctx context.Context, text string) ([]float32, error) {
	if strings.TrimSpace(text) == "" {
		return nil, ErrEmptyInput
	}
	vecs, err := c.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return vecs[0], nil
}

// EmbedBatch returns one embedding vector per input text, in the same order.
//
// llama.cpp accepts either a string or an array of strings as "input"; batch
// requests amortize per-request overhead on the server. The caller must keep
// batch sizes modest (llama.cpp slots are limited by --parallel).
func (c *Client) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	clean := make([]string, 0, len(texts))
	for _, t := range texts {
		if strings.TrimSpace(t) == "" {
			continue
		}
		clean = append(clean, t)
	}
	if len(clean) == 0 {
		return nil, ErrEmptyInput
	}

	body := map[string]any{
		"input":           clean,
		"model":           c.Model,
		"encoding_format": "float",
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("embeddings: marshal request: %w", err)
	}

	url := c.BaseURL + "/embeddings"
	var lastErr error
	attempts := c.MaxRetries + 1
	if attempts < 1 {
		attempts = 1
	}
	for attempt := 0; attempt < attempts; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(c.retryDelay()):
			}
		}

		vecs, err := c.doOnce(ctx, url, payload)
		if err == nil {
			return vecs, nil
		}
		lastErr = err
		// Do not retry on dimension mismatch or bad request — it won't fix itself.
		if errors.Is(err, ErrDimMismatch) || isBadRequest(err) {
			return nil, err
		}
	}
	return nil, fmt.Errorf("embeddings: after %d attempts: %w", attempts, lastErr)
}

func (c *Client) doOnce(ctx context.Context, url string, payload []byte) ([][]float32, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient().Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("embeddings: server returned %s: %s", resp.Status, strings.TrimSpace(string(msg)))
	}

	var out embedResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, 64<<20)).Decode(&out); err != nil {
		return nil, fmt.Errorf("embeddings: decode response: %w", err)
	}
	if len(out.Data) == 0 {
		return nil, errors.New("embeddings: empty data array in response")
	}

	vecs := make([][]float32, len(out.Data))
	for i, item := range out.Data {
		v, err := decodeVector(item.Embedding, item.EmbeddingB64, c.Dim)
		if err != nil {
			return nil, err
		}
		vecs[i] = v
	}
	return vecs, nil
}

// embedResponse mirrors the OpenAI-compatible response shape. llama.cpp may
// return either a plain float array ("encoding_format": "float") or a base64
// string ("encoding_format": "base64"); both are handled.
type embedResponse struct {
	Data []struct {
		Object       string          `json:"object"`
		Index        int             `json:"index"`
		Embedding    []float32       `json:"embedding,omitempty"`
		EmbeddingB64 json.RawMessage `json:"embedding_base64,omitempty"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

func (c *Client) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return &http.Client{Timeout: DefaultTimeout}
}

func (c *Client) retryDelay() time.Duration {
	if c.RetryDelay > 0 {
		return c.RetryDelay
	}
	return DefaultRetryDelay
}

func isBadRequest(err error) bool {
	return err != nil && (strings.Contains(err.Error(), "400 Bad Request") ||
		strings.Contains(err.Error(), "422 Unprocessable"))
}

// decodeVector decodes either a float array or a base64 blob into []float32
// and verifies it matches the expected dimension.
func decodeVector(floats []float32, b64 json.RawMessage, wantDim int) ([]float32, error) {
	var vec []float32
	switch {
	case len(floats) > 0:
		vec = floats
	case len(b64) > 0:
		raw, err := base64.StdEncoding.DecodeString(strings.Trim(string(b64), `"`))
		if err != nil {
			return nil, fmt.Errorf("embeddings: base64 decode: %w", err)
		}
		if len(raw)%4 != 0 {
			return nil, errors.New("embeddings: base64 vector length not a multiple of 4")
		}
		vec = make([]float32, len(raw)/4)
		for i := range vec {
			bits := binary.LittleEndian.Uint32(raw[i*4 : i*4+4])
			vec[i] = math.Float32frombits(bits)
		}
	default:
		return nil, errors.New("embeddings: vector missing in response")
	}

	if wantDim > 0 && len(vec) != wantDim {
		return nil, fmt.Errorf("%w: got %d, want %d (model changed on server?)",
			ErrDimMismatch, len(vec), wantDim)
	}
	return vec, nil
}
