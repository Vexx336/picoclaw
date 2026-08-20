package embeddings

import (
	"math"
	"sort"
)

// Hit is a single nearest-neighbor result.
type Hit struct {
	MessageID int64   `json:"messageId"`
	Score     float64 `json:"score"` // cosine similarity, higher is better (1.0 = identical)
}

// Index is a flat in-memory nearest-neighbor index over message vectors.
//
// Flat brute-force is intentional: at ~27k messages × 1024 dims a query is a
// few million multiply-adds (single-digit milliseconds on any modern CPU).
// HNSW is only worth adding if index load or query time ever becomes a
// problem (see proposal §M4).
type Index struct {
	dim    int
	ids    []int64
	normed [][]float32 // pre-normalized vectors
}

// NewIndex creates an empty index with a fixed dimension.
func NewIndex(dim int) *Index {
	return &Index{dim: dim}
}

// Len returns the number of vectors in the index.
func (ix *Index) Len() int { return len(ix.ids) }

// Dim returns the vector dimension of the index.
func (ix *Index) Dim() int { return ix.dim }

// Add inserts (or replaces) a message vector. Vectors are normalized once at
// insert time so Search only needs dot products.
func (ix *Index) Add(messageID int64, vec []float32) {
	if len(vec) != ix.dim {
		return // dimension mismatch rows are dropped; store layer logs them
	}
	normed := make([]float32, len(vec))
	copy(normed, vec)
	normalizeInPlace(normed)

	for i, id := range ix.ids {
		if id == messageID {
			ix.normed[i] = normed
			return
		}
	}
	ix.ids = append(ix.ids, messageID)
	ix.normed = append(ix.normed, normed)
}

// Remove deletes a message vector. Removing is O(n); callers should batch
// deletes or rebuild the index when large fractions change.
func (ix *Index) Remove(messageID int64) {
	for i, id := range ix.ids {
		if id == messageID {
			ix.ids = append(ix.ids[:i], ix.ids[i+1:]...)
			ix.normed = append(ix.normed[:i], ix.normed[i+1:]...)
			return
		}
	}
}

// Search returns the top-k nearest neighbors by cosine similarity. The query
// vector is normalized internally and not modified.
//
// A score of 0 means "no vectors in index". minScore filters out weak hits;
// pass 0 to skip filtering.
func (ix *Index) Search(query []float32, topK int, minScore float64) []Hit {
	if ix.Len() == 0 || len(query) != ix.dim {
		return nil
	}
	q := make([]float32, len(query))
	copy(q, query)
	normalizeInPlace(q)

	type scored struct {
		id    int64
		score float64
	}
	all := make([]scored, 0, ix.Len())
	for i, id := range ix.ids {
		s := dot(q, ix.normed[i])
		if s < minScore {
			continue
		}
		all = append(all, scored{id: id, score: s})
	}
	if len(all) == 0 {
		return nil
	}

	sort.Slice(all, func(i, j int) bool {
		return all[i].score > all[j].score
	})
	if topK > 0 && len(all) > topK {
		all = all[:topK]
	}

	hits := make([]Hit, len(all))
	for i, s := range all {
		hits[i] = Hit{MessageID: s.id, Score: s.score}
	}
	return hits
}

// normalizeInPlace scales vec to unit length. Zero vectors stay zero.
func normalizeInPlace(vec []float32) {
	var sum float64
	for _, v := range vec {
		sum += float64(v) * float64(v)
	}
	norm := math.Sqrt(sum)
	if norm == 0 || math.IsNaN(norm) || math.IsInf(norm, 0) {
		return
	}
	inv := float32(1.0 / norm)
	for i := range vec {
		vec[i] *= inv
	}
}

// dot computes the dot product of two equal-length vectors.
func dot(a, b []float32) float64 {
	var sum float64
	for i := range a {
		sum += float64(a[i]) * float64(b[i])
	}
	return sum
}
