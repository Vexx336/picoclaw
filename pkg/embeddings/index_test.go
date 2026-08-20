package embeddings

import (
	"math"
	"testing"
)

func TestIndexAddAndLen(t *testing.T) {
	ix := NewIndex(3)
	if ix.Len() != 0 {
		t.Fatal("new index should be empty")
	}
	ix.Add(1, []float32{1, 0, 0})
	ix.Add(2, []float32{0, 1, 0})
	if ix.Len() != 2 {
		t.Fatalf("len = %d, want 2", ix.Len())
	}
}

func TestIndexReplacesOnSameID(t *testing.T) {
	ix := NewIndex(3)
	ix.Add(7, []float32{1, 0, 0})
	ix.Add(7, []float32{0, 1, 0})
	if ix.Len() != 1 {
		t.Fatalf("len = %d, want 1 (replace)", ix.Len())
	}
	hits := ix.Search([]float32{0, 1, 0}, 1, 0)
	if len(hits) != 1 || hits[0].MessageID != 7 {
		t.Fatalf("expected replaced vector for id 7, got %#v", hits)
	}
}

func TestIndexDropsDimMismatch(t *testing.T) {
	ix := NewIndex(3)
	ix.Add(1, []float32{1, 0}) // wrong dim — dropped
	if ix.Len() != 0 {
		t.Fatalf("len = %d, want 0", ix.Len())
	}
}

func TestIndexSearchOrdersBySimilarity(t *testing.T) {
	ix := NewIndex(4)
	ix.Add(100, []float32{1, 0, 0, 0})
	ix.Add(200, []float32{0.9, 0.1, 0, 0})
	ix.Add(300, []float32{0, 0, 1, 0})

	hits := ix.Search([]float32{1, 0, 0, 0}, 3, 0)
	if len(hits) != 3 {
		t.Fatalf("got %d hits, want 3", len(hits))
	}
	if hits[0].MessageID != 100 {
		t.Errorf("top hit = %d, want 100", hits[0].MessageID)
	}
	if hits[1].MessageID != 200 {
		t.Errorf("second hit = %d, want 200", hits[1].MessageID)
	}
	// Identical vectors should have cosine ≈ 1.0
	if math.Abs(hits[0].Score-1.0) > 1e-6 {
		t.Errorf("score = %v, want ~1.0", hits[0].Score)
	}
}

func TestIndexTopK(t *testing.T) {
	ix := NewIndex(2)
	for i := 1; i <= 10; i++ {
		v := float32(i) / 10
		ix.Add(int64(i), []float32{v, 1 - v})
	}
	hits := ix.Search([]float32{0.95, 0.05}, 3, 0)
	if len(hits) != 3 {
		t.Fatalf("got %d hits, want 3", len(hits))
	}
	// 10 (v=1.0) should be the top match, then 9, then 8.
	if hits[0].MessageID != 10 || hits[1].MessageID != 9 || hits[2].MessageID != 8 {
		t.Fatalf("unexpected top-3 order: %#v", hits)
	}
}

func TestIndexMinScoreFilter(t *testing.T) {
	ix := NewIndex(2)
	ix.Add(1, []float32{1, 0})
	ix.Add(2, []float32{0, 1})

	hits := ix.Search([]float32{1, 0}, 5, 0)
	if len(hits) != 2 {
		t.Fatalf("no-filter hits = %d, want 2", len(hits))
	}
	// Perpendicular vector {0,1} scores 0.0 against query {1,0}.
	hits = ix.Search([]float32{1, 0}, 5, 0.01)
	if len(hits) != 1 || hits[0].MessageID != 1 {
		t.Fatalf("minScore 0.01 hits = %#v, want only id 1", hits)
	}
	hits = ix.Search([]float32{1, 0}, 5, 0.9)
	if len(hits) != 1 || hits[0].MessageID != 1 {
		t.Fatalf("minScore 0.9 hits = %#v, want only id 1", hits)
	}
}

func TestIndexEmpty(t *testing.T) {
	ix := NewIndex(3)
	if hits := ix.Search([]float32{1, 0, 0}, 5, 0); hits != nil {
		t.Fatalf("empty index should return nil, got %#v", hits)
	}
}

func TestIndexRemove(t *testing.T) {
	ix := NewIndex(3)
	ix.Add(1, []float32{1, 0, 0})
	ix.Add(2, []float32{0, 1, 0})
	ix.Remove(1)
	if ix.Len() != 1 {
		t.Fatalf("len = %d, want 1", ix.Len())
	}
	hits := ix.Search([]float32{1, 0, 0}, 5, 0.01)
	for _, h := range hits {
		if h.MessageID == 1 {
			t.Fatalf("removed id 1 should not be found, got %#v", hits)
		}
	}
}

func TestIndexSearchDoesNotMutateQuery(t *testing.T) {
	ix := NewIndex(2)
	ix.Add(1, []float32{3, 4}) // norm = 5
	q := []float32{1, 0}
	_ = ix.Search(q, 5, 0)
	if q[0] != 1 || q[1] != 0 {
		t.Fatalf("query was mutated: %#v", q)
	}
}

func TestIndexZeroVector(t *testing.T) {
	ix := NewIndex(2)
	ix.Add(1, []float32{0, 0})
	ix.Add(2, []float32{1, 0})
	hits := ix.Search([]float32{1, 0}, 5, 0)
	if len(hits) != 2 {
		t.Fatalf("expected both vectors to remain searchable, got %#v", hits)
	}
	// Zero vector has no direction; its score should be 0, not NaN.
	for _, h := range hits {
		if math.IsNaN(h.Score) {
			t.Fatalf("NaN score for %#v", hits)
		}
	}
}
