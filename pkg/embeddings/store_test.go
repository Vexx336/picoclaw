package embeddings

import (
	"context"
	"database/sql"
	"errors"
	"hash/fnv"
	"path/filepath"
	"strings"
	"testing"

	_ "modernc.org/sqlite"
)

func openTestDB(t *testing.T) *sql.DB {
	t.Helper()
	path := filepath.Join(t.TempDir(), "embeddings_test.db")
	// _pragma=foreign_keys(1) mirrors how the seahorse engine should enable
	// FK enforcement when sharing the DB (per-connection PRAGMA alone does
	// not survive pool reuse).
	db, err := sql.Open("sqlite", "file:"+path+"?_pragma=foreign_keys(1)")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	return db
}

// createMessagesTable creates the minimal messages table the embeddings
// schema references via ON DELETE CASCADE (in real usage seahorse's
// runSchema creates it first).
func createMessagesTable(t *testing.T, db *sql.DB) {
	t.Helper()
	if _, err := db.Exec(`CREATE TABLE messages (
		message_id INTEGER PRIMARY KEY AUTOINCREMENT,
		conversation_id INTEGER NOT NULL DEFAULT 1,
		role TEXT NOT NULL DEFAULT '',
		content TEXT NOT NULL DEFAULT '',
		model_name TEXT NOT NULL DEFAULT '',
		reasoning_content TEXT NOT NULL DEFAULT '',
		token_count INTEGER NOT NULL DEFAULT 0,
		created_at TEXT NOT NULL DEFAULT (datetime('now'))
	)`); err != nil {
		t.Fatal(err)
	}
}

func TestStoreMigrateIdempotent(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()

	s1, err := NewStore(ctx, db, "model-a", 3)
	if err != nil {
		t.Fatal(err)
	}
	// Second NewStore on the same DB must not error (CREATE IF NOT EXISTS).
	s2, err := NewStore(ctx, db, "model-a", 3)
	if err != nil {
		t.Fatal(err)
	}
	if s1.Model() != "model-a" || s2.Dim() != 3 {
		t.Fatalf("unexpected store config: %#v %#v", s1, s2)
	}
}

func TestStoreUpsertRoundTrip(t *testing.T) {
	db := openTestDB(t)
	createMessagesTable(t, db)
	ctx := context.Background()

	s, err := NewStore(ctx, db, "jina-embeddings-v3", 3)
	if err != nil {
		t.Fatal(err)
	}
	// Insert two messages so CASCADE + pending queries make sense.
	if _, err := db.Exec(`INSERT INTO messages (content) VALUES ('alpha'), ('beta')`); err != nil {
		t.Fatal(err)
	}

	if err := s.Upsert(ctx, 1, []float32{0.1, 0.2, 0.3}); err != nil {
		t.Fatal(err)
	}
	if err := s.Upsert(ctx, 2, []float32{0.4, 0.5, 0.6}); err != nil {
		t.Fatal(err)
	}

	n, err := s.Count(ctx)
	if err != nil || n != 2 {
		t.Fatalf("count = %d, %v; want 2", n, err)
	}

	ix, err := s.LoadIndex(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if ix.Len() != 2 {
		t.Fatalf("index len = %d, want 2", ix.Len())
	}

	// The second message's vector should dominate a query near it.
	hits := ix.Search([]float32{0.41, 0.51, 0.61}, 1, 0)
	if len(hits) != 1 || hits[0].MessageID != 2 {
		t.Fatalf("expected message 2, got %#v", hits)
	}
}

func TestStoreUpsertReplaces(t *testing.T) {
	db := openTestDB(t)
	createMessagesTable(t, db)
	ctx := context.Background()

	s, err := NewStore(ctx, db, "m", 3)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`INSERT INTO messages (content) VALUES ('x')`); err != nil {
		t.Fatal(err)
	}

	if err := s.Upsert(ctx, 1, []float32{1, 0, 0}); err != nil {
		t.Fatal(err)
	}
	if err := s.Upsert(ctx, 1, []float32{0, 1, 0}); err != nil {
		t.Fatal(err)
	}

	n, _ := s.Count(ctx)
	if n != 1 {
		t.Fatalf("count = %d, want 1 (replaced)", n)
	}
	ix, _ := s.LoadIndex(ctx)
	hits := ix.Search([]float32{0, 1, 0}, 1, 0)
	if len(hits) != 1 || hits[0].MessageID != 1 {
		t.Fatalf("expected replaced vector, got %#v", hits)
	}
}

func TestStoreUpsertDimMismatch(t *testing.T) {
	db := openTestDB(t)
	createMessagesTable(t, db)
	ctx := context.Background()

	s, err := NewStore(ctx, db, "m", 3)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`INSERT INTO messages (content) VALUES ('x')`); err != nil {
		t.Fatal(err)
	}
	if err := s.Upsert(ctx, 1, []float32{1, 0}); !errors.Is(err, ErrDimMismatch) {
		t.Fatalf("expected ErrDimMismatch, got %v", err)
	}
}

func TestStoreDelete(t *testing.T) {
	db := openTestDB(t)
	createMessagesTable(t, db)
	ctx := context.Background()

	s, _ := NewStore(ctx, db, "m", 3)
	if _, err := db.Exec(`INSERT INTO messages (content) VALUES ('a'), ('b')`); err != nil {
		t.Fatal(err)
	}
	_ = s.Upsert(ctx, 1, []float32{1, 0, 0})
	_ = s.Upsert(ctx, 2, []float32{0, 1, 0})

	if err := s.Delete(ctx, 1); err != nil {
		t.Fatal(err)
	}
	n, _ := s.Count(ctx)
	if n != 1 {
		t.Fatalf("count = %d, want 1", n)
	}
}

func TestStoreModelSeparation(t *testing.T) {
	db := openTestDB(t)
	createMessagesTable(t, db)
	ctx := context.Background()

	sA, _ := NewStore(ctx, db, "model-a", 3)
	sB, _ := NewStore(ctx, db, "model-b", 4)
	if _, err := db.Exec(`INSERT INTO messages (content) VALUES ('a'), ('b'), ('c')`); err != nil {
		t.Fatal(err)
	}
	_ = sA.Upsert(ctx, 1, []float32{1, 0, 0})
	_ = sB.Upsert(ctx, 2, []float32{0, 1, 0, 0})

	nA, _ := sA.CountForModel(ctx)
	nB, _ := sB.CountForModel(ctx)
	if nA != 1 || nB != 1 {
		t.Fatalf("model counts = %d/%d, want 1/1", nA, nB)
	}
	total, _ := sA.Count(ctx)
	if total != 2 {
		t.Fatalf("total count = %d, want 2", total)
	}

	// LoadIndex for A must only include A's row (and match A's dim).
	ixA, err := sA.LoadIndex(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if ixA.Len() != 1 || ixA.Dim() != 3 {
		t.Fatalf("index A len/dim = %d/%d, want 1/3", ixA.Len(), ixA.Dim())
	}
}

func TestStorePendingIDs(t *testing.T) {
	db := openTestDB(t)
	createMessagesTable(t, db)
	ctx := context.Background()

	s, _ := NewStore(ctx, db, "m", 3)
	if _, err := db.Exec(`INSERT INTO messages (content) VALUES ('a'), ('b'), ('c')`); err != nil {
		t.Fatal(err)
	}
	_ = s.Upsert(ctx, 2, []float32{0, 1, 0})

	pending, err := s.PendingIDs(ctx, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(pending) != 2 || pending[0] != 1 || pending[1] != 3 {
		t.Fatalf("pending = %#v, want [1 3]", pending)
	}

	limited, _ := s.PendingIDs(ctx, 0, 1)
	if len(limited) != 1 || limited[0] != 1 {
		t.Fatalf("limited pending = %#v, want [1]", limited)
	}
}

func TestStoreGetContent(t *testing.T) {
	db := openTestDB(t)
	createMessagesTable(t, db)
	ctx := context.Background()

	s, _ := NewStore(ctx, db, "m", 3)
	if _, err := db.Exec(`INSERT INTO messages (content) VALUES ('remember the a770 saga')`); err != nil {
		t.Fatal(err)
	}
	content, err := s.GetContent(ctx, 1)
	if err != nil {
		t.Fatal(err)
	}
	if content != "remember the a770 saga" {
		t.Fatalf("content = %q", content)
	}
}

func TestStoreCascadeDelete(t *testing.T) {
	db := openTestDB(t)
	createMessagesTable(t, db)
	ctx := context.Background()

	s, _ := NewStore(ctx, db, "m", 3)
	if _, err := db.Exec(`INSERT INTO messages (content) VALUES ('x')`); err != nil {
		t.Fatal(err)
	}
	_ = s.Upsert(ctx, 1, []float32{1, 0, 0})
	if _, err := db.Exec(`DELETE FROM messages WHERE message_id = 1`); err != nil {
		t.Fatal(err)
	}
	n, _ := s.Count(ctx)
	if n != 0 {
		t.Fatalf("count after cascade = %d, want 0", n)
	}
}

// TestStoreBackfillSmoke is a tiny end-to-end: write rows, load index,
// search. It mirrors the M3 backfill loop shape.
func TestStoreBackfillSmoke(t *testing.T) {
	db := openTestDB(t)
	createMessagesTable(t, db)
	ctx := context.Background()

	s, _ := NewStore(ctx, db, "m", 16)
	if _, err := db.Exec(`INSERT INTO messages (content) VALUES
		('the arc a770 wedges under compute load'),
		('pi-hole dns runs on swarm1 and swarm2'),
		('the p70 uses nvidia driver 580')`); err != nil {
		t.Fatal(err)
	}

	// Simulate embedding each pending message with deterministic pseudo-vectors.
	for {
		pending, err := s.PendingIDs(ctx, 0, 10)
		if err != nil {
			t.Fatal(err)
		}
		if len(pending) == 0 {
			break
		}
		for _, id := range pending {
			content, err := s.GetContent(ctx, id)
			if err != nil {
				t.Fatal(err)
			}
			vec := pseudoVector(content)
			if err := s.Upsert(ctx, id, vec); err != nil {
				t.Fatal(err)
			}
		}
	}

	n, _ := s.Count(ctx)
	if n != 3 {
		t.Fatalf("backfilled count = %d, want 3", n)
	}

	ix, err := s.LoadIndex(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if ix.Len() != 3 {
		t.Fatalf("index len = %d, want 3", ix.Len())
	}
	hits := ix.Search(pseudoVector("arc gpu resets and wedges"), 1, 0)
	if len(hits) != 1 || hits[0].MessageID != 1 {
		t.Fatalf("expected arc message hit, got %#v", hits)
	}
}

// pseudoVector builds a small bag-of-words vector: each word hashes to a
// dimension and increments it. Words shared between texts produce cosine
// similarity, which is exactly what the smoke test needs without a real
// embedding server.
func pseudoVector(s string) []float32 {
	const dim = 16
	vec := make([]float32, dim)
	for _, word := range strings.Fields(strings.ToLower(s)) {
		h := fnv.New32a()
		h.Write([]byte(word))
		vec[h.Sum32()%dim] += 1
	}
	return vec
}
