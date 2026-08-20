# pkg/embeddings

Pure-Go (no CGO) semantic memory support for the seahorse context manager.

Three small pieces:

| File | Purpose |
|---|---|
| `client.go` | HTTP client for llama.cpp's OpenAI-compatible `/v1/embeddings` (float or base64 responses, batching, retry, dim validation) |
| `index.go` | In-memory flat nearest-neighbor index (cosine similarity; pre-normalized vectors) |
| `store.go` | `message_embeddings` table in the same SQLite DB as seahorse, with backfill helpers (`PendingIDs`, `GetContent`, `LoadIndex`) |

## Why flat index?

At ~27k messages × 1024 dims, a brute-force cosine query is a few million
multiply-adds — single-digit milliseconds on any modern CPU. HNSW adds
complexity and tuning for no measurable win at this scale. If index load or
query time ever becomes a problem, swap the `Index` implementation; the
interface (`Add` / `Remove` / `Search`) is small and stable.

## Why the schema is model-agnostic

Each row records `model` and `dim`, so swapping the embedding model (e.g.
English-only → multilingual) invalidates old rows by `model` and triggers a
clean re-embed with zero schema change. `LoadIndex` only loads vectors for the
configured model.

## Foreign keys

`message_embeddings.message_id` references `messages(message_id) ON DELETE
CASCADE`, but SQLite only enforces FKs when `PRAGMA foreign_keys = ON` per
connection. `Migrate` runs the pragma on its connection, but when sharing a
pooled `*sql.DB` with the seahorse engine you must enable it at the DSN:

```go
sql.Open("sqlite", "file:"+path+"?_pragma=foreign_keys(1)")
```

or call `Store.Delete` explicitly when seahorse deletes messages (M2 wiring).

## Usage sketch (M2 integration)

```go
// Open/store
db := engineDB // shared with seahorse
store, err := embeddings.NewStore(ctx, db, "jina-embeddings-v3", 1024)

// Client
client := embeddings.NewClient("http://100.88.1.92:18084/v1", "jina-embeddings-v3", 1024)

// Write path (async)
vec, err := client.Embed(ctx, messageText)
store.Upsert(ctx, messageID, vec)

// Query path
qvec, _ := client.Embed(ctx, userQuery)
index, _ := store.LoadIndex(ctx)      // at engine start
hits := index.Search(qvec, 8, 0.35)   // top-8, min score 0.35
```

## Tests

```bash
go test ./pkg/embeddings/ -count=1          # unit tests (no network)
go test ./pkg/embeddings/ -count=1 -race    # race detector
EMB_LIVE=1 go test ./pkg/embeddings/ -run TestLiveSkynet -v  # live smoke vs skynet
```
