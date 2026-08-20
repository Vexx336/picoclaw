package embeddings

import (
	"context"
	"database/sql"
	"encoding/binary"
	"fmt"
	"math"
)

// Store persists message embeddings inside the same SQLite database as the
// seahorse engine (typically sessions/seahorse.db).
//
// The schema is model-agnostic: each row records the embedding model name and
// dimension, so swapping models invalidates old rows by model and triggers a
// clean re-embed without any schema change.
type Store struct {
	db    *sql.DB
	model string
	dim   int
}

// NewStore opens (or reuses) a *sql.DB for the message_embeddings table and
// ensures the schema exists.
//
// db may be shared with the seahorse engine (it just runs CREATE TABLE IF NOT
// EXISTS statements); pass a dedicated connection if you prefer isolation.
func NewStore(ctx context.Context, db *sql.DB, model string, dim int) (*Store, error) {
	s := &Store{db: db, model: model, dim: dim}
	if err := s.Migrate(ctx); err != nil {
		return nil, err
	}
	return s, nil
}

// Migrate creates the message_embeddings table and supporting index.
// Idempotent: safe to call on every startup.
//
// Note: SQLite enforces foreign keys only when `PRAGMA foreign_keys = ON`
// is set on the connection. This runs the pragma for the connection the
// statement executes on; when sharing a pooled *sql.DB with the seahorse
// engine, enable it at the DSN level (e.g. `_pragma=foreign_keys(1)` with
// modernc.org/sqlite) or call Store.Delete explicitly on message removal.
func (s *Store) Migrate(ctx context.Context) error {
	stmts := []string{
		`PRAGMA foreign_keys = ON`,
		`CREATE TABLE IF NOT EXISTS message_embeddings (
			message_id  INTEGER PRIMARY KEY REFERENCES messages(message_id) ON DELETE CASCADE,
			model       TEXT NOT NULL,
			dim         INTEGER NOT NULL,
			vector      BLOB NOT NULL,
			created_at  TEXT NOT NULL DEFAULT (datetime('now'))
		)`,
		`CREATE INDEX IF NOT EXISTS idx_embeddings_model ON message_embeddings(model)`,
	}
	for _, stmt := range stmts {
		if _, err := s.db.ExecContext(ctx, stmt); err != nil {
			return fmt.Errorf("embeddings: migrate: %w", err)
		}
	}
	return nil
}

// Model returns the model name this store is configured for.
func (s *Store) Model() string { return s.model }

// Dim returns the expected vector dimension.
func (s *Store) Dim() int { return s.dim }

// Upsert inserts or replaces the embedding for one message.
func (s *Store) Upsert(ctx context.Context, messageID int64, vec []float32) error {
	if len(vec) != s.dim {
		return fmt.Errorf("%w: got %d, want %d", ErrDimMismatch, len(vec), s.dim)
	}
	blob := encodeVector(vec)
	_, err := s.db.ExecContext(ctx,
		`INSERT INTO message_embeddings (message_id, model, dim, vector)
		 VALUES (?, ?, ?, ?)
		 ON CONFLICT(message_id) DO UPDATE SET
		   model = excluded.model,
		   dim = excluded.dim,
		   vector = excluded.vector,
		   created_at = datetime('now')`,
		messageID, s.model, s.dim, blob)
	if err != nil {
		return fmt.Errorf("embeddings: upsert: %w", err)
	}
	return nil
}

// Delete removes the embedding for one message.
func (s *Store) Delete(ctx context.Context, messageID int64) error {
	_, err := s.db.ExecContext(ctx,
		`DELETE FROM message_embeddings WHERE message_id = ?`, messageID)
	if err != nil {
		return fmt.Errorf("embeddings: delete: %w", err)
	}
	return nil
}

// Count returns the number of stored vectors (all models).
func (s *Store) Count(ctx context.Context) (int, error) {
	var n int
	if err := s.db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM message_embeddings`).Scan(&n); err != nil {
		return 0, fmt.Errorf("embeddings: count: %w", err)
	}
	return n, nil
}

// CountForModel returns the number of vectors for the configured model.
func (s *Store) CountForModel(ctx context.Context) (int, error) {
	var n int
	if err := s.db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM message_embeddings WHERE model = ?`, s.model).Scan(&n); err != nil {
		return 0, fmt.Errorf("embeddings: count for model: %w", err)
	}
	return n, nil
}

// LoadIndex streams all vectors for the configured model into a flat Index.
// Called once at engine start; ~27k rows × 1KB ≈ 27MB of BLOBs, well under
// the 1MB exec output cap when run outside this tool.
func (s *Store) LoadIndex(ctx context.Context) (*Index, error) {
	ix := NewIndex(s.dim)
	rows, err := s.db.QueryContext(ctx,
		`SELECT message_id, vector FROM message_embeddings WHERE model = ?`, s.model)
	if err != nil {
		return nil, fmt.Errorf("embeddings: load index: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var (
			id   int64
			blob []byte
		)
		if err := rows.Scan(&id, &blob); err != nil {
			return nil, fmt.Errorf("embeddings: scan row: %w", err)
		}
		vec, err := decodeVectorBlob(blob, s.dim)
		if err != nil {
			// Skip malformed rows; they can be re-embedded in a backfill.
			continue
		}
		ix.Add(id, vec)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("embeddings: iterate rows: %w", err)
	}
	return ix, nil
}

// PendingIDs returns message IDs that have no embedding for the configured
// model, optionally limited to a conversation. Used by the backfill helper.
func (s *Store) PendingIDs(ctx context.Context, conversationID int64, limit int) ([]int64, error) {
	query := `SELECT m.message_id
	          FROM messages m
	          LEFT JOIN message_embeddings e
	            ON e.message_id = m.message_id AND e.model = ?
	          WHERE e.message_id IS NULL`
	args := []any{s.model}
	if conversationID > 0 {
		query += ` AND m.conversation_id = ?`
		args = append(args, conversationID)
	}
	query += ` ORDER BY m.message_id`
	if limit > 0 {
		query += ` LIMIT ?`
		args = append(args, limit)
	}

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("embeddings: pending ids: %w", err)
	}
	defer rows.Close()

	var ids []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("embeddings: scan pending id: %w", err)
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}

// GetContent returns the text content for a message, used when backfilling.
// Falls back to parts text when the top-level content column is empty.
func (s *Store) GetContent(ctx context.Context, messageID int64) (string, error) {
	var content string
	err := s.db.QueryRowContext(ctx,
		`SELECT content FROM messages WHERE message_id = ?`, messageID).Scan(&content)
	if err != nil {
		return "", fmt.Errorf("embeddings: get content: %w", err)
	}
	return content, nil
}

// encodeVector serializes []float32 as little-endian float32 bytes.
func encodeVector(vec []float32) []byte {
	blob := make([]byte, len(vec)*4)
	for i, v := range vec {
		binary.LittleEndian.PutUint32(blob[i*4:], math.Float32bits(v))
	}
	return blob
}

// decodeVectorBlob deserializes a vector BLOB, verifying the length.
func decodeVectorBlob(blob []byte, wantDim int) ([]float32, error) {
	if len(blob)%4 != 0 {
		return nil, fmt.Errorf("embeddings: blob length %d not a multiple of 4", len(blob))
	}
	if wantDim > 0 && len(blob)/4 != wantDim {
		return nil, fmt.Errorf("%w: blob has %d floats, want %d", ErrDimMismatch, len(blob)/4, wantDim)
	}
	vec := make([]float32, len(blob)/4)
	for i := range vec {
		bits := binary.LittleEndian.Uint32(blob[i*4 : i*4+4])
		vec[i] = math.Float32frombits(bits)
	}
	return vec, nil
}
