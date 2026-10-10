-- Run once against the target Postgres database. Requires the pgvector extension.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id              BIGSERIAL PRIMARY KEY,
    doc_id          TEXT NOT NULL,
    doc_sha256      TEXT NOT NULL,
    source_path     TEXT NOT NULL,
    page            INTEGER,
    chunk_index     INTEGER NOT NULL,
    text            TEXT NOT NULL,
    classification  TEXT NOT NULL,          -- e.g. UNCLASSIFIED / SECRET / TOP_SECRET
    extra           JSONB DEFAULT '{}',
    embedding       vector(1024) NOT NULL,   -- matches config.embedding_dim
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Every query MUST filter on classification (see security/classification.py) --
-- this index makes that filter cheap instead of optional.
CREATE INDEX IF NOT EXISTS idx_chunks_classification ON chunks (classification);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id);

-- IVFFlat ANN index; run `ANALYZE chunks;` after bulk load, tune `lists` to your row count.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
