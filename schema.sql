CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS naval_doc_chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ DEFAULT NOW(),

    -- Source info
    source_file     TEXT NOT NULL,
    page_start      INT,
    page_end        INT,
    chunk_index     INT,          -- position of chunk within the doc
    line_start      INT,          -- line number where chunk starts in page

    -- Content
    chunk_text      TEXT NOT NULL,
    question        TEXT,
    answer          TEXT,

    -- Embedding (2000 dimensions)
    embedding       VECTOR(2000)
);

-- Index for fast ANN search
CREATE INDEX IF NOT EXISTS naval_chunks_embedding_idx
    ON naval_doc_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Optional: index on source_file for filtering
CREATE INDEX IF NOT EXISTS naval_chunks_source_idx
    ON naval_doc_chunks (source_file);



CREATE OR REPLACE FUNCTION match_naval_chunks(
        query_embedding VECTOR(2000),
        match_count INT DEFAULT 5,
        filter_source TEXT DEFAULT NULL
    )
    RETURNS TABLE (
        id UUID, source_file TEXT, page_start INT, line_start INT,
        chunk_text TEXT, question TEXT, answer TEXT, similarity FLOAT
    )
    LANGUAGE sql STABLE AS $$
        SELECT id, source_file, page_start, line_start,
               chunk_text, question, answer,
               1 - (embedding <=> query_embedding) AS similarity
        FROM naval_doc_chunks
        WHERE (filter_source IS NULL OR source_file = filter_source)
        ORDER BY embedding <=> query_embedding
        LIMIT match_count;
    $$;