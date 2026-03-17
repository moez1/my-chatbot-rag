"""Initial migration: create documents, chunks, embeddings tables.

This sets up the complete RAG data model:
    documents → chunks → embeddings

Also enables the pgvector extension and creates a HNSW index
for fast cosine similarity search.

Why HNSW instead of IVFFlat?
- IVFFlat requires training on existing data (needs rows first)
- HNSW works immediately, even on an empty table
- HNSW is more accurate for small-to-medium datasets
- IVFFlat is only better at very large scale (millions of rows)
"""

from alembic import op


revision = "20260303_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # ── Documents table ──
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            title VARCHAR(500) NOT NULL,
            source_type VARCHAR(50) NOT NULL,
            source_uri VARCHAR(1000) NOT NULL,
            domain VARCHAR(100) NOT NULL DEFAULT 'general',
            language VARCHAR(20) NOT NULL DEFAULT 'fr',
            content_hash VARCHAR(64) NOT NULL UNIQUE,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )

    # ── Chunks table ──
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            start_char INTEGER NOT NULL DEFAULT 0,
            end_char INTEGER NOT NULL DEFAULT 0,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chunks_document_id
            ON chunks (document_id);
        """
    )

    # ── Embeddings table ──
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            chunk_id UUID NOT NULL UNIQUE REFERENCES chunks(id) ON DELETE CASCADE,
            provider VARCHAR(50) NOT NULL,
            model VARCHAR(100) NOT NULL,
            vector vector(1536) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )

    # HNSW index for fast cosine similarity search
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_embeddings_vector_hnsw
            ON embeddings
            USING hnsw (vector vector_cosine_ops);
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_embeddings_vector_hnsw;")
    op.execute("DROP TABLE IF EXISTS embeddings;")
    op.execute("DROP INDEX IF EXISTS idx_chunks_document_id;")
    op.execute("DROP TABLE IF EXISTS chunks;")
    op.execute("DROP TABLE IF EXISTS documents;")

