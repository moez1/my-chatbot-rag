"""Search service: find relevant chunks using vector similarity.

This is the core of RAG retrieval:
    question → embedding → vector search → top-k most similar chunks

How vector search works:
1. Your question is transformed into a vector (same model as stored embeddings)
2. pgvector computes the cosine distance between your question vector
   and ALL stored embedding vectors
3. It returns the K closest ones (most similar in meaning)

Cosine distance:
- 0.0 = identical meaning
- 1.0 = completely unrelated
- We sort by distance ascending (smallest = most relevant)
"""

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document, Embedding

logger = logging.getLogger(__name__)


async def search_chunks(
    db: AsyncSession,
    query_vector: list[float],
    top_k: int = 5,
    domain: str | None = None,
) -> list[dict]:
    """Search for the most relevant chunks using vector similarity.

    Args:
        db: Async database session.
        query_vector: The embedding of the user's question.
        top_k: Number of results to return.
        domain: Optional filter to search only in a specific domain.

    Returns:
        List of dicts with chunk info and similarity score.
    """
    # Build the query: join Embedding → Chunk → Document
    # Order by cosine distance (closest = most relevant)
    stmt = (
        select(
            Chunk.id.label("chunk_id"),
            Chunk.text,
            Document.id.label("document_id"),
            Document.title.label("document_title"),
            Document.source_type,
            Embedding.vector.cosine_distance(query_vector).label("distance"),
        )
        .join(Embedding, Embedding.chunk_id == Chunk.id)
        .join(Document, Document.id == Chunk.document_id)
        .order_by("distance")
        .limit(top_k)
    )

    # Optional domain filter
    if domain:
        stmt = stmt.where(Document.domain == domain)

    result = await db.execute(stmt)
    rows = result.all()

    return [
        {
            "chunk_id": str(row.chunk_id),
            "text": row.text,
            "document_id": str(row.document_id),
            "document_title": row.document_title,
            "source_type": row.source_type,
            # Convert distance to similarity score (1 - distance)
            "score": round(1 - row.distance, 4),
        }
        for row in rows
    ]
