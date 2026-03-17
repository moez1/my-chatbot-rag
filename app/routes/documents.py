"""Documents endpoint: list and manage ingested documents."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.db.models import Chunk, Document
from app.schemas import DocumentResponse

router = APIRouter(tags=["Documents"])


@router.get(
    "/documents",
    response_model=list[DocumentResponse],
    summary="List all documents",
    description="List all ingested documents with their chunk count.",
)
async def list_documents(
    domain: str | None = None,
    db: AsyncSession = Depends(get_session),
):
    """List all documents, optionally filtered by domain."""
    stmt = (
        select(
            Document,
            func.count(Chunk.id).label("chunks_count"),
        )
        .outerjoin(Chunk, Chunk.document_id == Document.id)
        .group_by(Document.id)
        .order_by(Document.created_at.desc())
    )

    if domain:
        stmt = stmt.where(Document.domain == domain)

    result = await db.execute(stmt)
    rows = result.all()

    return [
        DocumentResponse(
            id=str(doc.id),
            title=doc.title,
            source_type=doc.source_type,
            domain=doc.domain,
            language=doc.language,
            created_at=doc.created_at,
            chunks_count=count,
        )
        for doc, count in rows
    ]


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
    description="Delete a document and all its chunks and embeddings.",
)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_session),
):
    """Delete a document (cascades to chunks and embeddings)."""
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )
    await db.delete(doc)
    await db.commit()
