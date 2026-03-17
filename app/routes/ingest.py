"""Ingestion endpoint: upload a file to add it to the knowledge base.

Flow:
    POST /ingest (file upload)
        → parse the file
        → chunk the text
        → generate embeddings
        → store everything in DB
        → return document ID + chunks count
"""

import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.providers.router import get_embedding_provider
from app.schemas import IngestResponse
from app.services.ingestion import ingest_file
from app.services.parser import detect_source_type

router = APIRouter(tags=["Ingestion"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document",
    description="Upload a file (PDF, code, markdown, etc.) to add it to the knowledge base.",
)
async def ingest(
    file: UploadFile,
    title: str | None = None,
    domain: str = "general",
    language: str = "fr",
    db: AsyncSession = Depends(get_session),
):
    """Ingest a document into the RAG system."""
    # Save uploaded file to a temp location
    suffix = Path(file.filename or "upload.txt").suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        doc_title = title or Path(file.filename or "untitled").stem
        embedding_provider = get_embedding_provider()

        doc, chunks_count = await ingest_file(
            db=db,
            file_path=tmp_path,
            title=doc_title,
            domain=domain,
            language=language,
            embedding_provider=embedding_provider,
        )

        return IngestResponse(
            document_id=str(doc.id),
            title=doc.title,
            chunks_count=chunks_count,
            message=f"Document ingested: {chunks_count} chunks created.",
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)
