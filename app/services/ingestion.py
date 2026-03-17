"""Ingestion service: the full pipeline from file to searchable vectors.

This orchestrates the three steps:
    1. PARSE: file → raw text
    2. CHUNK: raw text → list of pieces
    3. EMBED: each piece → vector stored in DB

After this, the document is searchable via vector similarity.
"""

import logging
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document, Embedding
from app.services.chunker import chunk_text
from app.services.parser import detect_source_type, parse_file

logger = logging.getLogger(__name__)


async def ingest_file(
    db: AsyncSession,
    file_path: Path,
    title: str,
    domain: str = "general",
    language: str = "fr",
    metadata: dict | None = None,
    embedding_provider=None,
) -> tuple[Document, int]:
    """Ingest a file into the RAG system.

    Full pipeline: parse → chunk → embed → store.

    Args:
        db: Async database session.
        file_path: Path to the file to ingest.
        title: Human-readable document title.
        domain: Category (professional, personal, cloud, etc.).
        language: Content language.
        metadata: Optional extra info (tags, author, etc.).
        embedding_provider: Provider to generate embeddings.

    Returns:
        Tuple of (Document, number_of_chunks_created).

    Raises:
        ValueError: If file format is not supported or file already exists.
    """
    # Step 1: Parse the file
    text, content_hash = parse_file(file_path)
    logger.info("Parsed %s (%d characters)", file_path.name, len(text))

    # Check for duplicates via content hash
    existing = await db.execute(
        select(Document).where(Document.content_hash == content_hash)
    )
    if existing.scalar_one_or_none():
        msg = f"Document already ingested (same content hash): {file_path.name}"
        raise ValueError(msg)

    # Step 2: Create the document record
    source_type = detect_source_type(file_path)
    doc = Document(
        title=title,
        source_type=source_type,
        source_uri=str(file_path),
        domain=domain,
        language=language,
        content_hash=content_hash,
        metadata_=metadata or {},
    )
    db.add(doc)
    await db.flush()  # Get the doc.id without committing yet

    # Step 3: Chunk the text
    chunks_data = chunk_text(text)
    logger.info("Created %d chunks from %s", len(chunks_data), file_path.name)

    # Step 4: Create chunk records
    chunk_models = []
    for cd in chunks_data:
        chunk = Chunk(
            document_id=doc.id,
            chunk_index=cd.index,
            text=cd.text,
            start_char=cd.start_char,
            end_char=cd.end_char,
        )
        db.add(chunk)
        chunk_models.append(chunk)

    await db.flush()  # Get chunk IDs

    # Step 5: Generate and store embeddings
    if embedding_provider:
        texts = [c.text for c in chunk_models]
        vectors = await embedding_provider.embed_batch(texts)

        for chunk_model, vector in zip(chunk_models, vectors):
            emb = Embedding(
                chunk_id=chunk_model.id,
                provider=embedding_provider.provider_name,
                model=embedding_provider.model_name,
                vector=vector,
            )
            db.add(emb)

    await db.commit()
    logger.info("Ingested %s: %d chunks stored", title, len(chunks_data))

    return doc, len(chunks_data)
