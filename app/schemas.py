"""Pydantic schemas for request validation and response serialization.

These schemas define the shape of data going in and out of the API.
FastAPI uses them for:
- Automatic request validation (rejects bad input with 422)
- Automatic response serialization (converts to JSON)
- Automatic OpenAPI documentation (Swagger UI at /docs)
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# ── Ingestion ──


class IngestRequest(BaseModel):
    """Request to ingest a document."""

    title: str = Field(..., min_length=1, max_length=500)
    source_type: str = Field(..., description="pdf, markdown, code, text")
    domain: str = Field(default="general")
    language: str = Field(default="fr")
    metadata: dict | None = Field(default=None)


class IngestResponse(BaseModel):
    """Response after ingestion."""

    document_id: str
    title: str
    chunks_count: int
    message: str


# ── Search ──


class SearchRequest(BaseModel):
    """Request to search the knowledge base."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    domain: str | None = Field(default=None)


class ChunkResult(BaseModel):
    """A single search result (one chunk)."""

    chunk_id: str
    text: str
    score: float
    document_title: str
    document_id: str
    source_type: str


class SearchResponse(BaseModel):
    """Response from search."""

    query: str
    results: list[ChunkResult]


# ── Chat (RAG) ──


class ChatRequest(BaseModel):
    """Request to chat with the RAG system."""

    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    domain: str | None = Field(default=None)
    provider: str | None = Field(default=None, description="Force a specific chat provider")


class ChatResponse(BaseModel):
    """Response from RAG chat."""

    answer: str
    provider: str
    sources: list[ChunkResult]


# ── Documents ──


class DocumentResponse(BaseModel):
    """Response for a document listing."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    source_type: str
    domain: str
    language: str
    created_at: datetime
    chunks_count: int = 0
