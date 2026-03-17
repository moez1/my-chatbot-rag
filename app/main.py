"""FastAPI application entry point.

Endpoints:
    GET  /health     → health check
    POST /ingest     → ingest a document (file upload)
    POST /search     → search the knowledge base
    POST /chat       → ask a question (full RAG pipeline)
    GET  /documents  → list ingested documents
    DEL  /documents/{id} → delete a document
"""

from fastapi import FastAPI

from app.routes.chat import router as chat_router
from app.routes.documents import router as documents_router
from app.routes.ingest import router as ingest_router
from app.routes.search import router as search_router
from app.settings import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app.name,
    description="Personal RAG system — ask questions about your own documents.",
    version="0.1.0",
)

# Register all route modules
app.include_router(ingest_router)
app.include_router(search_router)
app.include_router(chat_router)
app.include_router(documents_router)


@app.get("/health", tags=["Health"])
async def healthcheck():
    """Health check endpoint."""
    return {"status": "ok", "env": settings.app.environment}


