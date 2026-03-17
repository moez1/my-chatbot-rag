"""Search endpoint: find relevant chunks in the knowledge base.

Flow:
    POST /search { query, top_k, domain }
        → embed the query
        → vector similarity search in pgvector
        → return top-k most relevant chunks
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.providers.router import get_embedding_provider
from app.schemas import ChunkResult, SearchRequest, SearchResponse
from app.services.search import search_chunks

router = APIRouter(tags=["Search"])


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search the knowledge base",
    description="Find the most relevant chunks for a given question.",
)
async def search(
    request: SearchRequest,
    db: AsyncSession = Depends(get_session),
):
    """Search for relevant chunks."""
    # Step 1: Embed the query (same model as stored embeddings)
    embedding_provider = get_embedding_provider()
    query_vector = await embedding_provider.embed(request.query)

    # Step 2: Vector search
    results = await search_chunks(
        db=db,
        query_vector=query_vector,
        top_k=request.top_k,
        domain=request.domain,
    )

    return SearchResponse(
        query=request.query,
        results=[ChunkResult(**r) for r in results],
    )
