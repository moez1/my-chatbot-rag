"""Chat endpoint: ask a question and get an answer based on your documents.

This is the full RAG pipeline:
    POST /chat { question, top_k, domain, provider }
        → embed the question
        → vector search → find relevant chunks
        → build context from chunks
        → send question + context to LLM
        → return answer + sources
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.providers.router import generate_with_fallback, get_embedding_provider
from app.schemas import ChatRequest, ChatResponse, ChunkResult
from app.services.search import search_chunks

router = APIRouter(tags=["Chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with your knowledge base",
    description="Ask a question and get an answer based on your ingested documents.",
)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_session),
):
    """Full RAG: search + generate answer."""
    # Step 1: Embed the question
    embedding_provider = get_embedding_provider()
    query_vector = await embedding_provider.embed(request.question)

    # Step 2: Find relevant chunks
    results = await search_chunks(
        db=db,
        query_vector=query_vector,
        top_k=request.top_k,
        domain=request.domain,
    )

    if not results:
        return ChatResponse(
            answer="Aucun document pertinent trouve dans la base de connaissances.",
            provider="none",
            sources=[],
        )

    # Step 3: Build context from chunks
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(f"[Source {i}: {r['document_title']}]\n{r['text']}")
    context = "\n\n---\n\n".join(context_parts)

    # Step 4: Generate answer with LLM (with fallback)
    try:
        answer, provider_used = await generate_with_fallback(
            question=request.question,
            context=context,
            preferred_provider=request.provider,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return ChatResponse(
        answer=answer,
        provider=provider_used,
        sources=[ChunkResult(**r) for r in results],
    )
