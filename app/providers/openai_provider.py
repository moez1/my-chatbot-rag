"""OpenAI provider for embeddings and chat.

OpenAI is used for:
- Embeddings: text-embedding-3-small (1536 dimensions, cheap, good quality)
- Chat: gpt-4o-mini (fast, cheaper than gpt-4o, good enough for RAG)
"""

import os

import openai

from app.providers.base import ChatProvider, EmbeddingProvider
from app.settings import get_settings

RAG_SYSTEM_PROMPT = (
    "Tu es un assistant qui repond aux questions en te basant "
    "UNIQUEMENT sur le contexte fourni. "
    "Si le contexte ne contient pas la reponse, dis-le clairement. "
    "Cite les sources quand c'est possible. "
    "Reponds en francais."
)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Generate embeddings using OpenAI's API."""

    def __init__(self) -> None:
        self.client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model = get_settings().providers.models.openai.embedding

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    async def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        response = await self.client.embeddings.create(
            input=text,
            model=self._model,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in one API call."""
        response = await self.client.embeddings.create(
            input=texts,
            model=self._model,
        )
        return [item.embedding for item in response.data]


class OpenAIChatProvider(ChatProvider):
    """Generate answers using OpenAI's GPT models."""

    def __init__(self) -> None:
        self.client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model = get_settings().providers.models.openai.chat

    @property
    def name(self) -> str:
        return "openai"

    async def generate(self, question: str, context: str) -> str:
        """Generate an answer using the configured OpenAI chat model."""
        response = await self.client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": f"Contexte:\n{context}\n\nQuestion: {question}"},
            ],
        )
        return response.choices[0].message.content or ""
