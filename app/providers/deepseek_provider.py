"""DeepSeek provider for chat.

DeepSeek uses the OpenAI-compatible API format, so we reuse
the openai Python client with a different base_url.
This is common — many LLM providers copy OpenAI's API format.

DeepSeek is ~10x cheaper than OpenAI, so it's the default
for everyday questions. OpenAI/Claude are fallbacks.
"""

import os

import openai

from app.providers.base import ChatProvider

RAG_SYSTEM_PROMPT = (
    "Tu es un assistant qui repond aux questions en te basant "
    "UNIQUEMENT sur le contexte fourni. "
    "Si le contexte ne contient pas la reponse, dis-le clairement. "
    "Cite les sources quand c'est possible. "
    "Reponds en francais."
)


class DeepSeekChatProvider(ChatProvider):
    """Generate answers using DeepSeek's API."""

    def __init__(self) -> None:
        self.client = openai.AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )

    @property
    def name(self) -> str:
        return "deepseek"

    async def generate(self, question: str, context: str) -> str:
        """Generate an answer using DeepSeek."""
        response = await self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": f"Contexte:\n{context}\n\nQuestion: {question}"},
            ],
        )
        return response.choices[0].message.content or ""
