"""Qwen (Alibaba) provider for chat.

Qwen uses the OpenAI-compatible API format, just like DeepSeek.
We reuse the openai Python client with Alibaba's DashScope base_url.

Qwen is the model Airbnb uses in production:
- Fast and cheap (~$0.30/1M tokens)
- Open source (can be self-hosted later)
- Good multilingual support (trained on Chinese + English + French)

DashScope is Alibaba's API platform for accessing Qwen models.
API key: https://dashscope.console.aliyun.com/
"""

import os

import openai

from app.providers.base import ChatProvider
from app.settings import get_settings

RAG_SYSTEM_PROMPT = (
    "Tu es un assistant qui repond aux questions en te basant "
    "UNIQUEMENT sur le contexte fourni. "
    "Si le contexte ne contient pas la reponse, dis-le clairement. "
    "Cite les sources quand c'est possible. "
    "Reponds en francais."
)


class QwenChatProvider(ChatProvider):
    """Generate answers using Alibaba's Qwen API via DashScope."""

    def __init__(self) -> None:
        self.client = openai.AsyncOpenAI(
            api_key=os.getenv("QWEN_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self._model = get_settings().providers.models.qwen.chat

    @property
    def name(self) -> str:
        return "qwen"

    async def generate(self, question: str, context: str) -> str:
        """Generate an answer using the configured Qwen chat model."""
        response = await self.client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": f"Contexte:\n{context}\n\nQuestion: {question}"},
            ],
        )
        return response.choices[0].message.content or ""
