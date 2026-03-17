"""Claude (Anthropic) provider for chat.

Claude uses a DIFFERENT API format than OpenAI:
- system prompt is a separate parameter, not a message
- response is in response.content[0].text, not response.choices[0].message.content
- uses its own anthropic SDK, not the openai SDK

This is why we have the abstract ChatProvider base class:
each provider has different API details, but the rest of our
code just calls provider.generate(question, context).
"""

import os

import anthropic

from app.providers.base import ChatProvider

RAG_SYSTEM_PROMPT = (
    "Tu es un assistant qui repond aux questions en te basant "
    "UNIQUEMENT sur le contexte fourni. "
    "Si le contexte ne contient pas la reponse, dis-le clairement. "
    "Cite les sources quand c'est possible. "
    "Reponds en francais."
)


class ClaudeChatProvider(ChatProvider):
    """Generate answers using Claude's API."""

    def __init__(self) -> None:
        self.client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    @property
    def name(self) -> str:
        return "claude"

    async def generate(self, question: str, context: str) -> str:
        """Generate an answer using Claude."""
        response = await self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=RAG_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Contexte:\n{context}\n\nQuestion: {question}"},
            ],
        )
        return response.content[0].text
