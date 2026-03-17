"""Abstract base classes for LLM providers.

Every provider (OpenAI, DeepSeek, Claude) must implement these interfaces.
This ensures all providers are interchangeable — the rest of the code
doesn't care which provider is used.

This is the Strategy pattern: define an interface, swap implementations.
"""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Base class for embedding providers.

    An embedding provider transforms text into a vector (list of floats).
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier (openai, deepseek, etc.)."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier (text-embedding-3-small, etc.)."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Transform a single text into a vector."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Transform multiple texts into vectors in one API call.

        More efficient than calling embed() in a loop:
        1 HTTP request instead of N.
        """


class ChatProvider(ABC):
    """Base class for chat/generation providers.

    Takes a question + context and generates an answer.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (openai, deepseek, claude)."""

    @abstractmethod
    async def generate(self, question: str, context: str) -> str:
        """Generate an answer based on context from the knowledge base."""
