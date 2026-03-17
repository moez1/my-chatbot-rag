"""Chat provider router with fallback logic.

The router:
1. If user specified a provider → use that one
2. Otherwise → try the default (DeepSeek = cheapest)
3. If default fails → try the next provider in the fallback chain

This saves money while keeping reliability.

The fallback chain is configured in settings.yaml:
    routing.fallback_chain: [openai, anthropic, deepseek]
"""

import logging

from app.providers.base import ChatProvider, EmbeddingProvider
from app.providers.claude_provider import ClaudeChatProvider
from app.providers.deepseek_provider import DeepSeekChatProvider
from app.providers.openai_provider import OpenAIChatProvider, OpenAIEmbeddingProvider
from app.settings import get_settings

logger = logging.getLogger(__name__)

# Registry: maps provider name → class
CHAT_PROVIDERS: dict[str, type[ChatProvider]] = {
    "deepseek": DeepSeekChatProvider,
    "openai": OpenAIChatProvider,
    "claude": ClaudeChatProvider,
}


def get_embedding_provider() -> EmbeddingProvider:
    """Get the configured embedding provider.

    Currently only OpenAI — because you must NEVER mix
    embeddings from different providers in the same DB.
    """
    return OpenAIEmbeddingProvider()


def get_chat_provider(provider_name: str | None = None) -> ChatProvider:
    """Get a chat provider by name, or the default."""
    settings = get_settings()
    name = provider_name or settings.providers.default_chat
    provider_class = CHAT_PROVIDERS.get(name)
    if not provider_class:
        msg = f"Unknown chat provider: {name}. Available: {list(CHAT_PROVIDERS.keys())}"
        raise ValueError(msg)
    return provider_class()


async def generate_with_fallback(
    question: str,
    context: str,
    preferred_provider: str | None = None,
) -> tuple[str, str]:
    """Generate an answer, with automatic fallback on failure.

    Returns:
        Tuple of (answer_text, provider_name_that_succeeded).

    Raises:
        RuntimeError: If ALL providers fail.
    """
    settings = get_settings()
    preferred = preferred_provider or settings.providers.default_chat

    # Build order: preferred first, then fallback chain
    fallback = settings.routing.fallback_chain
    order = [preferred] + [p for p in fallback if p != preferred]

    errors = []
    for name in order:
        try:
            provider = get_chat_provider(name)
            answer = await provider.generate(question, context)
            if name != preferred:
                logger.info("Fallback: %s succeeded (preferred %s failed)", name, preferred)
            return answer, name
        except Exception as e:
            logger.warning("Provider %s failed: %s", name, e)
            errors.append(f"{name}: {e}")

    msg = "All chat providers failed:\n" + "\n".join(errors)
    raise RuntimeError(msg)
