import logging

from langchain_openai import ChatOpenAI

from src.config import settings

logger = logging.getLogger(__name__)


def get_llm() -> ChatOpenAI:
    """Return a configured chat model based on settings.llm_provider.

    Both branches currently return a ChatOpenAI instance because HF's
    router exposes an OpenAI-compatible endpoint — but callers only
    depend on this function, not on that implementation detail.
    """
    if settings.llm_provider == "huggingface":
        if not settings.hf_token:
            raise RuntimeError("HF_TOKEN is not set but llm_provider='huggingface'")
        logger.info("Using Hugging Face router model: %s", settings.hf_model)
        return ChatOpenAI(
            model=settings.hf_model,
            api_key=settings.hf_token,
            base_url=settings.hf_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.request_timeout_seconds,
        )

    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set but llm_provider='openai'")
        logger.info("Using OpenAI model: %s", settings.openai_model)
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.request_timeout_seconds,
        )

    raise ValueError(f"Unknown llm_provider: {settings.llm_provider!r}")
