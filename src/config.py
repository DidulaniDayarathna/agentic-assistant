from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider ---
    llm_provider: str = Field(default="openai", description="'openai' or 'huggingface'")
    openai_api_key: str | None = None         
    openai_model: str = "gpt-4o-mini"

    hf_token: str | None = None
    hf_model: str = "deepseek-ai/DeepSeek-R1:fastest"
    hf_base_url: str = "https://router.huggingface.co/v1"

    llm_temperature: float = 0.3
    llm_max_tokens: int = 800

    # --- Tools ---
    openweathermap_api_key: str | None = None
    tavily_api_key: str | None = None  # preferred search backend if present

    # --- Agent behavior ---
    max_agent_iterations: int = 6
    request_timeout_seconds: int = 20

    # --- RAG tool ---
    rag_data_dir: str = "data/sample_docs"
    rag_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Observability ---
    langsmith_api_key: str | None = None
    langsmith_project: str = "agentic-assistant"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Settings are cached so we parse the environment only once."""
    return Settings()


settings = get_settings()
