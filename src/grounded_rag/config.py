"""Central configuration. Every tunable knob lives here, sourced from env/.env.

Loaded once as a module-level singleton (`settings`). Nothing else in the package
hardcodes paths, model names, or thresholds.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM backend (OpenAI-compatible; base_url makes it provider-agnostic) ---
    openai_api_key: str = Field(default="sk-missing")
    openai_base_url: str | None = Field(default=None)
    llm_model: str = Field(default="gpt-4o-mini")
    llm_temperature: float = Field(default=0.0)
    llm_max_tokens: int = Field(default=700)

    # --- local models ---
    embed_model: str = Field(default="BAAI/bge-small-en-v1.5")
    rerank_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # --- prompts (versioned; changing a prompt is a reviewable diff, not a buried string) ---
    answer_prompt_version: str = Field(default="answer_v3")

    # --- chunking ---
    chunk_max_tokens: int = Field(default=700)  # within the 500-800 band
    chunk_min_tokens: int = Field(default=500)
    chunk_overlap_tokens: int = Field(default=100)

    # --- retrieval ---
    top_k: int = Field(default=20)  # candidates pulled per retriever before fusion/rerank
    top_n: int = Field(default=5)  # contexts handed to the LLM
    refusal_min_contexts: int = Field(default=1)  # below this -> structured refusal

    # --- storage ---
    chroma_dir: Path = Field(default=REPO_ROOT / ".chroma")
    collection: str = Field(default="grounded_rag")
    data_dir: Path = Field(default=REPO_ROOT / "data")

    @property
    def resolved_chroma_dir(self) -> Path:
        p = self.chroma_dir if self.chroma_dir.is_absolute() else REPO_ROOT / self.chroma_dir
        return p

    @property
    def resolved_data_dir(self) -> Path:
        p = self.data_dir if self.data_dir.is_absolute() else REPO_ROOT / self.data_dir
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
