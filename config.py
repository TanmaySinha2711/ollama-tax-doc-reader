from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """
    Central configuration container for the entire application.

    Uses pydantic-settings so every field can be overridden with an environment
    variable prefixed TAX_AI_ (e.g. TAX_AI_LLM_MODEL=llama3). This makes
    deployment-specific tweaks (Docker, CI, different machines) trivial without
    editing any code.
    """

    # Instruct pydantic to look for env vars like TAX_AI_OLLAMA_BASE_URL,
    # optionally read a .env file, and ignore any extra env vars it finds.
    model_config = SettingsConfigDict(env_prefix="TAX_AI_", env_file=".env", extra="ignore")

    # ── Ollama connection ───────────────────────────────────────────────
    # These two models MUST already exist in your local Ollama instance.
    # The embedding model converts text to vectors; the LLM model generates
    # answers. Both run 100% locally — no data ever leaves your machine.

    ollama_base_url: str = Field(default="http://localhost:11434")
    llm_model: str = Field(default="qwen3.5:9b")
    embedding_model: str = Field(default="nomic-embed-text:latest")

    # ── Disk paths ──────────────────────────────────────────────────────
    # Every persistent artifact lives under a single data/ tree for clean
    # cleanup or backup. Knowledge markdown files live separately.

    data_dir: Path = Field(default=Path("data"))
    chroma_dir: Path = Field(default=Path("data/chroma_db"))
    structured_dir: Path = Field(default=Path("data/structured"))
    bm25_path: Path = Field(default=Path("data/bm25_index.pkl"))
    ingestion_state_path: Path = Field(default=Path("data/ingestion_state.json"))

    knowledge_dir: Path = Field(default=Path("knowledge"))

    # ── Chunking ────────────────────────────────────────────────────────
    # Documents are split into overlapping token windows. Overlap prevents
    # cutting a sentence in half — every token boundary appears in at least
    # two consecutive chunks, giving retrieval multiple chances to find it.
    #
    #   chunk 0: tokens   0-800
    #   chunk 1: tokens 600-1400   (200-token overlap)
    #   chunk 2: tokens 1200-2000
    #   ...

    chunk_size_tokens: int = Field(default=800)
    chunk_overlap_tokens: int = Field(default=200)

    # ── Retrieval top-K ─────────────────────────────────────────────────
    # During a query:
    #   1. Vector (semantic) search returns vector_top_K results.
    #   2. Keyword (BM25) search returns keyword_top_k results.
    #   3. RRF fuses both lists together and keeps the final_top_k best.
    #
    # The fused top 10 chunks + structured data fit comfortably inside
    # the LLM's context window (~8K tokens for Qwen 3.5 9B).

    vector_top_k: int = Field(default=8)
    keyword_top_k: int = Field(default=8)
    final_top_k: int = Field(default=10)

    # ── LLM behavior ────────────────────────────────────────────────────
    # temperature=0.1: very low — tax answers should be deterministic and
    # factual, not creative. memory_turns controls how many conversation
    # exchanges are included in the prompt for follow-up questions.

    llm_temperature: float = Field(default=0.1)
    memory_turns: int = Field(default=5)


def get_config() -> AppConfig:
    """Factory: creates AppConfig and ensures all required directories exist."""
    cfg = AppConfig()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.chroma_dir.mkdir(parents=True, exist_ok=True)
    cfg.structured_dir.mkdir(parents=True, exist_ok=True)
    cfg.knowledge_dir.mkdir(parents=True, exist_ok=True)
    return cfg
