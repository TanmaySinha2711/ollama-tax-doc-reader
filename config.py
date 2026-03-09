from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TAX_AI_", env_file=".env", extra="ignore")

    ollama_base_url: str = Field(default="http://localhost:11434")
    llm_model: str = Field(default="qwen3.5:9b")
    embedding_model: str = Field(default="nomic-embed-text:latest")

    data_dir: Path = Field(default=Path("data"))
    chroma_dir: Path = Field(default=Path("data/chroma_db"))
    structured_dir: Path = Field(default=Path("data/structured"))
    bm25_path: Path = Field(default=Path("data/bm25_index.pkl"))
    ingestion_state_path: Path = Field(default=Path("data/ingestion_state.json"))

    knowledge_dir: Path = Field(default=Path("knowledge"))

    chunk_size_tokens: int = Field(default=800)
    chunk_overlap_tokens: int = Field(default=200)

    vector_top_k: int = Field(default=8)
    keyword_top_k: int = Field(default=8)
    final_top_k: int = Field(default=10)

    llm_temperature: float = Field(default=0.1)
    memory_turns: int = Field(default=5)


def get_config() -> AppConfig:
    cfg = AppConfig()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.chroma_dir.mkdir(parents=True, exist_ok=True)
    cfg.structured_dir.mkdir(parents=True, exist_ok=True)
    cfg.knowledge_dir.mkdir(parents=True, exist_ok=True)
    return cfg
