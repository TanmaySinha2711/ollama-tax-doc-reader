from __future__ import annotations

from langchain_ollama import OllamaEmbeddings

from config import AppConfig


class EmbeddingClient:
    def __init__(self, config: AppConfig) -> None:
        self.client = OllamaEmbeddings(
            base_url=config.ollama_base_url,
            model=config.embedding_model,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed_query(text)
