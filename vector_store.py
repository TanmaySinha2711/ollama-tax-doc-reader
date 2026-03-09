from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import AppConfig
from embeddings import EmbeddingClient


class VectorStore:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.embeddings = EmbeddingClient(config).client
        self.store = Chroma(
            collection_name="tax_documents",
            embedding_function=self.embeddings,
            persist_directory=str(config.chroma_dir),
        )

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        docs = [Document(page_content=c["text"], metadata=c["metadata"]) for c in chunks]
        if docs:
            self.store.add_documents(docs)

    def similarity_search(self, query: str, k: int) -> list[Document]:
        return self.store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int) -> list[tuple[Document, float]]:
        return self.store.similarity_search_with_score(query, k=k)

    def clear(self) -> None:
        self.store.delete_collection()
        self.store = Chroma(
            collection_name="tax_documents",
            embedding_function=self.embeddings,
            persist_directory=str(self.config.chroma_dir),
        )

    def count(self) -> int:
        try:
            collection = self.store._collection
            return collection.count()
        except Exception:
            return 0
