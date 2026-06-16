from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import AppConfig
from embeddings import EmbeddingClient


class VectorStore:
    """
    Wraps ChromaDB, a persistent vector database.

    ChromaDB stores text chunks as embedding vectors + metadata.
    At query time, it finds the *k* nearest neighbors by cosine similarity
    between the query embedding and all stored document embeddings.

    What is a vector database?
      - Traditional DB: "find me documents WHERE name = 'W-2.pdf'" (exact match)
      - Vector DB:      "find me documents SIMILAR TO 'federal tax withholding'"
                            ↳ embedding('federal tax withholding') → compare → top-k

    Persistence: ChromaDB data lives in data/chroma_db/ on disk. It survives
    app restarts — we don't need to re-embed on every launch.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        # Share the same LangChain OllamaEmbeddings instance that EmbeddingClient wraps
        self.embeddings = EmbeddingClient(config).client
        self.store = Chroma(
            collection_name="tax_documents",
            embedding_function=self.embeddings,
            persist_directory=str(config.chroma_dir),
        )

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """
        Convert chunk dicts to LangChain Document objects and add them to ChromaDB.

        ChromaDB calls the embedding function internally for each document's
        page_content. The metadata dict is stored alongside for retrieval.
        """
        docs = [Document(page_content=c["text"], metadata=c["metadata"]) for c in chunks]
        if docs:
            self.store.add_documents(docs)

    def similarity_search(self, query: str, k: int) -> list[Document]:
        """
        Embed the query, find the k most similar chunks, return Documents.
        Does NOT include similarity scores.
        """
        return self.store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int) -> list[tuple[Document, float]]:
        """
        Like similarity_search but each result is paired with its
        distance score (lower = more similar for ChromaDB's default L2).

        We use this in the query pipeline so we can pass scores to RRF.
        """
        return self.store.similarity_search_with_score(query, k=k)

    def clear(self) -> None:
        """
        Delete the existing ChromaDB collection and create a fresh one.

        Called during re-ingestion to ensure stale data is removed.
        ChromaDB collections don't support incremental deletion easily,
        so we just delete and recreate.
        """
        self.store.delete_collection()
        self.store = Chroma(
            collection_name="tax_documents",
            embedding_function=self.embeddings,
            persist_directory=str(self.config.chroma_dir),
        )

    def count(self) -> int:
        """Return the number of documents in the collection (0 if not accessible)."""
        try:
            collection = self.store._collection
            return collection.count()
        except Exception:
            return 0
