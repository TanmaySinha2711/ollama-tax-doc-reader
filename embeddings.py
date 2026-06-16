from __future__ import annotations

from langchain_ollama import OllamaEmbeddings

from config import AppConfig


class EmbeddingClient:
    """
    Thin wrapper around LangChain's OllamaEmbeddings.

    An embedding is a dense vector (list of floats) that represents the
    "meaning" of a piece of text. Sentences with similar meaning produce
    similar vectors, which can be compared by cosine similarity.

    We use nomic-embed-text (via Ollama) which produces 768-dimensional
    vectors. The model runs locally — no data is sent to any external API.

    Two use cases:
      - embed_documents(): batch-embeds all chunks during ingestion.
      - embed_query(): embeds the user's question at query time so it
        can be compared against the stored document vectors.
    """

    def __init__(self, config: AppConfig) -> None:
        self.client = OllamaEmbeddings(
            base_url=config.ollama_base_url,
            model=config.embedding_model,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single batch call (faster than one-by-one)."""
        return self.client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.client.embed_query(text)
