from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


class KeywordIndex:
    def __init__(self) -> None:
        self.bm25: BM25Okapi | None = None
        self.chunks: list[dict[str, Any]] = []
        self.tokenized: list[list[str]] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def build(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.tokenized = [self._tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized) if self.tokenized else None

    def query(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        if not self.bm25:
            return []
        q_tokens = self._tokenize(query_text)
        scores = self.bm25.get_scores(q_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            chunk = self.chunks[idx]
            results.append({"text": chunk["text"], "metadata": chunk["metadata"], "score": float(score)})
        return results

    def save(self, path: Path) -> None:
        with path.open("wb") as f:
            pickle.dump({"chunks": self.chunks, "tokenized": self.tokenized}, f)

    def load(self, path: Path) -> None:
        with path.open("rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.tokenized = data["tokenized"]
        self.bm25 = BM25Okapi(self.tokenized) if self.tokenized else None
