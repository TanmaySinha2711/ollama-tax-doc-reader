from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


class KeywordIndex:
    """
    BM25 keyword search index.

    BM25 is a bag-of-words ranking function — the modern successor to TF-IDF.
    It scores each chunk by how many of the query terms appear in it, weighted
    by how rare those terms are across the whole corpus.

    Why BM25 in addition to vector search? See section 4 of the companion doc.
    In short: BM25 catches EXACT term matches ("Box 17", "line 35a") that
    vector embeddings might overlook because they focus on meaning, not spelling.

    Scoring intuition:
      - A chunk containing "federal income tax withheld" gets a high score
        when the query is "federal tax withholding" (all words match).
      - If "withholding" appears in only 2 out of 100 chunks, that word
        contributes more than common words like "tax" (appears everywhere).
      - Longer chunks are slightly penalized (length normalization).

    Persisted to disk via pickle at data/bm25_index.pkl.
    """

    def __init__(self) -> None:
        self.bm25: BM25Okapi | None = None
        self.chunks: list[dict[str, Any]] = []
        self.tokenized: list[list[str]] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace tokenizer: lowercase, split on spaces.

        This is intentionally naive — no stemming, no stop-word removal,
        no punctuation handling. BM25 handles stop-words fine (they get
        low IDF because they appear everywhere), and stemming would be
        more complexity than it's worth for tax form text.
        """
        return text.lower().split()

    def build(self, chunks: list[dict[str, Any]]) -> None:
        """
        Build the BM25 index from a list of chunk dicts.

        Stores both the chunk dicts (for returning results) and their
        tokenized forms (for BM25 scoring).
        """
        self.chunks = chunks
        self.tokenized = [self._tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized) if self.tokenized else None

    def query(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        """
        Run a BM25 query, return the top_k chunks with scores.

        Each result dict:
          text     – chunk content
          metadata – source, page, chunk_index, etc.
          score    – BM25 relevance score (positive float, higher = better)

        BM25 scores are not normalized to any fixed range — they vary with
        corpus size and query length. That's fine because RRF (Reciprocal
        Rank Fusion) only uses the RANK, not the raw score.
        """
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
        """
        Persist the index to disk via pickle.

        We save chunks + tokenized list, NOT the BM25Okapi object itself.
        This avoids compatibility issues if rank_bm25 changes its internals
        between versions. We rebuild BM25Okapi on load.
        """
        with path.open("wb") as f:
            pickle.dump({"chunks": self.chunks, "tokenized": self.tokenized}, f)

    def load(self, path: Path) -> None:
        """
        Restore the index from a pickle file saved by save().

        Reconstructs the BM25Okapi object from the tokenized corpus.
        """
        with path.open("rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.tokenized = data["tokenized"]
        self.bm25 = BM25Okapi(self.tokenized) if self.tokenized else None
