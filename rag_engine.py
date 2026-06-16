from __future__ import annotations

from collections import defaultdict
from typing import Any


def _chunk_key(metadata: dict[str, Any]) -> str:
    """
    Build a unique, deterministic key for a chunk.

    Uses source filename + page number + chunk_index. This is the
    deduplication key for RRF — if the same chunk appears in both
    the vector results and keyword results, we merge them into one.
    """
    src = metadata.get("source", "")
    page = metadata.get("page", "")
    idx = metadata.get("chunk_index", "")
    return f"{src}::{page}::{idx}"


def reciprocal_rank_fusion(
    vector_results: list[dict[str, Any]], keyword_results: list[dict[str, Any]], k: int = 60
) -> list[dict[str, Any]]:
    """
    Fuse two ranked result lists into one using Reciprocal Rank Fusion (RRF).

    RRF formula for each chunk:

        score(chunk) = Σ 1 / (k + rank_i)   over all result lists i

    where rank_i is the chunk's position in list i (1-indexed).

    Why RRF over other fusion methods?
      - No training/weights needed — works out of the box.
      - Chunks appearing in BOTH lists get a score boost (they appear in
        two terms of the sum).
      - Robust to different scoring scales — it only uses ranks, not raw
        scores, so we don't need to normalize BM25 scores and cosine
        similarities to the same range.

    The constant k = 60 is the standard value from literature (Cormack et al.).
    It acts as a smoothing factor:
      - Large k: scores are flatter, rank position matters less.
      - Small k: top-ranked results dominate more heavily.
    k=60 works well across many domains without tuning.

    Parameters
    ----------
    vector_results : list[dict]
        Results from ChromaDB similarity search, each with "text", "metadata", "score".
    keyword_results : list[dict]
        Results from BM25 keyword search, same structure.
    k : int
        RRF constant (default 60).

    Returns
    -------
    list[dict]
        Fused results sorted by RRF score descending, each augmented with
        an "rrf_score" key.
    """
    fused: dict[str, dict[str, Any]] = {}
    scores = defaultdict(float)

    # Process vector results (rank 1..N)
    for rank, res in enumerate(vector_results, start=1):
        key = _chunk_key(res["metadata"])
        fused[key] = {"text": res["text"], "metadata": res["metadata"]}
        scores[key] += 1.0 / (k + rank)

    # Process keyword results (rank 1..N)
    for rank, res in enumerate(keyword_results, start=1):
        key = _chunk_key(res["metadata"])
        fused[key] = {"text": res["text"], "metadata": res["metadata"]}
        scores[key] += 1.0 / (k + rank)

    # Sort by RRF score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    output = []
    for key, score in ranked:
        item = fused[key]
        item["rrf_score"] = round(score, 6)
        output.append(item)
    return output


def format_context(chunks: list[dict[str, Any]], structured_data: dict[str, Any] | None = None) -> str:
    """
    Build a plain-text context block for the LLM prompt.

    Structure:
      1. [STRUCTURED_DATA] section — extracted numerical fields from regex.
      2. For each chunk: [SOURCE: filename page N] followed by chunk text.

    The [SOURCE:] markers serve double duty:
      - They tell the LLM which document each piece came from.
      - The query pipeline parses them out to display citations to the user.

    Example output:
        [STRUCTURED_DATA]
        {"w2_wages": 50000, "federal_tax_withheld": 6200}

        [SOURCE: w2_sample.pdf page 1]
        Box 1: Wages, tips, other compensation..... 50000.00
        Box 2: Federal income tax withheld......... 6200.00
    """
    parts: list[str] = []
    if structured_data:
        parts.append("[STRUCTURED_DATA]\n" + str(structured_data))

    for c in chunks:
        source = c["metadata"].get("source", "unknown")
        page = c["metadata"].get("page", "?")
        parts.append(f"[SOURCE: {source} page {page}]\n{c['text']}")
    return "\n\n".join(parts)
