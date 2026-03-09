from __future__ import annotations

from collections import defaultdict
from typing import Any


def _chunk_key(metadata: dict[str, Any]) -> str:
    src = metadata.get("source", "")
    page = metadata.get("page", "")
    idx = metadata.get("chunk_index", "")
    return f"{src}::{page}::{idx}"


def reciprocal_rank_fusion(
    vector_results: list[dict[str, Any]], keyword_results: list[dict[str, Any]], k: int = 60
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    scores = defaultdict(float)

    for rank, res in enumerate(vector_results, start=1):
        key = _chunk_key(res["metadata"])
        fused[key] = {"text": res["text"], "metadata": res["metadata"]}
        scores[key] += 1.0 / (k + rank)

    for rank, res in enumerate(keyword_results, start=1):
        key = _chunk_key(res["metadata"])
        fused[key] = {"text": res["text"], "metadata": res["metadata"]}
        scores[key] += 1.0 / (k + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    output = []
    for key, score in ranked:
        item = fused[key]
        item["rrf_score"] = round(score, 6)
        output.append(item)
    return output


def format_context(chunks: list[dict[str, Any]], structured_data: dict[str, Any] | None = None) -> str:
    parts: list[str] = []
    if structured_data:
        parts.append("[STRUCTURED_DATA]\n" + str(structured_data))

    for c in chunks:
        source = c["metadata"].get("source", "unknown")
        page = c["metadata"].get("page", "?")
        parts.append(f"[SOURCE: {source} page {page}]\n{c['text']}")
    return "\n\n".join(parts)
