from __future__ import annotations

from typing import Any

import tiktoken


def _get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    enc = _get_encoder()
    return len(enc.encode(text))


def _split_text_by_tokens(text: str, chunk_size: int, overlap: int) -> list[str]:
    enc = _get_encoder()
    tokens = enc.encode(text)
    if not tokens:
        return []

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens).strip())
        if end == len(tokens):
            break
        start += step
    return [c for c in chunks if c]


def chunk_documents(
    docs: list[dict[str, Any]],
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    for doc in docs:
        for page in doc.get("pages", []):
            page_text = (page.get("text") or "").strip()
            if not page_text:
                continue

            has_table = "[TABLE]" in page_text
            page_chunks = _split_text_by_tokens(page_text, chunk_size_tokens, chunk_overlap_tokens)
            for idx, text in enumerate(page_chunks):
                chunks.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": doc["filename"],
                            "path": doc["path"],
                            "doc_type": doc.get("doc_type"),
                            "tax_year": doc.get("tax_year"),
                            "page": page["page_num"],
                            "chunk_index": idx,
                            "has_table": has_table,
                        },
                    }
                )
    return chunks
