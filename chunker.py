from __future__ import annotations

from typing import Any

import tiktoken


def _get_encoder() -> tiktoken.Encoding:
    """
    Get the tiktoken tokenizer.

    We use OpenAI's cl100k_base encoding (used by GPT-4, GPT-3.5-turbo,
    text-embedding-ada-002, etc.) as a generic tokenizer. It works well
    for English text even though we use a Qwen model — token counts are
    approximate anyway, and the key property we need is *consistent*
    chunk sizes, not absolute token accuracy matching the LLM.
    """
    return tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    """Return the number of tokens in *text*."""
    enc = _get_encoder()
    return len(enc.encode(text))


def _split_text_by_tokens(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split *text* into overlapping chunks of approximately *chunk_size* tokens.

    Algorithm:
      1. Encode the entire text into a token array.
      2. Start at position 0. Take *chunk_size* tokens → chunk.
      3. Slide forward by (chunk_size - overlap) tokens.
      4. Repeat until we've covered all tokens.

    Example (chunk_size=800, overlap=200):
      Chunk 0: tokens   0-800
      Chunk 1: tokens 600-1400   (slide by 600)
      Chunk 2: tokens 1200-2000
      ...

    Overlap ensures that a sentence spanning a chunk boundary appears
    in both chunks, giving retrieval multiple chances to find it.
    The last chunk may be shorter than *chunk_size*.
    """
    enc = _get_encoder()
    tokens = enc.encode(text)
    if not tokens:
        return []

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)  # how far to slide the window each time
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
    """
    Convert a list of parsed documents into a list of overlapping text chunks.

    Each chunk is a dict:
      text     – the chunk content (string)
      metadata – dict with source filename, path, doc_type, tax_year,
                 page number, chunk_index (within that page), and whether
                 the page contained a [TABLE] marker.

    Chunking is done per-page (not per-document) so that page boundaries
    are preserved in the metadata. This lets the LLM cite exact page numbers.
    """
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
