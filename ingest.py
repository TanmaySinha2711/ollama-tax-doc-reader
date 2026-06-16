from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from chunker import chunk_documents
from config import AppConfig
from keyword_search import KeywordIndex
from pdf_parser import discover_pdfs, parse_pdfs
from structured_extractor import extract_structured_data
from vector_store import VectorStore


def _folder_signature(folder: Path) -> str:
    """
    Compute a SHA-256 hash of the folder contents for change detection.

    We hash a string composed of every PDF's:
      - Relative path (so moving files changes the signature)
      - File size in bytes  (so editing the content changes it)
      - Last-modified timestamp (so touching a file changes it)

    This is faster than hashing actual file contents because stat calls
    don't read the file data. However, a file modified in-place without
    changing its size or mtime (extremely rare) would NOT be detected.
    """
    pdfs = sorted([p for p in folder.rglob("*.pdf") if p.is_file()])
    # Build a deterministic string: one line per PDF with :: delimiters
    payload = "\n".join(f"{p.relative_to(folder)}::{p.stat().st_size}::{int(p.stat().st_mtime)}" for p in pdfs)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_state(path: Path) -> dict[str, Any]:
    """Load the ingestion state JSON from disk. Returns empty dict if missing."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(path: Path, state: dict[str, Any]) -> None:
    """Persist the ingestion state so we can skip unchanged folders next time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _knowledge_docs(config: AppConfig) -> list[dict[str, Any]]:
    """
    Load reference markdown files as pseudo-documents.

    Knowledge files (e.g. tax_rules_massachusetts.md) are treated as
    first-class documents during ingestion — they get chunked, embedded,
    and stored in the vector store alongside PDFs. This means the LLM
    can retrieve tax rules the same way it retrieves form data.

    The doc_type is set to "knowledge_base" so downstream components
    (like structured_extractor) can skip them if needed.
    """
    docs: list[dict[str, Any]] = []
    for md in sorted(config.knowledge_dir.glob("*.md")):
        text = md.read_text(encoding="utf-8").strip()
        if not text:
            continue
        # Build a document dict that mimics the output of pdf_parser.parse_pdfs()
        # so chunker.py doesn't need to know about knowledge docs.
        docs.append(
            {
                "path": str(md),
                "filename": md.name,
                "doc_type": "knowledge_base",
                "tax_year": None,
                "pages": [{"page_num": 1, "text": text, "tables": []}],
                "page_count": 1,
            }
        )
    return docs


def ingest_folder(
    folder: Path,
    config: AppConfig,
    vector_store: VectorStore,
    keyword_index: KeywordIndex,
    force: bool = False,
) -> dict[str, Any]:
    """
    Full document ingestion pipeline.

    Steps (in order):
      1. Compute folder signature → check if anything changed → skip if not.
      2. Discover all PDFs in the folder (recursive).
      3. Parse each PDF with pdfplumber (fallback PyMuPDF).
      4. Load knowledge/*.md files as extra documents.
      5. Chunk all documents into overlapping token windows.
      6. Clear and re-populate the vector store (ChromaDB).
      7. Build and persist the BM25 keyword index.
      8. Run regex-based structured extraction on tax forms.
      9. Save ingestion state (folder path + signature + counts).

    Parameters
    ----------
    folder : Path
        Directory containing tax PDFs (may include subdirectories).
    config : AppConfig
        Application settings (chunk sizes, top-Ks, paths, etc.).
    vector_store : VectorStore
        ChromaDB wrapper — will be cleared and rebuilt.
    keyword_index : KeywordIndex
        BM25 index — will be built from scratch.
    force : bool
        If True, skip the change-detection check and re-ingest unconditionally.
    """
    folder = folder.expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder does not exist or is not a directory: {folder}")

    # ── Change detection ──────────────────────────────────────────────
    signature = _folder_signature(folder)
    state = _load_state(config.ingestion_state_path)
    if not force and state.get("folder") == str(folder) and state.get("signature") == signature:
        # Same folder, same files — nothing to do. Just reload the BM25 index
        # from disk so it's ready for queries.
        if config.bm25_path.exists() and vector_store.count() > 0:
            keyword_index.load(config.bm25_path)
            return {"status": "skipped", "reason": "No changes detected", "folder": str(folder)}

    # ── Parse ─────────────────────────────────────────────────────────
    pdf_paths = discover_pdfs(folder)
    if not pdf_paths:
        raise ValueError("No PDF files found in the selected folder.")

    docs = parse_pdfs(pdf_paths)
    knowledge_docs = _knowledge_docs(config)
    all_docs = docs + knowledge_docs

    # ── Chunk ─────────────────────────────────────────────────────────
    chunks = chunk_documents(
        all_docs,
        chunk_size_tokens=config.chunk_size_tokens,
        chunk_overlap_tokens=config.chunk_overlap_tokens,
    )

    # ── Vector store ──────────────────────────────────────────────────
    # Clear the old collection first, then add new chunks. Embeddings are
    # computed automatically by ChromaDB via our callback (EmbeddingClient).
    vector_store.clear()
    vector_store.add_chunks(chunks)

    # ── Keyword index ─────────────────────────────────────────────────
    keyword_index.build(chunks)
    keyword_index.save(config.bm25_path)

    # ── Structured extraction ─────────────────────────────────────────
    # Only runs against PDF docs (not knowledge base markdown files).
    # Results are saved to data/structured/tax_summary.json.
    structured_summary = extract_structured_data(docs, config.structured_dir)

    # ── Save state ────────────────────────────────────────────────────
    _save_state(
        config.ingestion_state_path,
        {
            "folder": str(folder),
            "signature": signature,
            "pdf_count": len(pdf_paths),
            "chunk_count": len(chunks),
        },
    )

    return {
        "status": "ingested",
        "folder": str(folder),
        "pdf_count": len(pdf_paths),
        "chunk_count": len(chunks),
        "structured_summary": structured_summary,
    }
