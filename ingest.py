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
    pdfs = sorted([p for p in folder.rglob("*.pdf") if p.is_file()])
    payload = "\n".join(f"{p.relative_to(folder)}::{p.stat().st_size}::{int(p.stat().st_mtime)}" for p in pdfs)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _knowledge_docs(config: AppConfig) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for md in sorted(config.knowledge_dir.glob("*.md")):
        text = md.read_text(encoding="utf-8").strip()
        if not text:
            continue
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
    folder = folder.expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder does not exist or is not a directory: {folder}")

    signature = _folder_signature(folder)
    state = _load_state(config.ingestion_state_path)
    if not force and state.get("folder") == str(folder) and state.get("signature") == signature:
        if config.bm25_path.exists() and vector_store.count() > 0:
            keyword_index.load(config.bm25_path)
            return {"status": "skipped", "reason": "No changes detected", "folder": str(folder)}

    pdf_paths = discover_pdfs(folder)
    if not pdf_paths:
        raise ValueError("No PDF files found in the selected folder.")

    docs = parse_pdfs(pdf_paths)
    knowledge_docs = _knowledge_docs(config)
    all_docs = docs + knowledge_docs

    chunks = chunk_documents(
        all_docs,
        chunk_size_tokens=config.chunk_size_tokens,
        chunk_overlap_tokens=config.chunk_overlap_tokens,
    )

    vector_store.clear()
    vector_store.add_chunks(chunks)

    keyword_index.build(chunks)
    keyword_index.save(config.bm25_path)

    structured_summary = extract_structured_data(docs, config.structured_dir)

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
