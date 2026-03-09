from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz
import pdfplumber


def discover_pdfs(folder: Path) -> list[Path]:
    return sorted([p for p in folder.rglob("*.pdf") if p.is_file()])


def infer_tax_year(text: str, filename: str) -> str | None:
    matches = re.findall(r"\b(20\d{2})\b", f"{filename} {text[:2000]}")
    for m in matches:
        year = int(m)
        if 2000 <= year <= 2100:
            return m
    return None


def infer_doc_type(filename: str, text: str) -> str:
    hay = f"{filename.lower()}\n{text.lower()[:3000]}"
    if "w-2" in hay or "w2" in hay:
        return "w2"
    if "1099-div" in hay or "1099 div" in hay:
        return "1099-div"
    if "1099-b" in hay or "1099 b" in hay:
        return "1099-b"
    if "form 1040" in hay or "1040" in hay:
        return "1040"
    if "schedule" in hay and "1040" in hay:
        return "1040-schedule"
    if "massachusetts" in hay or "form 1" in hay:
        return "ma-form-1"
    if "india" in hay and "treaty" in hay:
        return "india-us-treaty"
    return "tax-document"


def _table_to_markdown(table: list[list[Any]]) -> str:
    cleaned = [["" if cell is None else str(cell).strip() for cell in row] for row in table if row]
    if not cleaned:
        return ""
    max_cols = max(len(row) for row in cleaned)
    normalized = [row + [""] * (max_cols - len(row)) for row in cleaned]
    header = normalized[0]
    sep = ["---"] * max_cols
    body = normalized[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def parse_pdf(path: Path) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    first_page_text = ""

    with pdfplumber.open(path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            tables = page.extract_tables() or []
            md_tables = [_table_to_markdown(table) for table in tables if table]
            md_tables = [t for t in md_tables if t.strip()]

            content_parts = []
            if text:
                content_parts.append(text)
            if md_tables:
                content_parts.append("\n\n".join([f"[TABLE]\n{tbl}" for tbl in md_tables]))

            page_text = "\n\n".join(content_parts).strip()
            pages.append(
                {
                    "page_num": idx,
                    "text": page_text,
                    "tables": md_tables,
                }
            )
            if idx == 1:
                first_page_text = page_text

    if any(not p["text"] for p in pages):
        fitz_doc = fitz.open(path)
        for p in pages:
            if not p["text"]:
                try:
                    fallback_text = fitz_doc.load_page(p["page_num"] - 1).get_text("text").strip()
                except Exception:
                    fallback_text = ""
                if fallback_text:
                    p["text"] = fallback_text

    non_empty_pages = [p for p in pages if p["text"].strip()]
    doc_text_preview = non_empty_pages[0]["text"] if non_empty_pages else ""
    tax_year = infer_tax_year(first_page_text or doc_text_preview, path.name)
    doc_type = infer_doc_type(path.name, first_page_text or doc_text_preview)

    return {
        "path": str(path),
        "filename": path.name,
        "doc_type": doc_type,
        "tax_year": tax_year,
        "pages": pages,
        "page_count": len(pages),
    }


def parse_pdfs(paths: list[Path]) -> list[dict[str, Any]]:
    docs = []
    for path in paths:
        docs.append(parse_pdf(path))
    return docs
