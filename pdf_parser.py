from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz
import pdfplumber


def discover_pdfs(folder: Path) -> list[Path]:
    """
    Recursively find all PDF files under *folder*.

    Returns a sorted list of Path objects so that ingestion order is
    deterministic across runs (important for reproducible chunk IDs).
    """
    return sorted([p for p in folder.rglob("*.pdf") if p.is_file()])


def infer_tax_year(text: str, filename: str) -> str | None:
    """
    Heuristically determine the tax year from filename + first-page text.

    Looks for 4-digit numbers in the range 2000-2100 appearing in either
    the filename or the first 2000 characters of text. Returns the FIRST
    match found (so "2023 W-2" → "2023").

    This is a heuristic — a document that happens to mention "2025" as
    a projection would be misidentified. For production, you'd want a
    more robust rule (e.g., prefer the latest year, or look for "Tax Year" labels).
    """
    matches = re.findall(r"\b(20\d{2})\b", f"{filename} {text[:2000]}")
    for m in matches:
        year = int(m)
        if 2000 <= year <= 2100:
            return m
    return None


def infer_doc_type(filename: str, text: str) -> str:
    """
    Classify the document type from filename + first ~3000 chars of text.

    Order matters — we check W-2 before 1040 because "1040" might appear
    in a Schedule that ships with a 1040, but the word "schedule" combined
    with "1040" gets its own label.

    Returns one of:
      w2, 1099-div, 1099-b, 1040, 1040-schedule, ma-form-1,
      india-us-treaty, tax-document (fallback generic)
    """
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
    """
    Convert a pdfplumber table (list of rows of cells) to GitHub-flavored
    markdown table syntax.

    pdfplumber returns tables as lists of rows, where each row is a list
    of cell values (or None for empty cells). We:
      1. Replace None with "".
      2. Normalize all rows to the same column count.
      3. Use the first row as the markdown header.
      4. Add a separator row (|---|---|).
      5. Render remaining rows as data rows.

    Returns an empty string if the table has no rows after cleanup.
    """
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
    """
    Parse a single PDF file into a structured dictionary.

    Strategy:
      1. Primary: pdfplumber — excellent text + table extraction.
         Tables are converted to markdown and prefixed with [TABLE].
      2. Fallback: PyMuPDF (fitz) — used on any page where pdfplumber
         returned empty text. This handles scanned PDFs or unusual layouts
         that pdfplumber can't parse.

    Returns a dict with:
      path       – absolute path to the file
      filename   – just the filename
      doc_type   – inferred form type (w2, 1040, etc.)
      tax_year   – inferred year or None
      pages      – list of dicts, each with page_num, text, tables
      page_count – total pages

    The [TABLE] marker in the text allows downstream components (chunker,
    LLM) to distinguish tabular data from running text.
    """
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

    # ── Fallback: PyMuPDF for empty pages ──────────────────────────
    # pdfplumber sometimes extracts nothing from image-heavy PDFs.
    # PyMuPDF handles a wider range of formats. We only re-extract
    # pages that came back empty.
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

    # ── Metadata inference ────────────────────────────────────────
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
    """Convenience: parse a list of PDF paths."""
    docs = []
    for path in paths:
        docs.append(parse_pdf(path))
    return docs
