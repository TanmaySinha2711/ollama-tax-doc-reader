from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# ── Shared regex pattern for capturing dollar/numeric values ──────────
# This is the core number-capture regex used by ALL field extractors.
# Breakdown:
#   \$?       optional dollar sign
#   \(?       optional opening paren (for negative numbers in accounting format: (500))
#   (         capture group start
#   [0-9]     first digit (ensures at least one digit)
#   [0-9,]*   zero or more digits or commas (handles 1,234.56)
#   (?:\.\d{1,2})?  optional decimal with 1-2 fractional digits
#   )         capture group end
#   \)?       optional closing paren
NUM_RE = r"\$?\(?([0-9][0-9,]*(?:\.\d{1,2})?)\)?"


def _to_number(val: str | None) -> float | None:
    """Convert a matched string to a float, handling commas and None."""
    if val is None:
        return None
    cleaned = val.replace(",", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _guard_unrealistic(value: float | None, *, minimum: float = 50.0) -> float | None:
    """
    Reject values below a threshold to avoid matching the wrong number.

    For example, "Box 1: Wages" might be near "Page 1 of 3" — without
    this guard, we might capture "1" instead of the wage figure.
    The minimum varies by field (wages ≥ $50, tax ≥ $1).
    """
    if value is None:
        return None
    return value if value >= minimum else None


def _extract_match(text: str, pattern: str) -> tuple[float | None, dict[str, Any] | None]:
    """
    Apply a single regex pattern and return (value, audit_trail).

    The audit trail records:
      - The exact regex pattern used.
      - The matched text snippet (for human verification).
      - Character positions in the source text.

    Returns (None, None) if the pattern doesn't match or the value
    can't be parsed as a number.
    """
    m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return None, None

    # Find the first numeric capture group in the match
    value: float | None = None
    for idx in range(1, len(m.groups()) + 1):
        maybe = _to_number(m.group(idx))
        if maybe is not None:
            value = maybe
            break

    if value is None:
        return None, None

    snippet = m.group(0)
    snippet = re.sub(r"\s+", " ", snippet).strip()[:220]  # truncate long snippets
    audit = {
        "pattern": pattern,
        "matched_text": snippet,
        "match_start": m.start(),
        "match_end": m.end(),
    }
    return value, audit


def _extract_field(
    text: str,
    field_name: str,
    patterns: list[str],
    audit: dict[str, Any],
    minimum: float | None = None,
) -> float | None:
    """
    Try multiple regex patterns for a single field, return the first match.

    Each field has 1-2 patterns (e.g. "Box 1: Wages..." or "1 Wages, tips...").
    We try them in order and return the first valid match. This handles
    variations in how tax forms format their labels.

    If *minimum* is set, values below it are discarded (guard against
    false positives matching page numbers or small annotation figures).
    """
    for pat in patterns:
        value, raw_audit = _extract_match(text, pat)
        if value is None:
            continue

        if minimum is not None:
            value = _guard_unrealistic(value, minimum=minimum)
        if value is None:
            continue

        audit[field_name] = {
            "confidence": "high",
            "method": "regex_line_box",
            "value": value,
            **(raw_audit or {}),
        }
        return value

    return None


# ── Form-specific extraction functions ──────────────────────────────
# Each function knows the box numbers and labels of a specific IRS/state form.
# Patterns use:
#   \b  word boundary (so "1" doesn't match "10" or "100")
#   \D  non-digit (to prevent "16000" matching "16" in a different context)
#   NUM_RE defined above for the actual value capture


def _extract_w2_fields(text: str, audit: dict[str, Any]) -> dict[str, float | None]:
    """
    Extract key fields from IRS Form W-2 (Wage and Tax Statement).

    Fields:
      w2_wages           - Box 1: Wages, tips, other compensation
      federal_tax_withheld - Box 2: Federal income tax withheld
      w2_state_wages     - Box 16: State wages, tips, etc.
      state_tax_withheld - Box 17: State income tax
    """
    fields = {
        "w2_wages": _extract_field(
            text,
            "w2_wages",
            [
                rf"box\s*1\b[^\n$]*wages[^\n$]*{NUM_RE}",
                rf"\b1\s+wages,\s*tips,\s*other\s*compensation[^\n$]*{NUM_RE}",
            ],
            audit,
            minimum=50.0,  # wages below $50 are unrealistic
        ),
        "federal_tax_withheld": _extract_field(
            text,
            "federal_tax_withheld",
            [
                rf"box\s*2\b[^\n$]*federal\s+income\s+tax\s+withheld[^\n$]*{NUM_RE}",
                rf"\b2\s+federal\s+income\s+tax\s+withheld[^\n$]*{NUM_RE}",
            ],
            audit,
            minimum=1.0,  # even $1 of withholding is meaningful
        ),
        "w2_state_wages": _extract_field(
            text,
            "w2_state_wages",
            [rf"\b16\s+state\s+wages[^\n$]*{NUM_RE}"],
            audit,
            minimum=50.0,
        ),
        "state_tax_withheld": _extract_field(
            text,
            "state_tax_withheld",
            [rf"\b17\s+state\s+income\s+tax[^\n$]*{NUM_RE}"],
            audit,
            minimum=1.0,
        ),
    }
    return fields


def _extract_1040_fields(text: str, audit: dict[str, Any]) -> dict[str, float | None]:
    """
    Extract key fields from IRS Form 1040 (U.S. Individual Income Tax Return).

    Fields follow standard 1040 line numbers:
      total_income           - Line 9: Total income
      adjusted_gross_income  - Line 11: Adjusted gross income
      taxable_income         - Line 15: Taxable income
      total_tax              - Line 24: Total tax
      federal_tax_withheld   - Line 25a: Federal income tax withheld
      refund_amount          - Line 35a: Amount refunded
    """
    fields = {
        "total_income": _extract_field(
            text,
            "total_income",
            [rf"(?:line\s*)?9\b[^\n$]*total\s+income[^\n$]*{NUM_RE}"],
            audit,
            minimum=50.0,
        ),
        "adjusted_gross_income": _extract_field(
            text,
            "adjusted_gross_income",
            [rf"(?:line\s*)?11\b[^\n$]*adjusted\s+gross\s+income[^\n$]*{NUM_RE}"],
            audit,
            minimum=50.0,
        ),
        "taxable_income": _extract_field(
            text,
            "taxable_income",
            [rf"(?:line\s*)?15\b[^\n$]*taxable\s+income[^\n$]*{NUM_RE}"],
            audit,
            minimum=50.0,
        ),
        "total_tax": _extract_field(
            text,
            "total_tax",
            [rf"(?:line\s*)?24\b[^\n$]*total\s+tax[^\n$]*{NUM_RE}"],
            audit,
            minimum=1.0,
        ),
        "federal_tax_withheld": _extract_field(
            text,
            "federal_tax_withheld",
            [rf"(?:line\s*)?25a\b[^\n$]*federal\s+income\s+tax\s+withheld[^\n$]*{NUM_RE}"],
            audit,
            minimum=1.0,
        ),
        "refund_amount": _extract_field(
            text,
            "refund_amount",
            [rf"(?:line\s*)?35a\b[^\n$]*refunded[^\n$]*{NUM_RE}", rf"(?:line\s*)?35a\b[^\n$]*{NUM_RE}"],
            audit,
            minimum=1.0,
        ),
    }
    return fields


def _extract_ma_form1_fields(text: str, audit: dict[str, Any]) -> dict[str, float | None]:
    """
    Extract key fields from Massachusetts Form 1 (Resident Income Tax Return).

    Fields:
      ma_taxable_income - Massachusetts taxable income
      ma_tax            - Massachusetts tax (or tax liability)
      ma_refund         - Massachusetts refund amount
      state_tax_withheld - State withholding amount
    """
    fields = {
        "ma_taxable_income": _extract_field(
            text,
            "ma_taxable_income",
            [rf"massachusetts\s+taxable\s+income[^\n$]*{NUM_RE}", rf"taxable\s+income[^\n$]*massachusetts[^\n$]*{NUM_RE}"],
            audit,
            minimum=50.0,
        ),
        "ma_tax": _extract_field(
            text,
            "ma_tax",
            [rf"massachusetts\s+tax(?:\s+liability)?[^\n$]*{NUM_RE}", rf"form\s*1[^\n$]*tax[^\n$]*{NUM_RE}"],
            audit,
            minimum=1.0,
        ),
        "ma_refund": _extract_field(
            text,
            "ma_refund",
            [rf"massachusetts\s+refund[^\n$]*{NUM_RE}", rf"form\s*1[^\n$]*refund[^\n$]*{NUM_RE}"],
            audit,
            minimum=1.0,
        ),
        "state_tax_withheld": _extract_field(
            text,
            "state_tax_withheld",
            [rf"state\s+withholding[^\n$]*{NUM_RE}"],
            audit,
            minimum=1.0,
        ),
    }
    return fields


def _extract_generic_fields(text: str, audit: dict[str, Any]) -> dict[str, float | None]:
    """
    Extract fields that may appear on any form type.

    These use broad patterns (\D{0,24} = up to 24 non-digit characters)
    to find numbers near relevant labels. They run AFTER the form-specific
    extraction so they don't interfere with precise box-level patterns.
    """
    return {
        "capital_gains": _extract_field(text, "capital_gains", [rf"capital\s+gain(?:s)?\D{{0,24}}{NUM_RE}"], audit, minimum=1.0),
        "dividends": _extract_field(text, "dividends", [rf"dividend(?:s)?\D{{0,24}}{NUM_RE}"], audit, minimum=1.0),
        "deductions": _extract_field(text, "deductions", [rf"deduction(?:s)?\D{{0,24}}{NUM_RE}"], audit, minimum=1.0),
        "credits": _extract_field(text, "credits", [rf"credit(?:s)?\D{{0,24}}{NUM_RE}"], audit, minimum=1.0),
    }


def _empty_fields() -> dict[str, float | None]:
    """
    Return a template with ALL known fields set to None.

    This ensures the aggregation logic always has the same keys regardless
    of which form-specific extractors actually ran.
    """
    return {
        "w2_wages": None,
        "w2_state_wages": None,
        "total_income": None,
        "adjusted_gross_income": None,
        "taxable_income": None,
        "federal_tax_withheld": None,
        "state_tax_withheld": None,
        "refund_amount": None,
        "total_tax": None,
        "capital_gains": None,
        "dividends": None,
        "deductions": None,
        "credits": None,
        "ma_taxable_income": None,
        "ma_tax": None,
        "ma_refund": None,
    }


def extract_fields_from_text(text: str, doc_type: str | None) -> tuple[dict[str, float | None], dict[str, Any]]:
    """
    Run the appropriate form-specific extractor based on doc_type.

    Strategy:
      1. Start with all fields set to None.
      2. If doc_type is known (w2, 1040, ma-form-1), run the form-specific
         extractor. It fills in only the fields it knows about.
      3. Run the generic extractor (capital gains, dividends, etc.) which
         may find values on ANY form type.
      4. Generic results override None values but don't erase existing values.

    Returns (fields_dict, audit_dict).
    """
    fields = _empty_fields()
    audit: dict[str, Any] = {}

    normalized_doc_type = (doc_type or "").lower()
    if normalized_doc_type == "w2":
        fields.update(_extract_w2_fields(text, audit))
    elif normalized_doc_type in {"1040", "1040-schedule"}:
        fields.update(_extract_1040_fields(text, audit))
    elif normalized_doc_type == "ma-form-1":
        fields.update(_extract_ma_form1_fields(text, audit))

    generic = _extract_generic_fields(text, audit)
    for key, value in generic.items():
        if value is not None:
            fields[key] = value

    return fields, audit


def extract_structured_data(docs: list[dict[str, Any]], structured_dir: Path) -> dict[str, Any]:
    """
    Run structured extraction on all tax PDF documents and save results.

    For each document (skipping knowledge_base docs):
      1. Concatenate all page text.
      2. Run extract_fields_from_text() with the document's doc_type.
      3. Save per-document results as data/structured/{filename}.json.
      4. Aggregate all fields into a summary (values are SUMMED across
         documents — e.g., two W-2s contribute to one w2_wages total).
      5. Build an audit trail showing which document & pattern produced
         each value.

    The aggregate result is saved as data/structured/tax_summary.json
    and also returned for immediate use.

    IMPORTANT: Values are SUMMED, not averaged. If you have two W-2s with
    $50k and $75k wages, the summary shows w2_wages = $125k. This is the
    correct behavior for tax return preparation.
    """
    structured_dir.mkdir(parents=True, exist_ok=True)

    per_doc: list[dict[str, Any]] = []
    summary: dict[str, float] = {}
    summary_audit: dict[str, list[dict[str, Any]]] = {}
    years: set[str] = set()

    for doc in docs:
        # Skip knowledge base markdown files — they don't have tax form fields
        if (doc.get("doc_type") or "").lower() == "knowledge_base":
            continue

        full_text = "\n\n".join(p.get("text", "") for p in doc.get("pages", []))
        fields, audit = extract_fields_from_text(full_text, doc.get("doc_type"))
        entry = {
            "filename": doc["filename"],
            "doc_type": doc.get("doc_type"),
            "tax_year": doc.get("tax_year"),
            "fields": fields,
            "audit": {
                "document": doc["filename"],
                "doc_type": doc.get("doc_type"),
                "method": "line_box_specific_regex",
                "field_evidence": audit,
            },
        }
        per_doc.append(entry)

        if doc.get("tax_year"):
            years.add(doc["tax_year"])

        # Aggregate: sum values across all documents
        for key, value in fields.items():
            if value is None:
                continue
            summary[key] = summary.get(key, 0.0) + value
            # Attach audit trail for this field
            if key in audit:
                evidence = dict(audit[key])
                evidence["source_document"] = doc["filename"]
                summary_audit.setdefault(key, []).append(evidence)

        # Save per-document extraction
        out_path = structured_dir / f"{Path(doc['filename']).stem}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2)

    aggregate = {
        "tax_years": sorted(years),
        "summary": summary,
        "summary_audit": summary_audit,
        "documents": [d["filename"] for d in per_doc],
    }
    with (structured_dir / "tax_summary.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    return aggregate


def load_tax_summary(structured_dir: Path) -> dict[str, Any]:
    """
    Load the aggregated tax summary from disk.

    Returns an empty template if the file doesn't exist yet
    (no ingestion has been performed).
    """
    path = structured_dir / "tax_summary.json"
    if not path.exists():
        return {"tax_years": [], "summary": {}, "summary_audit": {}, "documents": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
