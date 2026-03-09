from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


NUM_RE = r"\$?\(?([0-9][0-9,]*(?:\.\d{1,2})?)\)?"


def _to_number(val: str | None) -> float | None:
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
    if value is None:
        return None
    return value if value >= minimum else None


def _extract_match(text: str, pattern: str) -> tuple[float | None, dict[str, Any] | None]:
    m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return None, None

    value: float | None = None
    for idx in range(1, len(m.groups()) + 1):
        maybe = _to_number(m.group(idx))
        if maybe is not None:
            value = maybe
            break

    if value is None:
        return None, None

    snippet = m.group(0)
    snippet = re.sub(r"\s+", " ", snippet).strip()[:220]
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


def _extract_w2_fields(text: str, audit: dict[str, Any]) -> dict[str, float | None]:
    fields = {
        "w2_wages": _extract_field(
            text,
            "w2_wages",
            [
                rf"box\s*1\b[^\n$]*wages[^\n$]*{NUM_RE}",
                rf"\b1\s+wages,\s*tips,\s*other\s*compensation[^\n$]*{NUM_RE}",
            ],
            audit,
            minimum=50.0,
        ),
        "federal_tax_withheld": _extract_field(
            text,
            "federal_tax_withheld",
            [
                rf"box\s*2\b[^\n$]*federal\s+income\s+tax\s+withheld[^\n$]*{NUM_RE}",
                rf"\b2\s+federal\s+income\s+tax\s+withheld[^\n$]*{NUM_RE}",
            ],
            audit,
            minimum=1.0,
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
    return {
        "capital_gains": _extract_field(text, "capital_gains", [rf"capital\s+gain(?:s)?\D{{0,24}}{NUM_RE}"], audit, minimum=1.0),
        "dividends": _extract_field(text, "dividends", [rf"dividend(?:s)?\D{{0,24}}{NUM_RE}"], audit, minimum=1.0),
        "deductions": _extract_field(text, "deductions", [rf"deduction(?:s)?\D{{0,24}}{NUM_RE}"], audit, minimum=1.0),
        "credits": _extract_field(text, "credits", [rf"credit(?:s)?\D{{0,24}}{NUM_RE}"], audit, minimum=1.0),
    }


def _empty_fields() -> dict[str, float | None]:
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
    structured_dir.mkdir(parents=True, exist_ok=True)

    per_doc: list[dict[str, Any]] = []
    summary: dict[str, float] = {}
    summary_audit: dict[str, list[dict[str, Any]]] = {}
    years: set[str] = set()

    for doc in docs:
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

        for key, value in fields.items():
            if value is None:
                continue
            summary[key] = summary.get(key, 0.0) + value
            if key in audit:
                evidence = dict(audit[key])
                evidence["source_document"] = doc["filename"]
                summary_audit.setdefault(key, []).append(evidence)

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
    path = structured_dir / "tax_summary.json"
    if not path.exists():
        return {"tax_years": [], "summary": {}, "summary_audit": {}, "documents": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
