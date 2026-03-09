from __future__ import annotations

from typing import Any


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def calculate_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    s = summary.get("summary", {})

    total_tax = s.get("total_tax")
    agi = s.get("adjusted_gross_income")
    fed_withheld = s.get("federal_tax_withheld")

    ma_tax = s.get("ma_tax")
    ma_taxable_income = s.get("ma_taxable_income")
    state_withheld = s.get("state_tax_withheld")

    out: dict[str, Any] = {
        "effective_federal_tax_rate": None,
        "estimated_federal_refund": None,
        "effective_ma_tax_rate": None,
        "estimated_ma_refund": None,
        "federal_vs_state_tax_difference": None,
        "missing_fields": [],
    }

    if total_tax is not None and agi is not None:
        rate = _safe_div(float(total_tax), float(agi))
        out["effective_federal_tax_rate"] = None if rate is None else round(rate * 100, 2)
    else:
        out["missing_fields"].append("total_tax or adjusted_gross_income")

    if fed_withheld is not None and total_tax is not None:
        out["estimated_federal_refund"] = round(float(fed_withheld) - float(total_tax), 2)
    else:
        out["missing_fields"].append("federal_tax_withheld or total_tax")

    if ma_tax is not None and ma_taxable_income is not None:
        ma_rate = _safe_div(float(ma_tax), float(ma_taxable_income))
        out["effective_ma_tax_rate"] = None if ma_rate is None else round(ma_rate * 100, 2)
    else:
        out["missing_fields"].append("ma_tax or ma_taxable_income")

    if state_withheld is not None and ma_tax is not None:
        out["estimated_ma_refund"] = round(float(state_withheld) - float(ma_tax), 2)
    else:
        out["missing_fields"].append("state_tax_withheld or ma_tax")

    if total_tax is not None and ma_tax is not None:
        out["federal_vs_state_tax_difference"] = round(float(total_tax) - float(ma_tax), 2)

    return out
