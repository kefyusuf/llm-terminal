"""Helpers for responsive results-table column layout."""

from __future__ import annotations


MIN_COLUMN_WIDTHS = {
    "inst": 8,
    "source": 10,
    "publisher": 8,
    "name": 18,
    "params": 6,
    "use_case": 5,
    "score": 8,
    "quant": 6,
    "mode": 8,
    "fit": 7,
    "download": 4,
}

EXPANDABLE_COLUMNS = ("name", "publisher", "use_case", "download")


def column_keys_for_width(available_width: int) -> list[str]:
    """Return visible column keys based on available table width."""
    if available_width >= 140:
        return [
            "inst",
            "source",
            "publisher",
            "name",
            "params",
            "use_case",
            "score",
            "quant",
            "mode",
            "fit",
            "download",
        ]
    if available_width >= 120:
        return [
            "inst",
            "source",
            "publisher",
            "name",
            "params",
            "score",
            "quant",
            "fit",
            "download",
        ]
    if available_width >= 100:
        return ["inst", "source", "name", "params", "score", "fit", "download"]
    return ["inst", "source", "name", "score", "download"]


def compute_column_widths(next_keys: list[str], available_width: int) -> dict[str, int]:
    """Compute responsive widths for visible result-table columns."""
    separator_budget = len(next_keys) + 2
    target_content_width = max(40, available_width - separator_budget)
    base_widths = {key: MIN_COLUMN_WIDTHS[key] for key in next_keys}
    min_total = sum(base_widths.values())
    extra = max(0, target_content_width - min_total)

    while extra > 0:
        expanded = False
        for key in EXPANDABLE_COLUMNS:
            if key in base_widths and extra > 0:
                base_widths[key] += 1
                extra -= 1
                expanded = True
        if not expanded:
            break

    return base_widths
