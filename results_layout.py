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
    "fit": 8,
    "download": 4,
}

EXPANDABLE_COLUMNS = ("name", "publisher", "use_case", "download")
COMPACT_EXPANDABLE_COLUMNS = ("score", "fit", "params", "quant", "use_case", "name")

COMPACT_MAX_COLUMN_WIDTHS = {
    "name": 32,
    "use_case": 14,
    "score": 10,
    "params": 8,
    "quant": 8,
    "fit": 10,
}


def column_keys_for_width(available_width: int, *, compact: bool = False) -> list[str]:
    """Return visible column keys based on available table width."""
    if compact:
        if available_width >= 130:
            return ["inst", "name", "params", "score", "quant", "fit", "use_case"]
        if available_width >= 110:
            return ["inst", "name", "params", "score", "quant", "fit"]
        if available_width >= 95:
            return ["inst", "name", "score", "quant", "fit"]
        return ["inst", "name", "score", "fit"]

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


def compute_column_widths(
    next_keys: list[str], available_width: int, *, compact: bool = False
) -> dict[str, int]:
    """Compute responsive widths for visible result-table columns."""
    separator_budget = len(next_keys) + 2
    target_content_width = max(40, available_width - separator_budget)
    base_widths = {key: MIN_COLUMN_WIDTHS[key] for key in next_keys}
    min_total = sum(base_widths.values())
    extra = max(0, target_content_width - min_total)

    expand_order = COMPACT_EXPANDABLE_COLUMNS if compact else EXPANDABLE_COLUMNS

    while extra > 0:
        expanded = False
        for key in expand_order:
            if key in base_widths and extra > 0:
                if compact:
                    max_width = COMPACT_MAX_COLUMN_WIDTHS.get(key)
                    if max_width is not None and base_widths[key] >= max_width:
                        continue
                base_widths[key] += 1
                extra -= 1
                expanded = True
        if not expanded:
            break

    return base_widths
