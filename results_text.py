"""Text and row-shape helpers for results table rendering."""

from __future__ import annotations

import re


def truncate_cell(value: object, max_len: int) -> str:
    """Truncate plain text to a maximum length using ellipsis."""
    text = str(value or "-")
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3] + "..."


def truncate_plain_cell(value: object, max_len: int) -> str:
    """Strip simple Rich-style markup and truncate text."""
    text = re.sub(r"\[[^\]]+\]", "", str(value or "-")).strip()
    if not text:
        text = "-"
    return truncate_cell(text, max_len)


def format_header_label(label: object, width: int) -> str:
    """Format a column header label to fixed width."""
    if width <= 0:
        return str(label)
    text = truncate_plain_cell(label, width)
    return text.ljust(width)


def align_plain_cell(value: object, width: int, align: str = "left") -> str:
    """Align plain text to fixed-width cell content."""
    text = str(value or "-")
    if width <= 0:
        return text
    if len(text) > width:
        text = truncate_cell(text, width)
    if align == "center":
        return text.center(width)
    if align == "right":
        return text.rjust(width)
    return text.ljust(width)


def blank_result_row() -> dict[str, str]:
    """Return an empty results-table row payload."""
    return {
        "inst": "-",
        "source": "-",
        "publisher": "-",
        "name": "-",
        "params": "-",
        "use_case": "-",
        "score": "-",
        "quant": "-",
        "mode": "-",
        "fit": "-",
        "download": "-",
    }
