"""Cell-level formatting helpers for the results table."""

from __future__ import annotations

import re
from typing import Callable


_TruncatePlain = Callable[[object, int], str]
_AlignPlain = Callable[[object, int, str], str]


def fit_cell_markup(
    fit_text: object,
    *,
    width: int,
    truncate_plain: _TruncatePlain,
    align_plain: _AlignPlain,
) -> str:
    plain = truncate_plain(fit_text, 20)
    plain = align_plain(plain, width, "left")
    lowered = plain.lower()
    if "perfect" in lowered or lowered == "fit":
        return f"[bold #4fe08a]{plain}[/bold #4fe08a]"
    if "partial" in lowered or "slow" in lowered:
        return f"[bold #f2c46d]{plain}[/bold #f2c46d]"
    if "no fit" in lowered or "tight" in lowered:
        return f"[bold #ff7f8f]{plain}[/bold #ff7f8f]"
    return plain


def mode_cell_markup(
    mode_text: object,
    *,
    width: int,
    truncate_plain: _TruncatePlain,
    align_plain: _AlignPlain,
) -> str:
    plain = truncate_plain(mode_text, 20)
    plain = align_plain(plain, width, "left")
    lowered = plain.lower()
    if "gpu+cpu" in lowered:
        return f"[bold #f2c46d]{plain}[/bold #f2c46d]"
    if "gpu" in lowered:
        return f"[bold #4fe08a]{plain}[/bold #4fe08a]"
    if "cpu" in lowered:
        return f"[bold #f2c46d]{plain}[/bold #f2c46d]"
    return f"[#ff7f8f]{plain}[/#ff7f8f]" if plain != "-" else plain


def installed_cell_markup(
    installed_text: object,
    *,
    width: int,
    truncate_plain: _TruncatePlain,
    align_plain: _AlignPlain,
) -> str:
    plain = truncate_plain(installed_text, 8)
    marker = "●" if ("✔" in plain or "install" in plain.lower()) else "·"
    if marker == "●":
        return f"[bold #4fe08a]{align_plain(marker, width, 'left')}[/bold #4fe08a]"
    return f"[#6b789e]{align_plain(marker, width, 'left')}[/#6b789e]"


def source_cell_markup(
    source_text: object,
    *,
    width: int,
    truncate_plain: _TruncatePlain,
    align_plain: _AlignPlain,
) -> str:
    plain = truncate_plain(source_text, 20)
    plain = align_plain(plain, width, "left")
    lowered = plain.lower()
    if "ollama" in lowered:
        return f"[bold #7edfff]{plain}[/bold #7edfff]"
    if "hugging" in lowered:
        return f"[bold #b6a3ff]{plain}[/bold #b6a3ff]"
    return f"[#9bb1e0]{plain}[/#9bb1e0]"


def score_cell_markup(
    score_text: object,
    *,
    width: int,
    truncate_plain: _TruncatePlain,
    align_plain: _AlignPlain,
) -> str:
    plain = truncate_plain(score_text, 24)
    match = re.search(r"(\d+(?:\.\d+)?)([KkMm]?)", plain)
    if not match:
        return "[#6b789e]-[/#6b789e]"

    value = float(match.group(1))
    suffix = match.group(2).upper()
    if suffix == "K":
        numeric = value * 1000.0
    elif suffix == "M":
        numeric = value * 1000000.0
    else:
        numeric = value

    if numeric >= 1000000:
        color = "#4fe08a"
    elif numeric >= 100000:
        color = "#7edfff"
    else:
        color = "#f2c46d"
    shown = align_plain(f"{value:g}{suffix}", width, "left")
    return f"[bold {color}]{shown}[/bold {color}]"


def use_case_cell_markup(
    use_case_text: object,
    *,
    width: int,
    truncate_plain: _TruncatePlain,
    align_plain: _AlignPlain,
) -> str:
    plain = truncate_plain(use_case_text, 20)
    plain = align_plain(plain, width, "left")
    lowered = plain.lower()
    if "coding" in lowered:
        return f"[#86a7ff]{plain}[/#86a7ff]"
    if "chat" in lowered:
        return f"[#8e97c7]{plain}[/#8e97c7]"
    if "reason" in lowered:
        return f"[#b89df3]{plain}[/#b89df3]"
    if "vision" in lowered:
        return f"[#9fc6ff]{plain}[/#9fc6ff]"
    return f"[#8ea3cf]{plain}[/#8ea3cf]"


def download_cell_markup(
    download_text: object,
    *,
    width: int,
    truncate_plain: _TruncatePlain,
    align_plain: _AlignPlain,
) -> str:
    plain = truncate_plain(download_text, 24)
    aligned = align_plain(plain, width, "left")
    lowered = plain.lower()
    if "downloading" in lowered or "queued" in lowered:
        return f"[bold #f2c46d]{aligned}[/bold #f2c46d]"
    if "done" in lowered or "installed" in lowered:
        return f"[bold #4fe08a]{aligned}[/bold #4fe08a]"
    if "failed" in lowered or "error" in lowered:
        return f"[bold #ff7f8f]{aligned}[/bold #ff7f8f]"
    return f"[#9bb1e0]{aligned}[/#9bb1e0]"
