"""Cell-level formatting helpers for the results table."""

from __future__ import annotations

import re
from collections.abc import Callable

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


def _score_color(score: int) -> str:
    """Return Rich color tag for a 0-100 score value."""
    if score >= 75:
        return "#4fe08a"  # green
    if score >= 50:
        return "#7edfff"  # blue
    if score >= 30:
        return "#f2c46d"  # yellow
    return "#ff7f8f"  # red


def score_bar_cell_markup(
    result: dict,
    *,
    width: int,
    truncate_plain: _TruncatePlain,
    align_plain: _AlignPlain,
) -> str:
    """Render compact 4-dimension score bar: Q72 S45 F88 C60.

    Each dimension is color-coded by its value. Returns a Rich-markup string.
    """
    q = result.get("score_quality", 0)
    s = result.get("score_speed", 0)
    f = result.get("score_fit", 0)
    c = result.get("score_context", 0)

    parts = [
        f"[{_score_color(q)}]Q{q}[/{_score_color(q)}]",
        f"[{_score_color(s)}]S{s}[/{_score_color(s)}]",
        f"[{_score_color(f)}]F{f}[/{_score_color(f)}]",
        f"[{_score_color(c)}]C{c}[/{_score_color(c)}]",
    ]
    return " ".join(parts)


def composite_score_cell_markup(
    result: dict,
    *,
    width: int,
    truncate_plain: _TruncatePlain,
    align_plain: _AlignPlain,
) -> str:
    """Render the composite score as a single color-coded number."""
    composite = result.get("score_composite", 0)
    color = _score_color(composite)
    text = align_plain(str(composite), width, "center")
    return f"[bold {color}]{text}[/bold {color}]"


def tok_s_cell_markup(
    result: dict,
    *,
    width: int,
    truncate_plain: _TruncatePlain,
    align_plain: _AlignPlain,
) -> str:
    """Render estimated tok/s as a compact string."""
    tok_s = result.get("estimated_tok_s", 0.0)
    if tok_s <= 0:
        plain = "-"
        color = "#6b789e"
    elif tok_s >= 100:
        plain = f"{tok_s:.0f} t/s"
        color = "#4fe08a"
    elif tok_s >= 30:
        plain = f"{tok_s:.0f} t/s"
        color = "#7edfff"
    else:
        plain = f"{tok_s:.0f} t/s"
        color = "#f2c46d"
    aligned = align_plain(plain, width, "right")
    return f"[{color}]{aligned}[/{color}]"


def moe_cell_markup(
    result: dict,
    *,
    width: int,
    truncate_plain: _TruncatePlain,
    align_plain: _AlignPlain,
) -> str:
    """Render MoE indicator if model is Mixture-of-Experts."""
    if not result.get("is_moe", False):
        return align_plain("-", width, "center")
    total = result.get("total_experts", 0)
    active = result.get("active_experts", 0)
    plain = f"MoE {active}/{total}" if total and active else "MoE"
    aligned = align_plain(plain, width, "center")
    return f"[bold #b89df3]{aligned}[/bold #b89df3]"
