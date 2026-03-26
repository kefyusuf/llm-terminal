"""Helpers for filtering and ordering model results in the UI."""

from __future__ import annotations

import re
from typing import Any


def result_unique_key(result: dict[str, Any]) -> str:
    """Return stable unique key used for result table rows."""
    return f"{result['source']}:{result.get('id', result.get('name', ''))}"


def filter_results_for_view(
    all_results: list[dict[str, Any]],
    *,
    current_filter: str,
    use_case_filter: str,
    hidden_gems_only: bool,
    sort_mode: str = "score",
    fit_filter: str = "all",
) -> list[dict[str, Any]]:
    """Apply provider/use-case/hidden-gem filters and sort if needed."""
    filtered_results = []
    for result in all_results:
        if result.get("source") != current_filter:
            continue
        if use_case_filter != "all" and result.get("use_case_key") != use_case_filter:
            continue
        if hidden_gems_only and not result.get("is_hidden_gem", False):
            continue
        if fit_filter != "all" and not _fit_matches(result.get("fit", ""), fit_filter):
            continue
        filtered_results.append(result)

    if sort_mode == "name":
        filtered_results.sort(key=lambda item: str(item.get("name", "")).lower())
    elif sort_mode == "downloads":
        filtered_results.sort(
            key=lambda item: (
                int(item.get("downloads", 0) or 0),
                int(item.get("likes", 0) or 0),
            ),
            reverse=True,
        )
    elif hidden_gems_only:
        filtered_results.sort(
            key=lambda item: (
                item.get("is_hidden_gem", False),
                item.get("gem_score", 0.0),
                item.get("downloads", 0),
            ),
            reverse=True,
        )
    else:
        filtered_results.sort(
            key=lambda item: (
                _score_rank(item),
                int(item.get("downloads", 0) or 0),
                str(item.get("name", "")).lower(),
            ),
            reverse=True,
        )

    return filtered_results


def _strip_markup(value: object) -> str:
    return re.sub(r"\[[^\]]+\]", "", str(value or "")).strip().lower()


def _fit_matches(fit_value: object, fit_filter: str) -> bool:
    fit_text = _strip_markup(fit_value)
    if fit_filter == "fit":
        return "perfect" in fit_text or fit_text == "fit"
    if fit_filter == "partial":
        return "partial" in fit_text or "slow" in fit_text
    if fit_filter == "nofit":
        return "no fit" in fit_text or "tight" in fit_text or fit_text == "-"
    return True


def _score_rank(item: dict[str, Any]) -> float:
    likes = float(item.get("likes", 0) or 0)
    downloads = float(item.get("downloads", 0) or 0)
    gem_score = float(item.get("gem_score", 0.0) or 0.0)
    if likes or downloads or gem_score:
        return likes * 10.0 + downloads * 0.01 + gem_score

    score_text = _strip_markup(item.get("score", ""))
    match = re.search(r"(\d+(?:\.\d+)?)", score_text)
    if match:
        return float(match.group(1))
    return 0.0
