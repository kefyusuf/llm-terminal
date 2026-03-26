"""Helpers for filtering and ordering model results in the UI."""

from __future__ import annotations

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
        filtered_results.append(result)

    if hidden_gems_only:
        filtered_results.sort(
            key=lambda item: (
                item.get("is_hidden_gem", False),
                item.get("gem_score", 0.0),
                item.get("downloads", 0),
            ),
            reverse=True,
        )

    return filtered_results
