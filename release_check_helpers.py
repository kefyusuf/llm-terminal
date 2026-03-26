"""Helpers used by release preflight checks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def unpack_search_output(search_output: Sequence[Any]) -> tuple[list[Any], list[str]]:
    """Normalize provider search output to ``(results, errors)``.

    Supported shapes:
    - ``(results, errors)``
    - ``(results, errors, has_more_pages)``
    """
    if len(search_output) == 2:
        results, errors = search_output
        return list(results), list(errors)
    if len(search_output) == 3:
        results, errors, _has_more_pages = search_output
        return list(results), list(errors)
    raise ValueError("unexpected provider search output shape")
