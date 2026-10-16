"""Retain provider-reported transfer information without estimating metrics."""

from __future__ import annotations

import re

_TERMINAL_ESCAPE = re.compile(
    r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]"
)
_OLLAMA_PHASE = re.compile(
    r"^(?:pulling\s+|verifying sha256 digest\b|writing manifest\b|removing unused layers\b|success\b)"
)


def ollama_transfer_detail(line: str) -> str | None:
    """Return bounded, plain CLI transfer/phase text, or None for other output.

    Bytes, speed and remaining time belong to the current provider layer.
    Retain only what the CLI supplies; never infer model-wide totals or ETA.
    """
    clean = _TERMINAL_ESCAPE.sub("", line)
    clean = " ".join("".join(char for char in clean if char.isprintable() or char.isspace()).split())
    return clean[:512] if _OLLAMA_PHASE.match(clean) else None
