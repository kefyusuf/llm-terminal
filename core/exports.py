"""Versioned data exports; no discovery, rescoring or terminal rendering."""

import json

from core.artifacts import bound_artifact_metadata

SCORE_DIMENSIONS = ("quality", "speed", "fit", "context", "composite")


def export_model(model):
    """Select public model facts while preserving explicit unknowns."""
    fields = ("id", "name", "source", "publisher", "params", "quant", "size",
              "target_file", "resolved_revision", "estimated_tok_s", "score_provenance")
    result = {field: model.get(field) for field in fields}
    prior_scores = model.get("scores") or {}
    result["scores"] = {dimension: model.get("score_" + dimension,
                                           prior_scores.get(dimension, 0))
                        for dimension in SCORE_DIMENSIONS}
    try:
        metadata = bound_artifact_metadata(model)
        status = "available" if metadata else (
            "invalid" if model.get("artifact_metadata_status") == "invalid" else "unavailable"
        )
    except ValueError:
        metadata, status = None, "invalid"
    result["artifact_metadata"] = metadata
    result["artifact_metadata_status"] = status
    return result


def read_search_export(path):
    """Accept a bounded schema-1 saved search, without following its contents."""
    with open(path, "rb") as handle:
        raw = handle.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError("saved search exceeds the 2 MiB input limit")
    data = json.loads(raw)
    if (not isinstance(data, dict) or data.get("schema_version") != 1
        or data.get("kind") != "search" or not isinstance(data.get("models"), list)
        or any(not isinstance(model, dict) for model in data["models"])
        or any(not isinstance(model.get("scores", {}), dict) for model in data["models"])
        or not isinstance(data.get("errors", []), list)
        or any(not isinstance(error, str) for error in data.get("errors", []))):
        raise ValueError("unsupported saved search schema")
    return data


def comparison_export(data, identities):
    if not 2 <= len(identities) <= 8 or len(set(identities)) != len(identities):
        raise ValueError("select between two and eight distinct model IDs")
    selected = []
    for identity in identities:
        matches = [model for model in data["models"] if model.get("id") == identity]
        if len(matches) != 1:
            raise ValueError("comparison identity is missing or ambiguous")
        selected.append(export_model(matches[0]))
    return {"schema_version": 1, "kind": "comparison", "models": selected,
            "errors": data.get("errors", []), "input_query": data.get("query")}
