"""Versioned, JSON-safe upstream artifact declarations, never usage permission."""

import json
import re
import time
from urllib.parse import quote, urlparse


def _text(value):
    return value if isinstance(value, str) and 0 < len(value) <= 2048 else None


def hf_artifact_metadata(repository, filename, info):
    source = "https://huggingface.co/" + quote(repository, safe="/")
    revision = getattr(info, "sha", None)
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        revision = None
    files = getattr(info, "siblings", None) or []
    names = {getattr(item, "rfilename", "") for item in files}
    selected = next((item for item in files if getattr(item, "rfilename", None) == filename), None)
    size = getattr(selected, "size", None)
    if type(size) is not int or size < 0:
        size = None
    lfs = getattr(selected, "lfs", None)
    digest = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        digest = None
    card = getattr(info, "card_data", None) or {}
    if hasattr(card, "to_dict"):
        card = card.to_dict()
    if not isinstance(card, dict):
        card = {}
    license_id = _text(card.get("license"))
    license_name = _text(card.get("license_name"))
    link = _text(card.get("license_link"))
    license_url = None
    if link:
        parsed = urlparse(link)
        if (
            parsed.scheme == "https"
            and parsed.hostname
            and not parsed.username
            and not parsed.password
        ):
            license_url = link
        elif not parsed.scheme and link in names:
            license_url = f"{source}/blob/{revision or 'main'}/{quote(link, safe='/')}"
    else:
        license_file = next(
            (name for name in ("LICENSE", "LICENSE.md", "LICENSE.txt") if name in names), None
        )
        if license_file:
            license_url = f"{source}/blob/{revision or 'main'}/{license_file}"
    return {
        "schema_version": 1,
        "provider": "huggingface",
        "repository": repository,
        "filename": filename,
        "resolved_revision": revision,
        "sha256": digest,
        "revision_status": "pinned" if revision else "unknown",
        "size_bytes": size,
        "source_url": source,
        "model_card_url": f"{source}/blob/{revision or 'main'}/README.md"
        if "README.md" in names
        else None,
        "license": {
            "id": license_id,
            "name": license_name,
            "url": license_url,
            "status": "declared" if license_id or license_name or license_url else "unknown",
        },
        "metadata_observed_at": time.time() if info is not None else None,
        "metadata_status": "unavailable"
        if info is None
        else "available"
        if size is not None and revision
        else "partial",
    }


def bound_artifact_metadata(model):
    """Reject a declaration that does not describe the queued selection."""
    metadata = model.get("artifact_metadata")
    if metadata is None:
        return None
    expected = (
        model.get("id") or model.get("name"),
        model.get("target_file"),
        model.get("resolved_revision"),
    )
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version") != 1
        or metadata.get("provider") != "huggingface"
        or tuple(metadata.get(key) for key in ("repository", "filename", "resolved_revision"))
        != expected
    ):
        raise ValueError("artifact metadata does not match the selected Hugging Face file")
    size = metadata.get("size_bytes")
    if size is not None and (type(size) is not int or size < 0):
        raise ValueError("artifact metadata size must be a non-negative byte count")
    digest = metadata.get("sha256")
    if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest)):
        raise ValueError("artifact metadata SHA-256 must be a full digest")
    if metadata.get("license") is not None and not isinstance(metadata["license"], dict):
        raise ValueError("artifact metadata license must be an object")
    try:
        return json.loads(json.dumps(metadata, allow_nan=False))
    except (ValueError, TypeError) as exc:
        raise ValueError("artifact metadata must contain JSON-safe values") from exc
