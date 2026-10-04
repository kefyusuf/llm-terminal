"""Read-only HF disk plans and bounded checks of managed artifact directories."""

import hashlib
import os
import shlex
import shutil
import sys
from pathlib import Path

from core.artifacts import bound_artifact_metadata
from downloads.download_manager import build_download_command, validate_hf_target_file

DISK_SAFETY_MARGIN = 64 * 1024 * 1024
MAX_MANAGED_ENTRIES = 4096


def hf_download_command(model, *, powershell=None):
    """Display one exact SDK CLI download, quoting for the host shell."""
    command = build_download_command(model)
    args = [sys.executable, "-m", "huggingface_hub.commands.huggingface_cli", "download"]
    if len(command) > 3:
        args.extend(["--revision", command[3]])
    plan = model.get("download_plan") or {}
    if plan.get("model_directory"):
        args.extend(["--local-dir", plan["model_directory"]])
    args.extend(["--", command[1], command[2]])
    if powershell is True or (powershell is None and os.name == "nt"):
        return "& " + " ".join("'" + arg.replace("'", "''") + "'" for arg in args)
    return shlex.join(args)


def _raise_walk_error(error):
    raise error


def check_managed_directory(directory):
    """Fail closed on links/hardlinks and oversized existing directory trees."""
    directory = Path(directory)
    if directory.resolve() != directory:
        raise ValueError("managed directory contains a link")
    if directory.exists() and not directory.is_dir():
        raise ValueError("managed destination is not a directory")
    if not directory.exists():
        return
    count = 0
    for parent, dirs, files in os.walk(directory, followlinks=False, onerror=_raise_walk_error):
        for name in (*dirs, *files):
            count += 1
            if count > MAX_MANAGED_ENTRIES:
                raise ValueError("managed directory exceeds the preflight entry limit")
            path = Path(parent) / name
            if path.is_symlink() or path.resolve() != path:
                raise ValueError("managed directory contains a link")
            if path.is_file() and path.stat().st_nlink > 1:
                raise ValueError("managed directory contains a hard link")


def plan_hf_download(model, models_dir, *, reserved_bytes=0, allow_unknown_size=False):
    command = build_download_command(model)
    if command[0] != "hf_api_download":
        raise ValueError("download plans currently support Hugging Face files")
    if type(reserved_bytes) is not int or reserved_bytes < 0:
        raise ValueError("disk reservations must be non-negative bytes")
    metadata = bound_artifact_metadata(model) or {}
    root = Path(models_dir).resolve()
    repo_key = hashlib.sha256(command[1].encode("utf-8")).hexdigest()
    revision = command[3] if len(command) > 3 else None
    directory = root / "huggingface" / repo_key / (revision or "unresolved")
    destination = directory / command[2]
    size = metadata.get("size_bytes")
    warnings = []
    if not revision:
        warnings.append("revision_unknown")
    if not metadata.get("sha256"):
        warnings.append("checksum_unknown")
    if (metadata.get("license") or {}).get("status", "unknown") == "unknown":
        warnings.append("license_unknown")
    plan = {
        "schema_version": 1,
        "provider": "huggingface",
        "repository": command[1],
        "filename": command[2],
        "resolved_revision": revision,
        "model_root": str(root),
        "model_directory": str(directory),
        "destination": str(destination),
        "size_bytes": size,
        "sha256": metadata.get("sha256"),
        "free_bytes": None,
        "reserved_bytes": reserved_bytes,
        "safety_margin_bytes": DISK_SAFETY_MARGIN,
        "required_bytes": size + reserved_bytes + DISK_SAFETY_MARGIN if size is not None else None,
        "allowed": False,
        "status": "disk_unavailable",
        "warnings": warnings,
    }
    try:
        if root.exists() and not root.is_dir():
            raise ValueError("model root is not a directory")
        check_managed_directory(directory)
        if not destination.resolve().is_relative_to(directory) or destination.is_dir():
            raise ValueError("selected artifact escapes managed directory")
    except (OSError, RuntimeError, ValueError):
        plan["status"] = "unsafe_destination"
        return plan
    try:
        ancestor = root
        while not ancestor.exists():
            ancestor = ancestor.parent
        free = shutil.disk_usage(ancestor).free
    except OSError:
        return plan
    plan["free_bytes"] = free
    if free < (plan["required_bytes"] or reserved_bytes + DISK_SAFETY_MARGIN):
        plan["status"] = "insufficient_disk"
    elif size is None:
        plan["status"] = "unknown_size"
        plan["allowed"] = bool(allow_unknown_size)
        warnings.append("size_unknown")
    else:
        plan["status"] = "ready"
        plan["allowed"] = True
    return plan


def verify_hf_artifact(directory, filename, size=None, sha256=None):
    """Require a real contained file and match all supplied byte/digest facts."""
    validate_hf_target_file(filename)
    root = Path(directory).resolve()
    check_managed_directory(root)
    path = root / filename
    try:
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError("downloaded artifact is missing or outside the destination")
        if size is not None and path.stat().st_size != size:
            raise ValueError("downloaded artifact byte count does not match metadata")
        if sha256:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest().lower() != sha256.lower():
                raise ValueError("downloaded artifact SHA-256 does not match metadata")
    except OSError as exc:
        raise ValueError("downloaded artifact is unavailable") from exc
