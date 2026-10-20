"""Local paths for resumable Hugging Face downloads.

Mirrors the layout used by ``huggingface_hub`` for ``local_dir`` downloads
(``<dir>/.cache/huggingface/download/<file>.{metadata,lock}`` and an
etag-scoped ``.incomplete`` partial) so partial files written by earlier SDK
based downloads keep resuming, without depending on the SDK's private module.
"""

from __future__ import annotations

import base64
import hashlib
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

_WINDOWS_PATH_LIMIT = 255
_LONG_PREFIX = "\\\\?\\"


def _long_path(path: Path) -> Path:
    """Use the Windows extended-length prefix when a path exceeds MAX_PATH."""
    resolved = os.path.abspath(path)
    if os.name == "nt" and not resolved.startswith(_LONG_PREFIX):
        return Path(_LONG_PREFIX + resolved)
    return path


def _needs_long_path(path: Path) -> bool:
    return os.name == "nt" and len(os.path.abspath(path)) > _WINDOWS_PATH_LIMIT


@dataclass(frozen=True)
class HfDownloadPaths:
    file_path: Path
    lock_path: Path
    metadata_path: Path

    def incomplete_path(self, etag: str) -> Path:
        """Return the partial-download path for one exact artifact identity."""
        digest = hashlib.sha1(self.metadata_path.name.encode()).digest()
        name = f"{base64.urlsafe_b64encode(digest).decode()}.{etag}.incomplete"
        path = self.metadata_path.parent / name
        resolved = path.resolve()
        return _long_path(resolved) if _needs_long_path(resolved) else path


def hf_download_paths(local_dir: Path, filename: str) -> HfDownloadPaths:
    """Compute and create the directories for *filename* downloaded into *local_dir*."""
    relative = os.path.join(*filename.split("/"))
    if os.name == "nt" and (relative.startswith("..\\") or "\\..\\" in relative):
        raise ValueError(f"Invalid filename for Windows: {relative}")

    cache_dir = Path(local_dir) / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    gitignore = cache_dir / ".gitignore"
    if not gitignore.exists():
        with suppress(OSError):
            gitignore.write_text("*")

    file_path = Path(local_dir) / relative
    metadata_path = cache_dir / "download" / f"{relative}.metadata"
    lock_path = metadata_path.with_suffix(".lock")
    # Like the SDK, the lock path decides whether all three need the long-path prefix.
    if not str(local_dir).startswith(_LONG_PREFIX) and _needs_long_path(lock_path):
        file_path, lock_path, metadata_path = (
            _long_path(path) for path in (file_path, lock_path, metadata_path)
        )
    paths = HfDownloadPaths(file_path=file_path, lock_path=lock_path, metadata_path=metadata_path)
    paths.file_path.parent.mkdir(parents=True, exist_ok=True)
    paths.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    return paths
