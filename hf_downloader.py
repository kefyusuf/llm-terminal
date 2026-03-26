#!/usr/bin/env python
"""Download GGUF files from a Hugging Face repository."""

from __future__ import annotations

import os
import sys

from huggingface_hub import HfApi, snapshot_download


def _resolve_token() -> str | None:
    return os.environ.get("AIMODEL_HF_TOKEN") or os.environ.get("HF_TOKEN")


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args:
        print("Usage: python hf_downloader.py <repo_id>")
        return 1

    repo_id = args[0]
    token = _resolve_token()

    try:
        print(f"Downloading {repo_id}...")
        if token:
            print("Using authentication token")

        api = HfApi(token=token)
        files = api.list_repo_files(repo_id)
        gguf_files = [name for name in files if name.endswith(".gguf")]

        if not gguf_files:
            print("No GGUF files found in repository")
            return 1

        print(f"Found {len(gguf_files)} GGUF file(s)")
        snapshot_download(
            repo_id=repo_id,
            allow_patterns="*.gguf",
            local_dir="models",
            local_dir_use_symlinks=False,
            force_download=False,
            token=token,
        )
        print(f"Downloaded {repo_id}")
        return 0
    except KeyboardInterrupt:
        print("\nDownload interrupted")
        return 1
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
