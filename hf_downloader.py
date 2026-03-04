#!/usr/bin/env python
"""Download a model from Hugging Face with progress tracking"""

import sys
import os
from huggingface_hub import snapshot_download, HfApi

if len(sys.argv) < 2:
    print("Usage: python hf_downloader.py <repo_id>")
    sys.exit(1)

repo_id = sys.argv[1]

# Get HF token from environment or config
hf_token = os.environ.get("AIMODEL_HF_TOKEN") or os.environ.get("HF_TOKEN")

try:
    print(f"Downloading {repo_id}...")
    if hf_token:
        print("Using authentication token")

    # Get file list first
    api = HfApi(token=hf_token)
    files = api.list_repo_files(repo_id)
    gguf_files = [f for f in files if f.endswith(".gguf")]

    if not gguf_files:
        print("No GGUF files found in repository")
        sys.exit(1)

    print(f"Found {len(gguf_files)} GGUF file(s)")

    # Download with token
    snapshot_download(
        repo_id=repo_id,
        allow_patterns="*.gguf",
        local_dir="models",
        local_dir_use_symlinks=False,
        force_download=False,
        token=hf_token,
    )
    print(f"Downloaded {repo_id}")
except KeyboardInterrupt:
    print("\nDownload interrupted")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

repo_id = sys.argv[1]

try:
    print(f"Downloading {repo_id}...")

    # Get file list first
    api = HfApi()
    files = api.list_repo_files(repo_id)
    gguf_files = [f for f in files if f.endswith(".gguf")]

    if not gguf_files:
        print("No GGUF files found in repository")
        sys.exit(1)

    print(f"Found {len(gguf_files)} GGUF file(s)")

    # Download
    snapshot_download(
        repo_id=repo_id,
        allow_patterns="*.gguf",
        local_dir="models",
        local_dir_use_symlinks=False,
        force_download=False,
    )
    print(f"Downloaded {repo_id}")
except KeyboardInterrupt:
    print("\nDownload interrupted")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

repo_id = sys.argv[1]

try:
    print(f"Downloading {repo_id}...")

    # Get file list first
    api = HfApi()
    files = api.list_repo_files(repo_id)
    gguf_files = [f for f in files if f.endswith(".gguf")]

    if not gguf_files:
        print("❌ No GGUF files found in repository")
        sys.exit(1)

    print(f"Found {len(gguf_files)} GGUF file(s)")

    # Download with progress
    snapshot_download(
        repo_id=repo_id,
        allow_patterns="*.gguf",
        local_dir="models",
        local_dir_use_symlinks=False,
        show_progress=True,
    )
    print(f"✅ Downloaded {repo_id}")
except KeyboardInterrupt:
    print("\n⚠️  Download interrupted")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

repo_id = sys.argv[1]

try:
    print(f"Downloading {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        allow_patterns="*.gguf",
        local_dir="models",
        local_dir_use_symlinks=False,
    )
    print(f"✅ Downloaded {repo_id}")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
