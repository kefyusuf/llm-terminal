"""Record and verify the same wheel/sdist bytes for all candidate consumers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename
from packaging.version import Version


def _metadata(path, kind):
    if kind == "wheel":
        with zipfile.ZipFile(path) as archive:
            members = [member for member in archive.infolist()
                       if member.filename.endswith(".dist-info/METADATA")]
            if len(members) != 1 or members[0].file_size > 65536:
                raise ValueError("candidate wheel metadata is missing or ambiguous")
            raw = archive.read(members[0])
        name, version, *_ = parse_wheel_filename(path.name)
    else:
        with tarfile.open(path, "r:gz") as archive:
            members = [member for member in archive.getmembers()
                       if len(Path(member.name).parts) == 2 and member.name.endswith("/PKG-INFO")
                       and member.isfile()]
            if len(members) != 1 or members[0].size > 65536:
                raise ValueError("candidate source metadata is missing or ambiguous")
            handle = archive.extractfile(members[0])
            if handle is None:
                raise ValueError("candidate source metadata is missing")
            with handle:
                raw = handle.read(65537)
        name, version = parse_sdist_filename(path.name)
    metadata = BytesParser().parsebytes(raw)
    if len(metadata.get_all("Name", [])) != 1 or len(metadata.get_all("Version", [])) != 1:
        raise ValueError("candidate identity metadata is missing or ambiguous")
    if (canonicalize_name(metadata["Name"]) != name or Version(metadata["Version"]) != version
        or name != "ai-model-explorer"):
        raise ValueError("candidate filename and metadata do not match")
    return str(name), str(version)


def create_manifest(directory, source_sha):
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha or ""):
        raise ValueError("candidate source must be a full lowercase commit SHA")
    root = Path(directory)
    wheels = list(root.glob("*.whl"))
    sources = list(root.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("candidate requires exactly one wheel and sdist")
    artifacts = []
    identities = []
    for kind, path in (("wheel", wheels[0]), ("sdist", sources[0])):
        identities.append(_metadata(path, kind))
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        artifacts.append({"kind": kind, "filename": path.name,
                          "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    if identities[0] != identities[1]:
        raise ValueError("candidate wheel and sdist identity do not match")
    return {"schema_version": 1, "source_sha": source_sha,
            "distribution": identities[0][0], "version": identities[0][1], "artifacts": artifacts}


def verify_manifest(directory, manifest, source_sha):
    if manifest.get("source_sha") != source_sha:
        raise ValueError("candidate source does not match the consumer checkout")
    if manifest != create_manifest(directory, source_sha):
        raise ValueError("candidate manifest does not match distribution bytes and metadata")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_manifest(args.dist_dir, json.loads(args.manifest.read_text(encoding="utf-8")),
                        args.source_sha)
    else:
        args.manifest.write_text(json.dumps(create_manifest(args.dist_dir, args.source_sha), indent=2)
                                 + "\n", encoding="utf-8")
    print("[candidate] verified" if args.verify else "[candidate] recorded")


if __name__ == "__main__":
    main()
