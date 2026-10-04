"""Build-once candidate identity and tamper detection."""

import io
import tarfile
import zipfile

import pytest

from scripts import package_candidate


def distributions(root, version="1.0.1"):
    metadata = f"Name: ai-model-explorer\nVersion: {version}\n\n".encode()
    with zipfile.ZipFile(root / "ai_model_explorer-1.0.1-py3-none-any.whl", "w") as wheel:
        wheel.writestr("ai_model_explorer-1.0.1.dist-info/METADATA", metadata)
    with tarfile.open(root / "ai_model_explorer-1.0.1.tar.gz", "w:gz") as source:
        member = tarfile.TarInfo("ai_model_explorer-1.0.1/PKG-INFO")
        member.size = len(metadata)
        source.addfile(member, io.BytesIO(metadata))


def test_candidate_binds_both_distribution_hashes_to_source(tmp_path):
    distributions(tmp_path)
    manifest = package_candidate.create_manifest(tmp_path, "a" * 40)
    assert manifest["schema_version"] == 1
    assert manifest["source_sha"] == "a" * 40
    assert manifest["version"] == "1.0.1"
    assert {item["kind"] for item in manifest["artifacts"]} == {"wheel", "sdist"}
    package_candidate.verify_manifest(tmp_path, manifest, "a" * 40)


def test_modified_distribution_cannot_pass_candidate_verification(tmp_path):
    distributions(tmp_path)
    manifest = package_candidate.create_manifest(tmp_path, "a" * 40)
    artifact = tmp_path / manifest["artifacts"][0]["filename"]
    with artifact.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="match"):
        package_candidate.verify_manifest(tmp_path, manifest, "a" * 40)


def test_candidate_cannot_be_reused_for_another_source(tmp_path):
    distributions(tmp_path)
    manifest = package_candidate.create_manifest(tmp_path, "a" * 40)
    with pytest.raises(ValueError, match="source"):
        package_candidate.verify_manifest(tmp_path, manifest, "b" * 40)


@pytest.mark.parametrize("revision", ["main", "", "g" * 40])
def test_candidate_requires_full_source_sha(tmp_path, revision):
    distributions(tmp_path)
    with pytest.raises(ValueError, match="source"):
        package_candidate.create_manifest(tmp_path, revision)


def test_candidate_requires_one_wheel_and_one_sdist(tmp_path):
    distributions(tmp_path)
    (tmp_path / "ai_model_explorer-1.0.1.tar.gz").unlink()
    with pytest.raises(ValueError, match="wheel and sdist"):
        package_candidate.create_manifest(tmp_path, "a" * 40)
