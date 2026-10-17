"""Regression coverage for the Hugging Face model download directory."""

from config import Settings, _default_data_dir


def test_hf_models_dir_defaults_to_user_data_directory(monkeypatch):
    """HF downloads must not default to a repository-relative models directory."""
    monkeypatch.delenv("AIMODEL_HF_MODELS_DIR", raising=False)

    assert Settings(_env_file=None).hf_models_dir == _default_data_dir() / "models"


def test_hf_models_dir_accepts_environment_override(monkeypatch, tmp_path):
    """Users may explicitly choose a different persistent model directory."""
    target = tmp_path / "hf-models"
    monkeypatch.setenv("AIMODEL_HF_MODELS_DIR", str(target))

    assert Settings(_env_file=None).hf_models_dir == target
