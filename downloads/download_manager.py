import re
from pathlib import Path


def validate_hf_target_file(filename):
    """Require an unambiguous relative Hub filename on every supported OS."""
    if not isinstance(filename, str) or not filename:
        raise ValueError("missing Hugging Face target file")
    for part in filename.split("/"):
        stem = part.split(".", maxsplit=1)[0].rstrip(" ").upper()
        if (
            not part
            or part in {".", ".."}
            or part.endswith((".", " "))
            or any(ord(char) < 32 or char in '\\:<>"|?*' for char in part)
            or stem in {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
            or re.fullmatch(r"(?:COM|LPT)[1-9¹²³]", stem)
        ):
            raise ValueError(
                "invalid Hugging Face target file: expected a portable relative filename"
            )


def prepare_hf_destination(models_dir, filename):
    """Check the resolved artifact path before creating the configured root."""
    validate_hf_target_file(filename)
    root = Path(models_dir).resolve()
    destination = (root / filename).resolve()
    if not destination.is_relative_to(root) or destination == root:
        raise ValueError("Hugging Face target file resolves outside the model directory")
    if destination.is_dir():
        raise ValueError("Hugging Face target file resolves to a directory")
    root.mkdir(parents=True, exist_ok=True)
    return root


def normalize_target_id(value):
    """Normalise *value* into a lowercase ``"source:identifier"`` string.

    Fills in ``"unknown"`` for any missing component.
    """
    raw = str(value or "unknown:unknown").strip().lower()
    if ":" not in raw:
        return f"unknown:{raw}"
    source, identifier = raw.split(":", maxsplit=1)
    source = source.strip() or "unknown"
    identifier = identifier.strip() or "unknown"
    return f"{source}:{identifier}"


def build_download_command(model):
    """Build the subprocess command list needed to download *model*.

    Raises:
        ValueError: If the model is missing required provider download metadata.
    """
    source = model.get("source")
    if source == "Hugging Face":
        repo_id = model.get("id") or model.get("name")
        if not repo_id:
            raise ValueError("missing Hugging Face repository id")
        target_file = model.get("target_file")
        if not target_file:
            raise ValueError("missing Hugging Face target file")
        validate_hf_target_file(target_file)
        command = ["hf_api_download", repo_id, target_file]
        revision = model.get("resolved_revision")
        if revision is not None:
            if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
                raise ValueError("Hugging Face resolved revision must be a full commit SHA")
            command.append(revision)
        return command

    if source == "Ollama":
        model_name = model.get("name")
        if not model_name:
            raise ValueError("missing Ollama model name")
        return ["ollama", "pull", model_name]

    raise ValueError(f"unsupported source: {source}")


def download_target_id(model):
    """Return the normalised ``"source:identifier"`` key for *model*."""
    source = str(model.get("source", "unknown")).strip().lower()
    identifier = str(model.get("id") or model.get("name") or "unknown").strip().lower()
    return normalize_target_id(f"{source}:{identifier}")
