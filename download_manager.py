def normalize_target_id(value):
    raw = str(value or "unknown:unknown").strip().lower()
    if ":" not in raw:
        return f"unknown:{raw}"
    source, identifier = raw.split(":", maxsplit=1)
    source = source.strip() or "unknown"
    identifier = identifier.strip() or "unknown"
    return f"{source}:{identifier}"


def build_download_command(model):
    source = model.get("source")
    if source == "Hugging Face":
        repo_id = model.get("id") or model.get("name")
        if not repo_id:
            raise ValueError("missing Hugging Face repository id")
        return ["hf_api_download", repo_id]

    if source == "Ollama":
        model_name = model.get("name")
        if not model_name:
            raise ValueError("missing Ollama model name")
        return ["ollama", "pull", model_name]

    raise ValueError(f"unsupported source: {source}")


def download_target_id(model):
    source = str(model.get("source", "unknown")).strip().lower()
    identifier = str(model.get("id") or model.get("name") or "unknown").strip().lower()
    return normalize_target_id(f"{source}:{identifier}")
