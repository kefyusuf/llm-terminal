import sys


def build_download_command(model):
    source = model.get("source")
    if source == "Hugging Face":
        repo_id = model.get("id") or model.get("name")
        if not repo_id:
            raise ValueError("missing Hugging Face repository id")
        return [
            sys.executable,
            "-m",
            "huggingface_hub.commands.huggingface_cli",
            "download",
            repo_id,
            "--include",
            "*.gguf",
        ]

    if source == "Ollama":
        model_name = model.get("name")
        if not model_name:
            raise ValueError("missing Ollama model name")
        return ["ollama", "pull", model_name]

    raise ValueError(f"unsupported source: {source}")


def download_target_id(model):
    source = model.get("source", "unknown")
    identifier = model.get("id") or model.get("name") or "unknown"
    return f"{source}:{identifier}"
