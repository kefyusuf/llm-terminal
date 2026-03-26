import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from providers.hf_provider import search_hf_models
from providers.ollama_provider import get_installed_ollama_models, search_ollama_models


TEST_SPECS = {
    "has_gpu": False,
    "vram_free": 0.0,
    "ram_free": 64.0,
}


def run_check(name, func):
    try:
        func()
        print(f"[PASS] {name}")
        return True
    except Exception as exc:
        print(f"[FAIL] {name}: {exc}")
        return False


def check_hf_search():
    results, errors = search_hf_models(
        query="qwen",
        specs=TEST_SPECS,
        model_info_cache={},
        limit=5,
    )
    if errors:
        raise RuntimeError("; ".join(errors))
    if not results:
        raise RuntimeError("no Hugging Face results returned")


def check_ollama_search():
    local_models = get_installed_ollama_models()
    search_output = search_ollama_models(
        query="llama",
        specs=TEST_SPECS,
        local_models=local_models,
    )
    if len(search_output) == 3:
        results, errors, _has_more_pages = search_output
    else:
        results, errors = search_output
    if errors:
        raise RuntimeError("; ".join(errors))
    if not results:
        raise RuntimeError("no Ollama search results returned")


def main():
    checks = [
        ("Hugging Face live search", check_hf_search),
        ("Ollama live search", check_ollama_search),
    ]

    outcomes = [run_check(name, func) for name, func in checks]
    if all(outcomes):
        print("Release check completed successfully.")
        return 0
    print("Release check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
