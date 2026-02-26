import requests
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from utils import (
    calculate_fit,
    determine_use_case,
    estimate_model_size_gb,
    extract_params,
    format_likes,
)


def search_hf_models(
    query,
    specs,
    repo_files_cache,
    model_info_cache,
    limit=15,
    detailed_limit=8,
):
    results = []
    errors = []
    found_keys = set()

    try:
        api = HfApi()
        hf_models = api.list_models(
            search=query, sort="downloads", limit=limit, filter="gguf"
        )
    except (HfHubHTTPError, requests.RequestException, ValueError, OSError) as exc:
        return results, [f"Hugging Face search failed: {exc}"]

    for index, model in enumerate(hf_models):
        try:
            provider = model.modelId.split("/")[0][:15]
            name = model.modelId.split("/")[-1]
            unique_key = f"Hugging Face:{model.modelId.lower()}"
            if unique_key in found_keys:
                continue
            found_keys.add(unique_key)

            params = extract_params(name)
            use_case = determine_use_case(name)
            likes = getattr(model, "likes", 0)
            score_str = (
                f"[red]❤️ {format_likes(likes)}[/red]"
                if likes > 0
                else "[grey50]-[/grey50]"
            )

            quant = "GGUF"
            size = estimate_model_size_gb(name)
            target = None
            detailed_lookup = index < detailed_limit

            if detailed_lookup:
                files = repo_files_cache.get(model.modelId)
                if files is None:
                    files = api.list_repo_files(repo_id=model.modelId)
                    repo_files_cache[model.modelId] = files

                target = next((file for file in files if "Q4_K_M.gguf" in file), None)
                if not target:
                    target = next((file for file in files if "Q4_0.gguf" in file), None)
                if not target:
                    target = next(
                        (file for file in files if "Q5_K_M.gguf" in file), None
                    )
                if not target:
                    target = next(
                        (file for file in files if file.endswith(".gguf")), None
                    )

                if target:
                    info = model_info_cache.get(model.modelId)
                    if info is None:
                        info = api.model_info(model.modelId, files_metadata=True)
                        model_info_cache[model.modelId] = info

                    siblings = info.siblings or []
                    metadata = next(
                        (item for item in siblings if item.rfilename == target), None
                    )
                    if metadata and metadata.size:
                        size = metadata.size / (1024**3)

                    quant = (
                        target.split(".")[-2] if len(target.split(".")) > 2 else "GGUF"
                    )
                    if "gguf" in quant.lower():
                        quant = "GGUF"

            fit_str, mode_str, _ = calculate_fit(size, specs)
            results.append(
                {
                    "inst": "[grey37]-[/grey37]",
                    "source": "Hugging Face",
                    "provider": provider,
                    "id": model.modelId,
                    "name": name,
                    "params": params,
                    "use_case": use_case,
                    "score": score_str,
                    "quant": quant,
                    "mode": mode_str,
                    "fit": fit_str,
                    "size": f"{size:.1f} GB",
                }
            )
        except (
            AttributeError,
            TypeError,
            ValueError,
            HfHubHTTPError,
            requests.RequestException,
            OSError,
        ) as exc:
            errors.append(f"Hugging Face model parse failed: {exc}")

    return results, errors
