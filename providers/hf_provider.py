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


def _repo_id_from_model(model):
    return getattr(model, "modelId", None) or getattr(model, "id", None)


def _select_preferred_gguf(siblings):
    filenames = [
        item.rfilename for item in siblings if getattr(item, "rfilename", None)
    ]
    priorities = ["Q4_K_M.gguf", "Q4_0.gguf", "Q5_K_M.gguf"]

    for suffix in priorities:
        target = next((name for name in filenames if suffix in name), None)
        if target:
            return target

    return next((name for name in filenames if name.endswith(".gguf")), None)


def search_hf_models(
    query,
    specs,
    model_info_cache,
    limit=15,
):
    results = []
    errors = []
    found_keys = set()
    _ = model_info_cache

    try:
        api = HfApi()
        hf_models = api.list_models(
            search=query,
            sort="downloads",
            limit=limit,
            filter="gguf",
            expand=["likes", "siblings"],
        )
    except (HfHubHTTPError, requests.RequestException, ValueError, OSError) as exc:
        return results, [f"Hugging Face search failed: {exc}"]

    for model in hf_models:
        try:
            repo_id = _repo_id_from_model(model)
            if not repo_id:
                raise ValueError("missing model repository id")

            provider = repo_id.split("/")[0][:15]
            name = repo_id.split("/")[-1]
            unique_key = f"Hugging Face:{repo_id.lower()}"
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
            siblings = getattr(model, "siblings", None) or []
            target = _select_preferred_gguf(siblings)

            if target:
                quant = target.split(".")[-2] if len(target.split(".")) > 2 else "GGUF"
                if "gguf" in quant.lower():
                    quant = "GGUF"

            fit_str, mode_str, _ = calculate_fit(size, specs)
            results.append(
                {
                    "inst": "[grey37]-[/grey37]",
                    "source": "Hugging Face",
                    "provider": provider,
                    "id": repo_id,
                    "name": name,
                    "params": params,
                    "use_case": use_case,
                    "score": score_str,
                    "quant": quant,
                    "target_file": target,
                    "size_source": "estimated",
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


def enrich_hf_model_details(model, specs, model_info_cache):
    repo_id = model.get("id")
    if not repo_id:
        return model

    try:
        api = HfApi()
        info = model_info_cache.get(repo_id)
        if info is None:
            info = api.model_info(repo_id, files_metadata=True)
            model_info_cache[repo_id] = info

        siblings = info.siblings or []
        target = model.get("target_file") or _select_preferred_gguf(siblings)
        if not target:
            return model

        metadata = next((item for item in siblings if item.rfilename == target), None)
        if metadata and metadata.size:
            size = metadata.size / (1024**3)
            fit_str, mode_str, _ = calculate_fit(size, specs)
            model["size"] = f"{size:.1f} GB"
            model["fit"] = fit_str
            model["mode"] = mode_str
            model["size_source"] = "exact"

        quant = target.split(".")[-2] if len(target.split(".")) > 2 else "GGUF"
        if "gguf" in quant.lower():
            quant = "GGUF"
        model["quant"] = quant
        model["target_file"] = target
    except (HfHubHTTPError, requests.RequestException, OSError, ValueError, TypeError):
        return model

    return model
