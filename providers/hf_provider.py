import requests
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from utils import (
    calculate_fit,
    determine_use_case,
    determine_use_case_key,
    estimate_model_size_gb,
    extract_params,
    format_likes,
    infer_quant_from_name,
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


def classify_hidden_gem(downloads, likes):
    if downloads < 5000:
        return False, 0.0
    if likes > 200:
        return False, 0.0
    like_ratio = likes / max(downloads, 1)
    if like_ratio > 0.005:
        return False, 0.0
    score = downloads / (likes + 1)
    return True, score


def _get_status_code(exc):
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) is not None:
        return response.status_code
    return getattr(exc, "status_code", None)


def _get_retry_after_seconds(exc):
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    retry_after = headers.get("Retry-After")
    if not retry_after:
        return None
    try:
        return int(retry_after)
    except (TypeError, ValueError):
        return None


def _format_hf_http_error(exc):
    status = _get_status_code(exc)
    retry_after = _get_retry_after_seconds(exc)
    if status == 429:
        if retry_after is not None:
            return f"Hugging Face rate-limited (429). Retry in {retry_after}s."
        return "Hugging Face rate-limited (429). Retry shortly."
    if status is not None:
        return f"Hugging Face request failed (HTTP {status})."
    return f"Hugging Face request failed: {exc}"


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
            expand=["likes", "siblings", "downloads"],
        )
    except HfHubHTTPError as exc:
        return results, [_format_hf_http_error(exc)]
    except (requests.RequestException, ValueError, OSError) as exc:
        return results, [f"Hugging Face search failed: {exc}"]

    for model in hf_models:
        try:
            repo_id = _repo_id_from_model(model)
            if not repo_id:
                raise ValueError("missing model repository id")

            publisher = repo_id.split("/")[0]
            provider = publisher[:15]
            name = repo_id.split("/")[-1]
            unique_key = f"Hugging Face:{repo_id.lower()}"
            if unique_key in found_keys:
                continue
            found_keys.add(unique_key)

            params = extract_params(name)
            use_case = determine_use_case(name)
            use_case_key = determine_use_case_key(name)
            likes = getattr(model, "likes", 0)
            downloads = int(getattr(model, "downloads", 0) or 0)
            is_hidden_gem, gem_score = classify_hidden_gem(downloads, likes)
            score_str = (
                f"[red]❤️ {format_likes(likes)}[/red]"
                if likes > 0
                else "[grey50]-[/grey50]"
            )
            if is_hidden_gem:
                score_str = f"{score_str} [yellow]💎[/yellow]"

            quant = infer_quant_from_name(name, default="GGUF")
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
                    "publisher": publisher,
                    "id": repo_id,
                    "name": name,
                    "params": params,
                    "use_case": use_case,
                    "use_case_key": use_case_key,
                    "score": score_str,
                    "likes": likes,
                    "downloads": downloads,
                    "is_hidden_gem": is_hidden_gem,
                    "gem_score": gem_score,
                    "quant": quant,
                    "target_file": target,
                    "size_source": "estimated",
                    "mode": mode_str,
                    "fit": fit_str,
                    "size": f"~{size:.1f} GB",
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
