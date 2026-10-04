"""Runtime declarations and bounded memory scenarios; no rescoring or allocation proof."""
from __future__ import annotations

import json
import re
import time


def _positive(value, maximum=1048576):
    return value if type(value) is int and 0 < value <= maximum else None


def _text(value):
    return value if isinstance(value, str) and 0 < len(value) <= 256 else None


def ollama_model_facts(show, entry, runtime_version):
    info = show.get("model_info") or {}
    info = info if isinstance(info, dict) else {}
    architecture = _text(info.get("general.architecture"))
    if architecture and not re.fullmatch(r"[a-z0-9_]+", architecture):
        architecture = None

    def field(name):
        return _positive(info.get(f"{architecture}.{name}"))

    heads, embedding = field("attention.head_count"), field("embedding_length")
    derived = embedding // heads if heads and embedding and embedding % heads == 0 else None
    key_declared = f"{architecture}.attention.key_length" in info
    value_declared = f"{architecture}.attention.value_length" in info
    details = show.get("details") or {}
    digest = entry.get("digest")
    digest = digest.lower() if isinstance(digest, str) and re.fullmatch(r"[a-fA-F0-9]{64}", digest) else None
    capabilities = show.get("capabilities") or []
    return {"schema_version": 1, "kind": "model_facts", "model": _text(entry.get("name")),
            "model_digest": digest, "architecture": architecture,
            "declared_supported_context": field("context_length"), "layers": field("block_count"),
            "kv_heads": field("attention.head_count_kv"),
            "key_head_dim": field("attention.key_length") if key_declared else derived,
            "value_head_dim": field("attention.value_length") if value_declared else derived,
            "head_dimension_provenance": {"key": "declared" if key_declared else "embedding_divided_by_attention_heads",
                                          "value": "declared" if value_declared else "embedding_divided_by_attention_heads"},
            "disk_size_bytes": _positive(entry.get("size"), 2**60),
            "quantization": _text(details.get("quantization_level")) if isinstance(details, dict) else None,
            "capabilities": [item for item in capabilities[:16] if _text(item)] if isinstance(capabilities, list) else [],
            "special_cache_layout": any(field(name) for name in (
                "attention.sliding_window", "attention.shared_kv_layers", "expert_count")),
            "provenance": {"kind": "runtime_declaration", "provider": "ollama",
                           "runtime_version": _text(runtime_version), "observed_at": time.time()}}


def read_model_facts(path, model):
    with open(path, "rb") as handle:
        raw = handle.read(65537)
    if len(raw) > 65536:
        raise ValueError("saved facts exceed the 64 KiB limit")
    data = json.loads(raw)
    if isinstance(data, dict) and data.get("kind") == "runtime_calibration":
        if data.get("schema_version") != 1:
            raise ValueError("unsupported calibration envelope schema")
        data = data.get("model_facts")
    if (not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("kind") != "model_facts"
        or data.get("model") != model or not isinstance(data.get("model_digest"), str)
        or not re.fullmatch(r"[a-fA-F0-9]{64}", data["model_digest"])):
        raise ValueError("saved model facts schema or identity does not match the selection")
    # Refuse non-finite or unsupported values even in fields not used by the formula.
    json.dumps(data, allow_nan=False)
    return data


def memory_scenario(facts, *, requested_context, concurrency=1, gpu_weight_percent=100,
                    kv_type="fp16", kv_device="gpu", overhead_mib=256):
    for value, low, high in ((requested_context, 1, 1048576), (concurrency, 1, 16),
                             (gpu_weight_percent, 0, 100), (overhead_mib, 0, 4096)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError("memory scenario assumption is outside the supported range")
    if kv_type not in ("fp16", "fp32") or kv_device not in ("gpu", "cpu"):
        raise ValueError("memory scenario assumption is unsupported")
    supported = _positive(facts.get("declared_supported_context"))
    result = {"status": "estimated", "requested_context": requested_context,
              "declared_supported_context": supported,
              "estimated_gpu_bytes": None, "estimated_cpu_bytes": None,
              "estimated_cpu_weight_bytes": None,
              "facts_provenance": {"model_digest": facts.get("model_digest"),
                                   "kind": "saved_runtime_declaration", "revalidated": False},
              "assumptions": {"concurrency": concurrency, "gpu_weight_percent": gpu_weight_percent,
                              "kv_type": kv_type, "kv_device": kv_device,
                              "overhead_mib": overhead_mib, "overhead_device": kv_device,
                              "weights": "one shared disk-size resident-weight proxy",
                              "cache": "full attention with uniform per-layer dimensions",
                              "runtime_allocation_measured": False}}
    if supported and requested_context > supported:
        result["status"] = "context_exceeds_declared_support"
        return result
    if facts.get("architecture") != "llama":
        result["status"] = "unsupported_architecture"
        return result
    if facts.get("special_cache_layout"):
        result["status"] = "unsupported_cache_layout"
        return result
    layers, heads, key_dim, value_dim, weights = (
        _positive(facts.get(name), 2**60 if name == "disk_size_bytes" else 1048576)
        for name in ("layers", "kv_heads", "key_head_dim", "value_head_dim", "disk_size_bytes")
    )
    if not all((layers, heads, key_dim, value_dim, weights)):
        result["status"] = "insufficient_metadata"
        return result
    cache = layers * heads * (key_dim + value_dim) * requested_context * concurrency * (2 if kv_type == "fp16" else 4)
    overhead = overhead_mib * 1024**2
    gpu_weights = (weights * gpu_weight_percent + 99) // 100
    cpu_weights = weights - gpu_weights
    result["components"] = {"weight_proxy_bytes": weights, "kv_cache_bytes": cache,
                            "reserved_overhead_bytes": overhead}
    result["estimated_gpu_bytes"] = gpu_weights + (cache + overhead if kv_device == "gpu" else 0)
    result["estimated_cpu_weight_bytes"] = cpu_weights
    result["estimated_cpu_bytes"] = cpu_weights + (cache + overhead if kv_device == "cpu" else 0)
    return result
