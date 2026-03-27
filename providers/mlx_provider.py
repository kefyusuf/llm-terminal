"""MLX provider — Apple Silicon optimized model discovery.

MLX is Apple's array framework optimized for Apple Silicon. Models are
typically stored in ``~/.cache/huggingface/hub/`` under ``mlx-community``
or downloaded via the ``mlx_lm`` package.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

from providers import BaseProvider
from scoring import enrich_result_with_scores
from utils import (
    calculate_fit,
    determine_use_case,
    determine_use_case_key,
    estimate_model_size_gb,
    extract_params,
    infer_quant_from_name,
)

# Common MLX model cache locations
_MLX_CACHE_PATHS = [
    Path.home() / ".cache" / "huggingface" / "hub",
    Path.home() / ".cache" / "lm-studio" / "models",
]


class MLXProvider(BaseProvider):
    """Provider for Apple Silicon MLX models."""

    slug = "mlx"
    display_name = "MLX"
    default_host = "local"

    def detect(self) -> bool:
        """Check if MLX is available (macOS + Apple Silicon only)."""
        if platform.system() != "Darwin":
            return False
        # Check for Apple Silicon
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "hw.optional.arm64"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return result.stdout.strip() == "1"
        except Exception:
            return False

    def search(
        self,
        query: str,
        specs: dict[str, Any],
        limit: int = 15,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Search locally cached MLX models."""
        results: list[dict[str, Any]] = []
        errors: list[str] = []

        for cache_path in _MLX_CACHE_PATHS:
            if not cache_path.exists():
                continue

            for model_dir in cache_path.iterdir():
                if not model_dir.is_dir():
                    continue

                dir_name = model_dir.name
                # Skip non-MLX community models if filtering
                if "mlx" not in dir_name.lower() and "lm-studio" not in str(cache_path).lower():
                    continue

                model_id = dir_name.replace("models--", "").replace("--", "/")
                name = model_id.split("/")[-1] if "/" in model_id else model_id

                if query and query != "*" and query.lower() not in model_id.lower():
                    continue

                publisher = model_id.split("/")[0] if "/" in model_id else "mlx"
                params = extract_params(name)
                use_case = determine_use_case(name)
                use_case_key = determine_use_case_key(name)
                quant = infer_quant_from_name(name, default="MLX")
                size_gb = self._estimate_dir_size(model_dir) or estimate_model_size_gb(name)

                fit_str, mode_str, _ = calculate_fit(size_gb, specs)

                result = {
                    "inst": "[green]✔[/green]",
                    "source": "MLX",
                    "provider": "MLX",
                    "publisher": publisher,
                    "id": model_id,
                    "name": name,
                    "params": params,
                    "use_case": use_case,
                    "use_case_key": use_case_key,
                    "score": "[grey50]-[/grey50]",
                    "likes": 0,
                    "downloads": 0,
                    "is_hidden_gem": False,
                    "gem_score": 0.0,
                    "quant": quant,
                    "size_source": "exact"
                    if size_gb != estimate_model_size_gb(name)
                    else "estimated",
                    "mode": mode_str,
                    "fit": fit_str,
                    "size": f"{size_gb:.1f} GB"
                    if size_gb != estimate_model_size_gb(name)
                    else f"~{size_gb:.1f} GB",
                    "_size_gb": size_gb,
                }
                enrich_result_with_scores(result, specs)
                results.append(result)

                if len(results) >= limit:
                    break

        return results, errors

    def list_installed(self) -> list[str]:
        """List locally cached MLX models."""
        installed: list[str] = []
        for cache_path in _MLX_CACHE_PATHS:
            if not cache_path.exists():
                continue
            for model_dir in cache_path.iterdir():
                if model_dir.is_dir():
                    model_id = model_dir.name.replace("models--", "").replace("--", "/")
                    installed.append(model_id.lower())
        return installed

    @staticmethod
    def _estimate_dir_size(path: Path) -> float | None:
        """Estimate directory size in GB."""
        try:
            total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            return total / (1024**3)
        except (OSError, PermissionError):
            return None
