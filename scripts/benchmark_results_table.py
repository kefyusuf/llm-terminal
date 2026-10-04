"""Offline headless measurements of the shipped results table, not terminal FPS."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from textual.widgets import DataTable

from app.viewer import AIModelViewer


class TableOnlyViewer(AIModelViewer):
    def on_mount(self, event):
        event.prevent_default()
        self._configure_results_table_columns(force=True)

    def on_resize(self, _event):
        pass

    def _apply_resize_reflow(self, _generation):
        pass


def model(index):
    return {"id": f"benchmark-{index:04d}", "source": "Ollama", "provider": "Ollama",
            "publisher": "Fixture", "name": f"model-{index:04d}", "params": "7B",
            "use_case": "Coding", "use_case_key": "coding", "score": "50",
            "score_quality": 50, "score_composite": 50, "score_speed": 50,
            "estimated_tok_s": 50.0, "quant": "Q4_K_M", "mode": "GPU", "fit": "Perfect",
            "size": "4.0 GB", "downloads": 0, "likes": 0, "gem_score": 0.0,
            "is_hidden_gem": False, "inst": "-", "download_state": "idle"}


async def measure(rows, width, repetitions):
    app = TableOnlyViewer()
    async with app.run_test(size=(width, 40)) as pilot:
        app.current_filter = "Ollama"
        app.use_case_filter = app.fit_filter = "all"
        app.hidden_gems_only = False
        app.sort_mode = "name"
        app.all_results = [model(index) for index in range(rows)]
        app.refresh_table()
        await pilot.pause()
        table = app.query_one("#results-table", DataTable)
        counts = {"clear": 0, "update_cell": 0}
        for name in counts:
            original = getattr(table, name)

            def counted(*args, _name=name, _original=original, **kwargs):
                counts[_name] += 1
                return _original(*args, **kwargs)

            setattr(table, name, counted)
        samples = []
        for operation in ("unchanged", "one_download", "structural"):
            timings = []
            totals = dict.fromkeys(counts, 0)
            for iteration in range(repetitions):
                if operation == "one_download":
                    app.all_results[0]["download_state"] = "running" if iteration % 2 == 0 else "idle"
                elif operation == "structural":
                    app.all_results[0]["publisher"] = f"Changed-{iteration}"
                counts.update(dict.fromkeys(counts, 0))
                start = time.perf_counter()
                app.refresh_table()
                timings.append((time.perf_counter() - start) * 1000)
                for name in counts:
                    totals[name] += counts[name]
                await pilot.pause()
            samples.append({"operation": operation, "median_refresh_ms": statistics.median(timings),
                            "samples_ms": timings, "calls_total": totals})
        return {"rows": rows, "width": width, "repetitions": repetitions,
                "visible_columns": list(app.results_column_keys), "measurements": samples}


async def run(repetitions):
    return [await measure(rows, width, repetitions) for rows in (100, 1000) for width in (90, 160)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5, choices=range(1, 21))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    root = Path(__file__).resolve().parents[1]
    fingerprints = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in ("tui_app.py", "app/viewer.py", "scripts/benchmark_results_table.py")}
    report = {"schema_version": 1, "source_sha": source, "python": platform.python_version(),
              "runtime_source_sha256": fingerprints,
              "platform": platform.platform(), "measurement": "synchronous headless refresh only",
              "cases": asyncio.run(run(args.repetitions))}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
