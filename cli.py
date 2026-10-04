import json
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

import config
from core import cache_db
from core.hardware import HardwareMonitor, check_ollama_running
from downloads.service_client import get_service_health
from providers.capabilities import get_all_provider_capabilities

console = Console()


_CLI_SEARCH_PROVIDER_ORDER = ("ollama", "huggingface")


def get_cli_search_provider_choices() -> tuple[str, ...]:
    """Return CLI search choices validated against canonical provider metadata."""
    capabilities = get_all_provider_capabilities()
    supported = tuple(
        slug
        for slug in _CLI_SEARCH_PROVIDER_ORDER
        if slug in capabilities and capabilities[slug].searchable
    )
    return ("all", *supported)


def _search_hf_models(query: str, specs: dict, model_info_cache: dict, *, limit: int):
    """Search Hugging Face from the CLI using the configured authentication token."""
    from providers.hf_provider import search_hf_models

    return search_hf_models(
        query,
        specs,
        model_info_cache,
        limit=limit,
        hf_token=config.settings.hf_token,
    )


def _emit_provider_errors(errors: list[str]) -> None:
    """Emit provider diagnostics on stderr without contaminating command stdout."""
    for error in errors:
        click.echo(f"Warning: {error}", err=True)


def _emit_json(payload):
    try:
        value = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise click.ClickException("export contains non-finite or unsupported JSON values") from exc
    click.echo(value)


def get_version() -> str:
    """Return the installed AI Model Explorer package version."""
    try:
        return package_version("ai-model-explorer")
    except PackageNotFoundError:
        return "0.0.0+local"


@click.group()
def cli():
    """AI Model Explorer - Terminal UI for discovering local LLM models."""
    pass


@cli.command()
def info():
    """Show system information and configuration."""
    console.print("\n[bold cyan]AI Model Explorer[/bold cyan]\n")

    console.print("[bold]Hardware:[/bold]")
    monitor = HardwareMonitor()
    specs = monitor.get_specs()
    console.print(f"  CPU: {specs['cpu_name']} ({specs['cpu_cores']} cores)")
    console.print(f"  RAM: {specs['ram_free']:.1f} GB free / {specs['ram_total']:.1f} GB")
    console.print(f"  GPU: {specs['gpu_name']}")
    console.print(f"  VRAM: {specs['vram_free']:.1f} GB free")

    console.print("\n[bold]Ollama:[/bold]")
    ollama_running = check_ollama_running()
    if ollama_running:
        console.print("  [green]Running[/green]")
    else:
        console.print("  [red]Not running[/red]")

    console.print("\n[bold]Download Service:[/bold]")
    try:
        health = get_service_health()
        console.print(f"  [green]Running[/green] (version {health.get('version', 'unknown')})")
    except Exception:
        console.print("  [red]Not running[/red]")

    console.print("\n[bold]Cache:[/bold]")
    console.print(f"  Database: {config.settings.cache_db_path}")
    console.print(f"  TTL: {config.settings.cache_ttl_seconds // 3600} hours")
    console.print(f"  Max entries per source: {config.settings.cache_max_per_source}")
    console.print()


@cli.command(name="system")
def system_info():
    """Show hardware system information."""
    monitor = HardwareMonitor()
    specs = monitor.get_specs()

    console.print("\n[bold cyan]System Hardware[/bold cyan]\n")
    console.print(f"  CPU:    {specs['cpu_name']} ({specs['cpu_cores']} cores)")
    console.print(f"  RAM:    {specs['ram_total']:.1f} GB total, {specs['ram_free']:.1f} GB free")
    console.print(f"  GPU:    {specs['gpu_name']}")
    console.print(f"  Vendor: {specs.get('gpu_vendor', 'unknown')}")
    console.print(f"  Backend: {specs.get('backend', 'cpu')}")
    console.print(f"  VRAM:   {specs['vram_total']:.1f} GB total, {specs['vram_free']:.1f} GB free")
    console.print(f"  GPUs:   {specs.get('gpu_count', 0)}")
    console.print()


@cli.command()
@click.argument("query")
@click.option(
    "--provider",
    "-p",
    type=click.Choice(get_cli_search_provider_choices()),
    default="all",
)
@click.option("--limit", "-n", default=10, help="Max results")
@click.option(
    "--sort",
    "-s",
    type=click.Choice(["composite", "speed", "quality", "name"]),
    default="composite",
)
@click.option("--json", "json_output", is_flag=True, help="Emit a schema-1 search export")
def search(query, provider, limit, sort, json_output=False):
    """Search for models matching QUERY."""
    from providers.ollama_provider import get_installed_ollama_models, search_ollama_models

    monitor = HardwareMonitor()
    specs = monitor.get_specs()
    results = []
    errors: list[str] = []

    with nullcontext() if json_output else console.status(f"Searching for '{query}'..."):
        if provider in ("all", "ollama"):
            local = get_installed_ollama_models()
            ollama_results, ollama_errors, _ = search_ollama_models(
                query, specs, local, page_size=limit
            )
            results.extend(ollama_results)
            errors.extend(ollama_errors)

        if provider in ("all", "huggingface"):
            hf_results, hf_errors = _search_hf_models(query, specs, {}, limit=limit)
            results.extend(hf_results)
            errors.extend(hf_errors)

    _emit_provider_errors(errors)

    if sort == "composite":
        results.sort(key=lambda r: r.get("score_composite", 0), reverse=True)
    elif sort == "speed":
        results.sort(key=lambda r: r.get("score_speed", 0), reverse=True)
    elif sort == "quality":
        results.sort(key=lambda r: r.get("score_quality", 0), reverse=True)
    elif sort == "name":
        results.sort(key=lambda r: r.get("name", "").lower())

    if json_output:
        from core.exports import export_model

        _emit_json({"schema_version": 1, "kind": "search", "query": query,
                    "provider": provider, "limit": limit, "sort": sort,
                    "models": [export_model(model) for model in results[:limit]], "errors": errors})
        return

    table = Table(title=f"Search: {query}")
    table.add_column("Model", style="bold")
    table.add_column("Source")
    table.add_column("Params")
    table.add_column("Quant")
    table.add_column("Size")
    table.add_column("Fit")
    table.add_column("Q")
    table.add_column("S")
    table.add_column("F")
    table.add_column("C")
    table.add_column("Score", style="bold")

    for r in results[:limit]:
        table.add_row(
            r.get("name", ""),
            r.get("source", ""),
            r.get("params", "-"),
            r.get("quant", ""),
            r.get("size", ""),
            r.get("fit", ""),
            str(r.get("score_quality", 0)),
            str(r.get("score_speed", 0)),
            str(r.get("score_fit", 0)),
            str(r.get("score_context", 0)),
            str(r.get("score_composite", 0)),
        )

    console.print(table)
    console.print(f"\n{len(results)} results found\n")


@cli.command()
@click.option("--perfect", is_flag=True, help="Show only perfect-fit models")
@click.option("--limit", "-n", default=5, help="Max results")
def fit(perfect, limit):
    """Find models that fit your hardware."""
    from providers.ollama_provider import get_installed_ollama_models, search_ollama_models

    monitor = HardwareMonitor()
    specs = monitor.get_specs()
    results = []
    errors: list[str] = []

    with console.status("Scanning models for hardware fit..."):
        local = get_installed_ollama_models()
        ollama_results, ollama_errors, _ = search_ollama_models(
            "*", specs, local, page_size=50
        )
        results.extend(ollama_results)
        errors.extend(ollama_errors)

        hf_results, hf_errors = _search_hf_models("*", specs, {}, limit=50)
        results.extend(hf_results)
        errors.extend(hf_errors)

    _emit_provider_errors(errors)
    results.sort(key=lambda r: r.get("score_composite", 0), reverse=True)

    if perfect:
        results = [r for r in results if "perfect" in r.get("fit", "").lower()]

    table = Table(title="Hardware Fit" + (" (Perfect Only)" if perfect else ""))
    table.add_column("Model", style="bold")
    table.add_column("Source")
    table.add_column("Size")
    table.add_column("Fit")
    table.add_column("Mode")
    table.add_column("Composite")

    for r in results[:limit]:
        table.add_row(
            r.get("name", ""),
            r.get("source", ""),
            r.get("size", ""),
            r.get("fit", ""),
            r.get("mode", ""),
            str(r.get("score_composite", 0)),
        )

    console.print(table)
    console.print(f"\nShowing top {min(limit, len(results))} of {len(results)} results\n")


@cli.command()
@click.option("--limit", "-n", default=5, help="Max recommendations")
@click.option(
    "--use-case",
    "-u",
    type=click.Choice(["chat", "coding", "vision", "reasoning", "math", "general"]),
    default="general",
)
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def recommend(limit, use_case, output_json):
    """Get model recommendations for your hardware."""
    from providers.ollama_provider import get_installed_ollama_models, search_ollama_models

    monitor = HardwareMonitor()
    specs = monitor.get_specs()
    results = []
    errors: list[str] = []

    with console.status(f"Finding best {use_case} models..."):
        local = get_installed_ollama_models()
        ollama_results, ollama_errors, _ = search_ollama_models(
            "*", specs, local, page_size=50
        )
        results.extend(ollama_results)
        errors.extend(ollama_errors)

        hf_results, hf_errors = _search_hf_models("*", specs, {}, limit=50)
        results.extend(hf_results)
        errors.extend(hf_errors)

    _emit_provider_errors(errors)
    results = [r for r in results if r.get("use_case_key") == use_case or use_case == "general"]
    results = [r for r in results if "no fit" not in r.get("fit", "").lower()]
    results.sort(key=lambda r: r.get("score_composite", 0), reverse=True)

    if output_json:
        import json as json_mod

        data = []
        for r in results[:limit]:
            data.append(
                {
                    "name": r.get("name"),
                    "source": r.get("source"),
                    "scores": {
                        "quality": r.get("score_quality", 0),
                        "speed": r.get("score_speed", 0),
                        "fit": r.get("score_fit", 0),
                        "context": r.get("score_context", 0),
                        "composite": r.get("score_composite", 0),
                    },
                    "score_provenance": r.get("score_provenance"),
                    "artifact_metadata": r.get("artifact_metadata"),
                }
            )
        click.echo(json_mod.dumps(data, indent=2))
    else:
        table = Table(title=f"Top {use_case.title()} Recommendations (heuristic estimates)")
        table.add_column("#", style="dim")
        table.add_column("Model", style="bold")
        table.add_column("Source")
        table.add_column("Q")
        table.add_column("S")
        table.add_column("F")
        table.add_column("C")
        table.add_column("Score", style="bold green")

        for i, r in enumerate(results[:limit], 1):
            table.add_row(
                str(i),
                r.get("name", ""),
                r.get("source", ""),
                str(r.get("score_quality", 0)),
                str(r.get("score_speed", 0)),
                str(r.get("score_fit", 0)),
                str(r.get("score_context", 0)),
                str(r.get("score_composite", 0)),
            )

        console.print(table)
    console.print()


@cli.command()
@click.argument("model_name")
@click.option("--context", "-c", type=click.IntRange(min=1), default=4096, help="Target context length")
@click.option("--json", "json_output", is_flag=True, help="Emit a schema-1 heuristic plan")
def plan(model_name, context, json_output=False):
    """Show hardware requirements for MODEL_NAME across quantization levels."""
    from core.model_intelligence import plan_hardware_for_model

    plans = plan_hardware_for_model(model_name, target_context=context)

    if json_output:
        _emit_json({"schema_version": 1, "kind": "hardware_plan", "model_name": model_name,
                    "requested_context": context, "artifact_metadata": None, "plans": plans,
                    "estimate_provenance": {
                        "kind": "heuristic", "weight_size": "model_name_estimate",
                        "context_overhead": "legacy_context_formula",
                        "offload_fraction": 0.7,
                        "limitation": "Requested context is not supported context; no runtime allocation or benchmark was measured."}})
        return

    table = Table(title=f"Hardware Plan: {model_name} (context={context})")
    table.add_column("Quant", style="bold")
    table.add_column("Size")
    table.add_column("VRAM Needed")
    table.add_column("Mode")
    table.add_column("Min GPU")

    for p in plans:
        color = "green" if p["quality_rank"] >= 8 else "yellow" if p["quality_rank"] >= 5 else "red"
        table.add_row(
            f"[{color}]{p['quant']}[/{color}]",
            f"{p['size_gb']:.1f} GB",
            f"{p['vram_needed']:.1f} GB",
            p["mode"],
            p["min_gpu_class"],
        )

    console.print(table)
    console.print()


@cli.command()
@click.argument("identities", nargs=-1)
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--json", "json_output", is_flag=True, help="Emit a schema-1 comparison")
def compare(identities, input_path, json_output):
    """Compare model IDs from a saved search export without discovery or rescoring."""
    from core.exports import comparison_export, read_search_export

    try:
        payload = comparison_export(read_search_export(input_path), identities)
    except (OSError, ValueError, TypeError, UnicodeError) as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_provider_errors(payload["errors"])
    if json_output:
        _emit_json(payload)
        return
    table = Table(title="Saved model comparison (heuristic estimates)")
    table.add_column("Model")
    table.add_column("Source")
    table.add_column("Composite")
    table.add_column("Revision")
    for model in payload["models"]:
        table.add_row(Text(str(model.get("name") or model.get("id"))),
                      Text(str(model.get("source") or "Unknown")),
                      str(model["scores"]["composite"]), Text(str(model.get("resolved_revision") or "Unknown")))
    console.print(table)


@cli.command()
@click.argument("model_name")
def scores(model_name):
    """Show detailed scoring breakdown for MODEL_NAME."""
    from core.scoring import score_model
    from core.utils import (
        determine_use_case_key,
        estimate_model_size_gb,
        extract_params,
        infer_quant_from_name,
    )

    monitor = HardwareMonitor()
    specs = monitor.get_specs()
    size_gb = estimate_model_size_gb(model_name)
    params = extract_params(model_name)
    quant = infer_quant_from_name(model_name)
    use_case_key = determine_use_case_key(model_name)
    mode = "GPU" if specs.get("has_gpu") else "CPU"

    result = score_model(
        model_name=model_name,
        size_gb=size_gb,
        params=params,
        quant=quant,
        use_case_key=use_case_key,
        specs=specs,
        mode=mode,
    )

    console.print(f"\n[bold cyan]Scores for {model_name}[/bold cyan]\n")
    console.print("Heuristic estimates; no runtime benchmark was performed.")
    console.print(f"  Params:    {params}")
    console.print(f"  Quant:     {quant}")
    console.print(f"  Size:      ~{size_gb:.1f} GB")
    console.print(f"  Use Case:  {use_case_key}")
    console.print(f"  Mode:      {mode}")
    console.print()
    console.print(f"  Quality:   [bold]{result.quality}[/bold]/100")
    console.print(
        f"  Speed:     [bold]{result.speed}[/bold]/100 ({result.estimated_tok_s:.0f} tok/s)"
    )
    console.print(f"  Fit:       [bold]{result.fit}[/bold]/100")
    console.print(f"  Context:   [bold]{result.context}[/bold]/100")
    console.print()
    console.print(
        f"  Composite: [bold green]{result.composite}[/bold green]/100 ({use_case_key} weights)"
    )
    for dimension in ("quality", "speed", "fit", "context"):
        console.print(f"  {dimension.title()}: {result.provenance[dimension]['limitation']}")
    speed_basis = result.provenance["speed"]
    console.print(f"  Bandwidth assumption: {speed_basis['bandwidth_gb_s']} GB/s ({speed_basis['bandwidth_source']})")
    console.print()


@cli.command()
def cache_clear():
    """Clear the model metadata cache."""
    cache_db.init_db()
    cache_db.cleanup_old_entries(max_per_source=0, ttl_seconds=0)
    console.print("[green]Cache cleared successfully[/green]")


@cli.command("download-plan")
@click.argument("repository")
@click.argument("filename")
@click.option("--revision", default=None, help="Resolve a specified upstream revision.")
@click.option("--offline", is_flag=True, help="Plan unknown metadata without network requests.")
@click.option("--allow-unknown-size", is_flag=True, help="Acknowledge an unknown disk requirement.")
@click.option("--json", "json_output", is_flag=True, help="Emit the versioned plan JSON.")
def download_plan(repository, filename, revision, offline, allow_unknown_size, json_output):
    """Inspect one exact HF file's destination and disk budget; never queue it."""
    import json
    import re

    from core.artifacts import hf_artifact_metadata
    from downloads.download_manager import validate_hf_target_file
    from downloads.preflight import plan_hf_download

    try:
        validate_hf_target_file(filename)
        info = None
        if not offline:
            from huggingface_hub import HfApi
            from huggingface_hub.errors import HfHubHTTPError
            from requests.exceptions import RequestException

            try:
                info = HfApi(token=config.settings.hf_token).model_info(repository, revision=revision, files_metadata=True, timeout=10)
            except (HfHubHTTPError, RequestException, OSError, ValueError) as exc:
                raise click.ClickException("Repository metadata is unavailable. Use doctor for diagnostics or --offline for an explicit unknown plan.") from exc
            if not any(getattr(item, "rfilename", None) == filename for item in info.siblings or []):
                raise click.ClickException("Selected file is absent from repository metadata.")
        artifact = hf_artifact_metadata(repository, filename, info)
        if offline and revision:
            if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
                raise ValueError("Offline revision must be a full immutable commit SHA.")
            artifact["resolved_revision"] = revision
            artifact["revision_status"] = "pinned"
            artifact["revision_origin"] = "requested"
        model = {"source": "Hugging Face", "id": repository, "target_file": filename,
                 "resolved_revision": artifact["resolved_revision"], "artifact_metadata": artifact}
        plan = plan_hf_download(model, config.settings.hf_models_dir, allow_unknown_size=allow_unknown_size)
        plan["artifact_metadata"] = artifact
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if json_output:
        click.echo(json.dumps(plan, indent=2))
    else:
        click.echo(f"Destination: {plan['destination']}\nFile bytes: {plan['size_bytes']}\nFree bytes: {plan['free_bytes']}\nRequired bytes (with safety margin): {plan['required_bytes']}\nStatus: {plan['status']}\nWarnings: {', '.join(plan['warnings']) or 'none'}")
        click.echo("Read the source/model license terms. This command never queues a download.")


@cli.command()
def cache_stats():
    """Show cache statistics."""
    import sqlite3

    conn = sqlite3.connect(str(config.settings.cache_db_path))
    cursor = conn.cursor()

    cursor.execute("SELECT source, COUNT(*) FROM model_cache GROUP BY source")
    counts = cursor.fetchall()

    console.print("\n[bold]Cache Statistics:[/bold]")
    for source, count in counts:
        console.print(f"  {source}: {count} entries")

    cursor.execute("SELECT COUNT(*) FROM hardware_snapshot")
    hw_count = cursor.fetchone()[0]
    console.print(f"  hardware snapshots: {hw_count}")

    conn.close()
    console.print()


@cli.command()
@click.option("--offline", is_flag=True, help="Check local configuration without network access.")
@click.option("--json", "json_output", is_flag=True, help="Emit shareable JSON diagnostics.")
@click.option(
    "--timeout",
    type=click.FloatRange(min=0, max=30, min_open=True),
    default=3.0,
    show_default=True,
    help="Total network wait budget in seconds.",
)
def doctor(offline: bool, json_output: bool, timeout: float):
    """Diagnose storage, runtime reachability, and TLS without exposing private configuration."""
    import json

    from core.diagnostics import collect_diagnostics

    report = collect_diagnostics(config.settings, offline=offline, timeout=timeout)
    report["app_version"] = get_version()
    if json_output:
        click.echo(json.dumps(report, indent=2))
    else:
        click.echo(f"AI Model Explorer {get_version()} diagnostics: {report['status']}")
        for item in report["checks"]:
            details = f" (version {item['version']})" if "version" in item else ""
            if "free_bytes" in item:
                details += f" ({item['free_bytes'] / 1024**3:.1f} GiB free)"
            click.echo(f"{item['name']}: {item['status']} / {item['code']}{details}")
            if item["action"]:
                click.echo(f"  {item['action']}")
    if report["status"] == "error":
        raise click.exceptions.Exit(1)


@cli.command()
def version():
    """Show version information."""
    console.print(f"AI Model Explorer v{get_version()}")


if __name__ == "__main__":
    cli()
