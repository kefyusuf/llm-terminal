
import click
from rich.console import Console
from rich.table import Table

import cache_db
import config
from hardware import HardwareMonitor, check_ollama_running
from service_client import get_service_health

console = Console()


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
    "--provider", "-p", type=click.Choice(["ollama", "huggingface", "all"]), default="all"
)
@click.option("--limit", "-n", default=10, help="Max results")
@click.option(
    "--sort",
    "-s",
    type=click.Choice(["composite", "speed", "quality", "name"]),
    default="composite",
)
def search(query, provider, limit, sort):
    """Search for models matching QUERY."""
    from providers.hf_provider import search_hf_models
    from providers.ollama_provider import get_installed_ollama_models, search_ollama_models

    monitor = HardwareMonitor()
    specs = monitor.get_specs()
    results = []

    with console.status(f"Searching for '{query}'..."):
        if provider in ("all", "ollama"):
            local = get_installed_ollama_models()
            ollama_results, _ = search_ollama_models(query, specs, local, page_size=limit)
            results.extend(ollama_results)

        if provider in ("all", "huggingface"):
            hf_results, _ = search_hf_models(query, specs, {}, limit=limit)
            results.extend(hf_results)

    # Sort
    if sort == "composite":
        results.sort(key=lambda r: r.get("score_composite", 0), reverse=True)
    elif sort == "speed":
        results.sort(key=lambda r: r.get("score_speed", 0), reverse=True)
    elif sort == "quality":
        results.sort(key=lambda r: r.get("score_quality", 0), reverse=True)
    elif sort == "name":
        results.sort(key=lambda r: r.get("name", "").lower())

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
    from providers.hf_provider import search_hf_models
    from providers.ollama_provider import get_installed_ollama_models, search_ollama_models

    monitor = HardwareMonitor()
    specs = monitor.get_specs()
    results = []

    with console.status("Scanning models for hardware fit..."):
        local = get_installed_ollama_models()
        ollama_results, _ = search_ollama_models("*", specs, local, page_size=50)
        results.extend(ollama_results)

        hf_results, _ = search_hf_models("*", specs, {}, limit=50)
        results.extend(hf_results)

    # Sort by composite score
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
    from providers.hf_provider import search_hf_models
    from providers.ollama_provider import get_installed_ollama_models, search_ollama_models

    monitor = HardwareMonitor()
    specs = monitor.get_specs()
    results = []

    with console.status(f"Finding best {use_case} models..."):
        local = get_installed_ollama_models()
        ollama_results, _ = search_ollama_models("*", specs, local, page_size=50)
        results.extend(ollama_results)

        hf_results, _ = search_hf_models("*", specs, {}, limit=50)
        results.extend(hf_results)

    # Filter by use case and fit
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
                }
            )
        console.print(json_mod.dumps(data, indent=2))
    else:
        table = Table(title=f"Top {use_case.title()} Recommendations")
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
@click.option("--context", "-c", default=4096, help="Target context length")
def plan(model_name, context):
    """Show hardware requirements for MODEL_NAME across quantization levels."""
    from model_intelligence import plan_hardware_for_model

    plans = plan_hardware_for_model(model_name, target_context=context)

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
@click.argument("model_name")
def scores(model_name):
    """Show detailed scoring breakdown for MODEL_NAME."""
    from scoring import score_model
    from utils import (
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
    console.print()


@cli.command()
def cache_clear():
    """Clear the model metadata cache."""
    cache_db.init_db()
    cache_db.cleanup_old_entries(max_per_source=0, ttl_seconds=0)
    console.print("[green]Cache cleared successfully[/green]")


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
def version():
    """Show version information."""
    from pathlib import Path

    version_file = Path(__file__).parent / "pyproject.toml"
    version_str = "0.1.0"
    if version_file.exists():
        import tomllib

        with open(version_file, "rb") as f:
            data = tomllib.load(f)
            version_str = data.get("project", {}).get("version", "0.1.0")

    console.print(f"AI Model Explorer v{version_str}")


if __name__ == "__main__":
    cli()
