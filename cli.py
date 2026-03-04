import sys

import click
from rich.console import Console

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
