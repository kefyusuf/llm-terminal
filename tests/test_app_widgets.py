"""Tests for app.widgets — SystemInfoWidget."""

from __future__ import annotations

from app.widgets import SystemInfoWidget


def test_system_info_widget_update_info_contains_cpu():
    """update_info should include CPU name in rendered text."""
    widget = SystemInfoWidget()
    widget.update_info(
        {
            "cpu_name": "TestCPU",
            "cpu_cores": 8,
            "ram_free": 16.0,
            "ram_total": 32.0,
            "gpu_name": "TestGPU",
            "vram_free": 8.0,
            "has_gpu": True,
        },
        ollama_running=True,
    )
    rendered = str(widget.renderable)
    assert "TestCPU" in rendered


def test_system_info_widget_shows_ollama_running():
    widget = SystemInfoWidget()
    widget.update_info(
        {
            "cpu_name": "CPU",
            "cpu_cores": 4,
            "ram_free": 8.0,
            "ram_total": 16.0,
            "gpu_name": "GPU",
            "vram_free": 4.0,
            "has_gpu": True,
        },
        ollama_running=True,
    )
    rendered = str(widget.renderable)
    assert "running" in rendered


def test_system_info_widget_shows_ollama_stopped():
    widget = SystemInfoWidget()
    widget.update_info(
        {
            "cpu_name": "CPU",
            "cpu_cores": 4,
            "ram_free": 8.0,
            "ram_total": 16.0,
            "gpu_name": "GPU",
            "vram_free": 4.0,
            "has_gpu": False,
        },
        ollama_running=False,
    )
    rendered = str(widget.renderable)
    assert "stopped" in rendered
