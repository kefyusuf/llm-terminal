"""Reusable widgets for the AI Model Explorer TUI."""

from __future__ import annotations

from textual.widgets import Static


class SystemInfoWidget(Static):
    """Header widget displaying live CPU, RAM, GPU VRAM, and Ollama status."""

    def update_info(self, specs, ollama_running):
        """Re-render the widget with fresh hardware *specs* and Ollama running state."""
        gpu_color = "#f0ef8a" if specs["has_gpu"] else "#ff7f8f"
        ollama_status = (
            "[bold #4fe08a]running[/bold #4fe08a]"
            if ollama_running
            else "[bold #ff7f8f]stopped[/bold #ff7f8f]"
        )
        text = (
            f"[#93a7d9]CPU:[/#93a7d9] {specs['cpu_name']} ({specs['cpu_cores']} cores)  "
            f"[#5f6f97]|[/#5f6f97] [#93a7d9]RAM:[/#93a7d9] [bold #7edfff]{specs['ram_free']:.1f} GB avail[/bold #7edfff] / {specs['ram_total']:.1f} GB  "
            f"[#5f6f97]|[/#5f6f97] [#93a7d9]GPU:[/#93a7d9] [{gpu_color}]{specs['gpu_name']} ({specs['vram_free']:.1f} GB)[/{gpu_color}]  "
            f"[#5f6f97]|[/#5f6f97] [#93a7d9]Ollama:[/#93a7d9] {ollama_status}"
        )
        self.update(text)
