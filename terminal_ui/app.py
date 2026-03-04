"""
╔══════════════════════════════════════════════════════════════════╗
║  OBSIDIAN CONSOLE                                               ║
║  A brutally refined terminal workspace — built with Textual     ║
║                                                                  ║
║  Run:  python -m terminal_ui                                    ║
║  Keys: ctrl+q quit · ctrl+t terminal · ctrl+d dashboard        ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.markup import escape
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Input,
    Label,
    RichLog,
    Static,
)

# ── Helpers ──────────────────────────────────────────────────────

def get_cpu_percent() -> float:
    """Get CPU usage without psutil (cross-platform)."""
    try:
        import psutil
        return psutil.cpu_percent(interval=0)
    except ImportError:
        pass
    # Fallback: very rough estimation
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "loadpercentage"],
                capture_output=True, text=True, timeout=2
            )
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line.isdigit():
                    return float(line)
        except Exception:
            pass
    return 0.0


def get_memory_info() -> tuple[float, float, float]:
    """Return (used_gb, total_gb, percent)."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.used / (1024**3), mem.total / (1024**3), mem.percent
    except ImportError:
        pass
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "os", "get", "TotalVisibleMemorySize,FreePhysicalMemory"],
                capture_output=True, text=True, timeout=2
            )
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            if len(lines) >= 2:
                parts = lines[1].split()
                if len(parts) >= 2:
                    free_kb = float(parts[0])
                    total_kb = float(parts[1])
                    used_kb = total_kb - free_kb
                    return used_kb / (1024**2), total_kb / (1024**2), (used_kb / total_kb) * 100
        except Exception:
            pass
    return 0.0, 0.0, 0.0


def get_disk_info() -> tuple[float, float, float]:
    """Return (used_gb, total_gb, percent)."""
    try:
        usage = shutil.disk_usage(os.path.expanduser("~"))
        total_gb = usage.total / (1024**3)
        used_gb = usage.used / (1024**3)
        pct = (usage.used / usage.total) * 100
        return used_gb, total_gb, pct
    except Exception:
        return 0.0, 0.0, 0.0


def run_shell_command(cmd: str, cwd: str | None = None) -> str:
    """Execute a shell command and return output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
            cwd=cwd,
        )
        output = result.stdout.strip()
        if result.returncode != 0 and result.stderr.strip():
            output += f"\n[red]{result.stderr.strip()}[/red]"
        return output if output else "[dim]No output[/dim]"
    except subprocess.TimeoutExpired:
        return "[yellow]Command timed out (15s)[/yellow]"
    except Exception as e:
        return f"[red]Error: {e}[/red]"


# ── Horizontal Bar Widget ───────────────────────────────────────

class BarGraph(Static):
    """A sleek inline progress bar."""

    value: reactive[float] = reactive(0.0)
    bar_max: reactive[float] = reactive(100.0)
    bar_style: reactive[str] = reactive("cyan")

    def __init__(
        self,
        value: float = 0.0,
        bar_max: float = 100.0,
        bar_style: str = "cyan",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.value = value
        self.bar_max = bar_max
        self.bar_style = bar_style

    def render(self) -> Text:
        width = max(self.size.width - 2, 10)
        pct = min(self.value / self.bar_max, 1.0) if self.bar_max > 0 else 0
        filled = int(width * pct)
        empty = width - filled

        # Choose color based on threshold
        if pct > 0.85:
            color = "red"
        elif pct > 0.65:
            color = "#e8a838"
        else:
            color = self.bar_style

        bar = Text()
        bar.append("▐", style="#1a1a2e")
        bar.append("█" * filled, style=color)
        bar.append("░" * empty, style="#1a1a2e")
        bar.append("▌", style="#1a1a2e")
        return bar


# ── System Monitor Card ────────────────────────────────────────

class SystemMonitor(Static):
    """Live system resource monitor."""

    def compose(self) -> ComposeResult:
        yield Static("  SYSTEM", classes="card-title")
        yield Static("resources & utilization", classes="card-subtitle")

        # CPU
        with Horizontal(classes="metric-row"):
            yield Static("  CPU", classes="metric-label")
            yield Static("0%", id="cpu-val", classes="metric-value")
        yield BarGraph(id="cpu-bar", bar_style="#4a9eff")

        yield Static("")

        # Memory
        with Horizontal(classes="metric-row"):
            yield Static("  MEM", classes="metric-label")
            yield Static("0 / 0 GB", id="mem-val", classes="metric-value")
        yield BarGraph(id="mem-bar", bar_style="#44cc88")

        yield Static("")

        # Disk
        with Horizontal(classes="metric-row"):
            yield Static("  DISK", classes="metric-label")
            yield Static("0 / 0 GB", id="disk-val", classes="metric-value")
        yield BarGraph(id="disk-bar", bar_style="#e8a838")

        yield Static("")

        # System Info
        yield Static("", id="sys-info")

    def update_metrics(self) -> None:
        """Refresh all system metrics."""
        # CPU
        cpu = get_cpu_percent()
        self.query_one("#cpu-val", Static).update(f"{cpu:.0f}%")
        self.query_one("#cpu-bar", BarGraph).value = cpu

        # Memory
        used, total, pct = get_memory_info()
        self.query_one("#mem-val", Static).update(f"{used:.1f} / {total:.1f} GB")
        self.query_one("#mem-bar", BarGraph).value = pct

        # Disk
        d_used, d_total, d_pct = get_disk_info()
        self.query_one("#disk-val", Static).update(f"{d_used:.0f} / {d_total:.0f} GB")
        self.query_one("#disk-bar", BarGraph).value = d_pct

        # System
        info_lines = [
            f"[#3d3d5c]  OS     [#5a5a7a]{platform.system()} {platform.release()}",
            f"[#3d3d5c]  HOST   [#5a5a7a]{platform.node()}",
            f"[#3d3d5c]  PY     [#5a5a7a]{platform.python_version()}",
        ]
        self.query_one("#sys-info", Static).update("\n".join(info_lines))


# ── Process Table Card ─────────────────────────────────────────

class ProcessCard(Static):
    """A data table showing running Python processes."""

    def compose(self) -> ComposeResult:
        yield Static("  PROCESSES", classes="card-title")
        yield Static("active python processes", classes="card-subtitle")
        yield DataTable(id="proc-table")

    def on_mount(self) -> None:
        table = self.query_one("#proc-table", DataTable)
        table.add_columns("PID", "NAME", "CPU%", "MEM MB")
        table.cursor_type = "row"
        table.zebra_stripes = True

    def update_processes(self) -> None:
        table = self.query_one("#proc-table", DataTable)
        table.clear()
        try:
            import psutil
            for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
                try:
                    info = proc.info
                    name = info["name"] or ""
                    if "python" in name.lower() or "node" in name.lower():
                        mem_mb = (info["memory_info"].rss / (1024**2)) if info["memory_info"] else 0
                        table.add_row(
                            str(info["pid"]),
                            name[:24],
                            f"{info['cpu_percent']:.1f}",
                            f"{mem_mb:.0f}",
                        )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except ImportError:
            # Fallback without psutil
            if platform.system() == "Windows":
                try:
                    result = subprocess.run(
                        ["tasklist", "/FI", "IMAGENAME eq python*", "/FO", "CSV", "/NH"],
                        capture_output=True, text=True, timeout=5,
                    )
                    for line in result.stdout.strip().split("\n"):
                        parts = line.strip().strip('"').split('","')
                        if len(parts) >= 5:
                            table.add_row(parts[1], parts[0][:24], "—", parts[4].replace(" K", ""))
                except Exception:
                    table.add_row("—", "psutil not installed", "—", "—")
            else:
                table.add_row("—", "psutil not installed", "—", "—")


# ── Quick Actions Card ─────────────────────────────────────────

class QuickActions(Static):
    """Shortcut buttons for common tasks."""

    def compose(self) -> ComposeResult:
        yield Static("  ACTIONS", classes="card-title")
        yield Static("quick commands", classes="card-subtitle")
        yield Button("  Git Status", id="act-git-status", classes="quick-action")
        yield Button("  Pip List", id="act-pip-list", classes="quick-action")
        yield Button("  Disk Usage", id="act-disk", classes="quick-action")
        yield Button("  Network", id="act-net", classes="quick-action")
        yield Button("  Env Vars", id="act-env", classes="quick-action")
        yield Button("  Clear Log", id="act-clear", classes="quick-action")


# ── Clock Widget ───────────────────────────────────────────────

class ClockWidget(Static):
    """Displays current time in the header."""

    def on_mount(self) -> None:
        self.set_interval(1, self.refresh_time)

    def refresh_time(self) -> None:
        now = datetime.now()
        self.update(f"[#3d3d5c]{now.strftime('%d %b')}  [#b8bfd6]{now.strftime('%H:%M')}[#3d3d5c]:{now.strftime('%S')}")


# ═══════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════

class ObsidianConsole(App):
    """A modern terminal workspace."""

    CSS_PATH = "theme.tcss"
    TITLE = "Obsidian Console"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("ctrl+t", "focus_terminal", "Terminal", show=True),
        Binding("ctrl+d", "toggle_dashboard", "Dashboard", show=True),
        Binding("ctrl+l", "clear_log", "Clear Log", show=True),
        Binding("ctrl+r", "refresh_metrics", "Refresh", show=True),
        Binding("escape", "focus_terminal", "Focus Terminal", show=False),
    ]

    show_dashboard: reactive[bool] = reactive(True)
    command_history: reactive[list] = reactive(list, init=False)
    history_index: reactive[int] = reactive(-1)
    working_dir: reactive[str] = reactive("")

    def __init__(self) -> None:
        super().__init__()
        self.command_history = []
        self.history_index = -1
        self.working_dir = os.getcwd()
        self._start_time = time.time()

    def compose(self) -> ComposeResult:
        # ── Header
        with Horizontal(id="header-bar"):
            yield Static("[bold #4a9eff]◆ OBSIDIAN[/bold #4a9eff] [#3d3d5c]CONSOLE[/#3d3d5c]", classes="title")
            yield Static("", classes="subtitle")
            yield ClockWidget()

        # ── Status bar (bottom)
        with Horizontal(id="status-bar"):
            yield Static(f"[#4a9eff]●[/] {platform.node()}", classes="status-item accent", id="st-host")
            yield Static(f"[#3d3d5c]│[/] [#5a5a7a]py {platform.python_version()}", classes="status-item")
            yield Static(f"[#3d3d5c]│[/] [#44cc88]●[/] [#5a5a7a]{platform.system()}", classes="status-item ok")
            yield Static("", id="st-uptime", classes="status-item")
            yield Static("", id="st-cwd", classes="status-item")

        # ── Command bar
        with Horizontal(id="command-bar"):
            yield Static("[bold #4a9eff]❯[/]", id="command-prompt")
            yield Input(
                placeholder="type a command…",
                id="command-input",
            )

        # ── Body
        with Horizontal(id="body"):
            # Sidebar
            with Vertical(id="sidebar"):
                yield Static("[bold #3d3d5c]WORKSPACE[/]", id="sidebar-title")
                yield Button("  Dashboard", id="nav-dashboard", classes="sidebar-item selected")
                yield Button("  Terminal", id="nav-terminal", classes="sidebar-item")
                yield Button("  Processes", id="nav-processes", classes="sidebar-item")
                yield Button("  Settings", id="nav-settings", classes="sidebar-item")
                yield Static("")
                yield Static("[bold #3d3d5c]QUICK INFO[/]", id="sidebar-title-2")
                yield Static("", id="sidebar-info")

            # Main content
            with Container(id="main-content"):
                # Dashboard view
                with Container(id="dashboard-grid"):
                    with Vertical(classes="card"):
                        yield SystemMonitor(id="sysmon")
                    with Vertical(classes="card"):
                        yield QuickActions(id="actions")
                    with Vertical(classes="card", id="log-card"):
                        yield Static("  LOG", classes="card-title")
                        yield Static("command output & events", classes="card-subtitle")
                        yield RichLog(id="app-log", wrap=True, highlight=True, markup=True)
                    with Vertical(classes="card"):
                        yield ProcessCard(id="proc-card")

        yield Footer()

    def on_mount(self) -> None:
        """Initialize on startup."""
        self._refresh_sidebar_info()
        log = self.query_one("#app-log", RichLog)
        log.write("[#4a9eff]◆[/] [bold #b8bfd6]Obsidian Console[/] [#3d3d5c]initialized[/]")
        log.write(f"[#3d3d5c]  working directory:[/] [#5a5a7a]{self.working_dir}[/]")
        log.write(f"[#3d3d5c]  python:[/] [#5a5a7a]{sys.executable}[/]")
        log.write(f"[#3d3d5c]  platform:[/] [#5a5a7a]{platform.platform()}[/]")
        log.write("")
        log.write("[#3d3d5c]  Type commands below or use quick actions.[/]")
        log.write("[#3d3d5c]  Keybinds: [#4a9eff]ctrl+t[/] terminal · [#4a9eff]ctrl+d[/] dashboard · [#4a9eff]ctrl+r[/] refresh[/]")
        log.write("")

        # Start periodic refresh
        self.set_interval(3, self._periodic_refresh)
        # Initial metric load
        self._do_refresh()

    def _refresh_sidebar_info(self) -> None:
        """Update sidebar quick info."""
        info_widget = self.query_one("#sidebar-info", Static)
        now = datetime.now()
        uptime_s = int(time.time() - self._start_time)
        uptime_m, uptime_sec = divmod(uptime_s, 60)
        uptime_h, uptime_m = divmod(uptime_m, 60)
        lines = [
            f"  [#3d3d5c]Time[/]  [#5a5a7a]{now.strftime('%H:%M')}[/]",
            f"  [#3d3d5c]Date[/]  [#5a5a7a]{now.strftime('%d %b %Y')}[/]",
            f"  [#3d3d5c]Up[/]    [#5a5a7a]{uptime_h:02d}:{uptime_m:02d}:{uptime_sec:02d}[/]",
            "",
            f"  [#3d3d5c]CWD[/]",
            f"  [#5a5a7a]{Path(self.working_dir).name}/[/]",
        ]
        info_widget.update("\n".join(lines))

        # Status bar uptime
        uptime_w = self.query_one("#st-uptime", Static)
        uptime_w.update(f"[#3d3d5c]│[/] [#5a5a7a]up {uptime_h:02d}:{uptime_m:02d}:{uptime_sec:02d}[/]")

        cwd_w = self.query_one("#st-cwd", Static)
        cwd_w.update(f"[#3d3d5c]│[/] [#5a5a7a]{self.working_dir}[/]")

    def _do_refresh(self) -> None:
        """Refresh all dashboard metrics."""
        try:
            self.query_one("#sysmon", SystemMonitor).update_metrics()
        except Exception:
            pass
        try:
            self.query_one("#proc-card", ProcessCard).update_processes()
        except Exception:
            pass
        self._refresh_sidebar_info()

    def _periodic_refresh(self) -> None:
        self._do_refresh()

    # ── Actions ──────────────────────────────────────────────

    def action_focus_terminal(self) -> None:
        self.query_one("#command-input", Input).focus()

    def action_toggle_dashboard(self) -> None:
        self.show_dashboard = not self.show_dashboard

    def action_clear_log(self) -> None:
        self.query_one("#app-log", RichLog).clear()

    def action_refresh_metrics(self) -> None:
        self._do_refresh()
        log = self.query_one("#app-log", RichLog)
        log.write("[#44cc88]●[/] [#5a5a7a]Metrics refreshed[/]")

    # ── Command Input ────────────────────────────────────────

    @on(Input.Submitted, "#command-input")
    def on_command_submitted(self, event: Input.Submitted) -> None:
        cmd = event.value.strip()
        if not cmd:
            return

        inp = self.query_one("#command-input", Input)
        inp.value = ""

        # Save to history
        self.command_history.append(cmd)
        self.history_index = -1

        log = self.query_one("#app-log", RichLog)
        log.write(f"[bold #4a9eff]❯[/] [#b8bfd6]{escape(cmd)}[/]")

        # Handle built-in commands
        if cmd.lower() in ("clear", "cls"):
            log.clear()
            return
        if cmd.lower() == "exit":
            self.exit()
            return
        if cmd.lower().startswith("cd "):
            new_dir = cmd[3:].strip()
            try:
                target = os.path.abspath(os.path.join(self.working_dir, new_dir))
                if os.path.isdir(target):
                    self.working_dir = target
                    log.write(f"[#44cc88]●[/] [#5a5a7a]{self.working_dir}[/]")
                else:
                    log.write(f"[#ff4466]●[/] [#5a5a7a]not a directory: {escape(target)}[/]")
            except Exception as e:
                log.write(f"[#ff4466]●[/] [#5a5a7a]{escape(str(e))}[/]")
            self._refresh_sidebar_info()
            return

        # Execute async
        self._run_command(cmd)

    @work(thread=True)
    def _run_command(self, cmd: str) -> None:
        output = run_shell_command(cmd, cwd=self.working_dir)
        self.call_from_thread(self._display_output, output)

    def _display_output(self, output: str) -> None:
        log = self.query_one("#app-log", RichLog)
        for line in output.split("\n"):
            log.write(f"  [#5a5a7a]{line}[/]")
        log.write("")

    # ── Quick Action Buttons ─────────────────────────────────

    @on(Button.Pressed, "#act-git-status")
    def git_status(self) -> None:
        self._log_and_run("git status --short")

    @on(Button.Pressed, "#act-pip-list")
    def pip_list(self) -> None:
        self._log_and_run("pip list --format=columns 2>&1 | head -20")

    @on(Button.Pressed, "#act-disk")
    def disk_usage(self) -> None:
        if platform.system() == "Windows":
            self._log_and_run("wmic logicaldisk get size,freespace,caption")
        else:
            self._log_and_run("df -h")

    @on(Button.Pressed, "#act-net")
    def network_info(self) -> None:
        if platform.system() == "Windows":
            self._log_and_run("ipconfig | findstr /i \"IPv4 Subnet Gateway\"")
        else:
            self._log_and_run("ip -brief addr")

    @on(Button.Pressed, "#act-env")
    def env_vars(self) -> None:
        if platform.system() == "Windows":
            self._log_and_run("set | findstr /i \"PATH PYTHON VIRTUAL\"")
        else:
            self._log_and_run("env | grep -iE 'PATH|PYTHON|VIRTUAL' | head -10")

    @on(Button.Pressed, "#act-clear")
    def clear_log_btn(self) -> None:
        self.query_one("#app-log", RichLog).clear()

    def _log_and_run(self, cmd: str) -> None:
        log = self.query_one("#app-log", RichLog)
        log.write(f"[bold #4a9eff]❯[/] [#b8bfd6]{escape(cmd)}[/]")
        self._run_command(cmd)

    # ── Sidebar Navigation ───────────────────────────────────

    @on(Button.Pressed, "#nav-dashboard")
    def nav_dashboard(self) -> None:
        self._select_nav("nav-dashboard")

    @on(Button.Pressed, "#nav-terminal")
    def nav_terminal(self) -> None:
        self._select_nav("nav-terminal")
        self.query_one("#command-input", Input).focus()

    @on(Button.Pressed, "#nav-processes")
    def nav_processes(self) -> None:
        self._select_nav("nav-processes")
        self.query_one("#proc-card", ProcessCard).update_processes()

    @on(Button.Pressed, "#nav-settings")
    def nav_settings(self) -> None:
        self._select_nav("nav-settings")
        log = self.query_one("#app-log", RichLog)
        log.write("[#e8a838]●[/] [#5a5a7a]Settings panel — coming soon[/]")

    def _select_nav(self, active_id: str) -> None:
        for btn in self.query(".sidebar-item"):
            btn.remove_class("selected")
        try:
            self.query_one(f"#{active_id}").add_class("selected")
        except Exception:
            pass


# ── Entry Point ──────────────────────────────────────────────

def main() -> None:
    app = ObsidianConsole()
    app.run()


if __name__ == "__main__":
    main()
