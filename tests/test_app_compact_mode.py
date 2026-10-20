import app.status_bar as status_module
import tui_app as app_module
from app.search_constants import (
    FIT_COMPACT_TAGS,
    SORT_COMPACT_TAGS,
    USE_CASE_COMPACT_TAGS,
    USE_CASE_OPTIONS,
    option_label,
)


def _make_viewer(monkeypatch):
    class _DummyMonitor:
        def __init__(self):
            self.cpu_name = "Test CPU"
            self.cpu_cores = 8
            self.gpu_name = "No GPU"
            self.nvidia_available = False

        def get_specs(self):
            return {
                "cpu_name": self.cpu_name,
                "cpu_cores": self.cpu_cores,
                "ram_free": 8.0,
                "ram_total": 16.0,
                "vram_free": 0.0,
                "vram_total": 0.0,
                "gpu_name": self.gpu_name,
                "has_gpu": self.nvidia_available,
            }

    monkeypatch.setattr(status_module, "HardwareMonitor", _DummyMonitor)
    return app_module.AIModelViewer()


def test_use_case_label_returns_expected_text():
    assert option_label(USE_CASE_OPTIONS, "coding", "Any Use") == "Coding"
    assert option_label(USE_CASE_OPTIONS, "unknown", "Any Use") == "Any Use"


def test_action_cycle_use_case_updates_filter_key(monkeypatch):
    viewer = _make_viewer(monkeypatch)
    viewer.state.use_case_filter = "all"

    viewer.search.cycle_use_case()
    assert viewer.state.use_case_filter == "chat"

    viewer.search.cycle_use_case()
    assert viewer.state.use_case_filter == "coding"


def test_action_cycle_sort_mode_updates_sort_key(monkeypatch):
    viewer = _make_viewer(monkeypatch)
    viewer.state.sort_mode = "score"

    viewer.search.cycle_sort()
    assert viewer.state.sort_mode == "downloads"


def test_action_cycle_fit_filter_updates_fit_key(monkeypatch):
    viewer = _make_viewer(monkeypatch)
    viewer.state.fit_filter = "all"

    viewer.search.cycle_fit()
    assert viewer.state.fit_filter == "fit"


def test_compact_tags_are_shortened_for_toolbar():
    assert USE_CASE_COMPACT_TAGS["coding"] == "CODE"
    assert SORT_COMPACT_TAGS["downloads"] == "DL"
    assert FIT_COMPACT_TAGS["partial"] == "PART"


def test_compact_chip_text_contains_key_segments(monkeypatch):
    viewer = _make_viewer(monkeypatch)
    viewer.state.current_filter = "Hugging Face"
    viewer.state.use_case_filter = "coding"
    viewer.state.sort_mode = "downloads"
    viewer.state.fit_filter = "partial"
    viewer.state.hidden_gems_only = True
    viewer.state.current_page = 2

    chip_text = viewer.table.compact_chip_text(12, 50)

    assert "M:" in chip_text
    assert "HF" in chip_text
    assert "CODE" in chip_text
    assert "DL" in chip_text
    assert "PART" in chip_text
    assert "ON" in chip_text
