import app as app_module


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

    monkeypatch.setattr(app_module, "HardwareMonitor", _DummyMonitor)
    return app_module.AIModelViewer()


def test_use_case_label_returns_expected_text(monkeypatch):
    viewer = _make_viewer(monkeypatch)
    assert viewer._use_case_label("coding") == "Coding"
    assert viewer._use_case_label("unknown") == "Any Use"


def test_action_cycle_use_case_updates_filter_key(monkeypatch):
    viewer = _make_viewer(monkeypatch)
    viewer.use_case_filter = "all"

    viewer.action_cycle_use_case()
    assert viewer.use_case_filter == "chat"

    viewer.action_cycle_use_case()
    assert viewer.use_case_filter == "coding"


def test_action_cycle_sort_mode_updates_sort_key(monkeypatch):
    viewer = _make_viewer(monkeypatch)
    viewer.sort_mode = "score"

    viewer.action_cycle_sort_mode()
    assert viewer.sort_mode == "downloads"


def test_action_cycle_fit_filter_updates_fit_key(monkeypatch):
    viewer = _make_viewer(monkeypatch)
    viewer.fit_filter = "all"

    viewer.action_cycle_fit_filter()
    assert viewer.fit_filter == "fit"


def test_compact_tags_are_shortened_for_toolbar(monkeypatch):
    viewer = _make_viewer(monkeypatch)

    assert viewer._use_case_compact_tag("coding") == "CODE"
    assert viewer._sort_compact_tag("downloads") == "DL"
    assert viewer._fit_compact_tag("partial") == "PART"


def test_compact_chip_text_contains_key_segments(monkeypatch):
    viewer = _make_viewer(monkeypatch)
    viewer.current_filter = "Hugging Face"
    viewer.use_case_filter = "coding"
    viewer.sort_mode = "downloads"
    viewer.fit_filter = "partial"
    viewer.hidden_gems_only = True
    viewer.current_page = 2

    chip_text = viewer._compact_chip_text(12, 50)

    assert "M:" in chip_text
    assert "HF" in chip_text
    assert "CODE" in chip_text
    assert "DL" in chip_text
    assert "PART" in chip_text
    assert "ON" in chip_text
