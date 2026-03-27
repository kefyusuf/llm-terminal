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


def test_modal_pause_blocks_non_forced_poll_requests(monkeypatch):
    viewer = _make_viewer(monkeypatch)
    viewer._modal_poll_pause_count = 1

    called = {"system": 0, "download": 0}

    monkeypatch.setattr(
        viewer,
        "_run_system_info_refresh_worker",
        lambda: called.__setitem__("system", called["system"] + 1),
    )
    monkeypatch.setattr(
        viewer,
        "_run_download_poll_worker",
        lambda: called.__setitem__("download", called["download"] + 1),
    )

    viewer.request_system_info_refresh(force=False)
    viewer.request_download_poll(force=False)

    assert called["system"] == 0
    assert called["download"] == 0


def test_modal_resume_triggers_forced_refresh(monkeypatch):
    viewer = _make_viewer(monkeypatch)
    viewer._modal_poll_pause_count = 1

    forced_calls = {"system": 0, "download": 0}

    def _system(force=False):
        if force:
            forced_calls["system"] += 1

    def _download(force=False):
        if force:
            forced_calls["download"] += 1

    monkeypatch.setattr(viewer, "request_system_info_refresh", _system)
    monkeypatch.setattr(viewer, "request_download_poll", _download)

    viewer._set_modal_poll_pause(False)

    assert forced_calls["system"] == 1
    assert forced_calls["download"] == 1
