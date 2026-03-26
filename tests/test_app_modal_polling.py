from app import AIModelViewer


def test_modal_pause_blocks_non_forced_poll_requests(monkeypatch):
    viewer = AIModelViewer()
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
    viewer = AIModelViewer()
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
