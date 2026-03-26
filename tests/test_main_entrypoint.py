import main


def test_main_invokes_app_run(monkeypatch):
    calls = []

    class _DummyViewer:
        def run(self):
            calls.append("run")

    monkeypatch.setattr(main, "AIModelViewer", _DummyViewer)
    main.main()

    assert calls == ["run"]
