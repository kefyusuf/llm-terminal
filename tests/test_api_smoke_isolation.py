from __future__ import annotations

import api_server


def test_api_health_smoke_does_not_repeat_hardware_probe(monkeypatch):
    """CLI smoke owns hardware probing; REST health smoke should test server lifecycle only."""

    def _unexpected_hardware_probe():
        raise AssertionError("API health smoke must not construct HardwareMonitor")

    monkeypatch.setattr(api_server, "HardwareMonitor", _unexpected_hardware_probe)

    assert api_server.run_smoke_check() == 0
