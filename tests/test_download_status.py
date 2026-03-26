from download_status import (
    is_active_state,
    label_for_state,
    map_service_job_status,
    state_markup_from_state_and_label,
)


def test_map_service_job_status_running_maps_to_downloading():
    assert map_service_job_status("running") == "downloading"


def test_map_service_job_status_running_with_cancel_maps_to_canceling():
    assert map_service_job_status("running", cancel_requested=True) == "canceling"


def test_label_for_state_cancelled_uses_canceled_spelling():
    assert label_for_state("cancelled") == "Canceled"


def test_state_markup_unknown_can_render_external():
    markup = state_markup_from_state_and_label("unknown", "", unknown_is_external=True)
    assert "External" in markup


def test_is_active_state_covers_canceling():
    assert is_active_state("canceling") is True
