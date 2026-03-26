from app import AIModelViewer


def test_use_case_label_returns_expected_text():
    viewer = AIModelViewer()
    assert viewer._use_case_label("coding") == "Coding"
    assert viewer._use_case_label("unknown") == "Any Use"


def test_action_cycle_use_case_updates_filter_key():
    viewer = AIModelViewer()
    viewer.use_case_filter = "all"

    viewer.action_cycle_use_case()
    assert viewer.use_case_filter == "chat"

    viewer.action_cycle_use_case()
    assert viewer.use_case_filter == "coding"
