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


def test_action_cycle_sort_mode_updates_sort_key():
    viewer = AIModelViewer()
    viewer.sort_mode = "score"

    viewer.action_cycle_sort_mode()
    assert viewer.sort_mode == "downloads"


def test_action_cycle_fit_filter_updates_fit_key():
    viewer = AIModelViewer()
    viewer.fit_filter = "all"

    viewer.action_cycle_fit_filter()
    assert viewer.fit_filter == "fit"
