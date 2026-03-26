from results_layout import column_keys_for_width, compute_column_widths


def test_column_keys_for_width_full_layout():
    keys = column_keys_for_width(140)
    assert keys == [
        "inst",
        "source",
        "publisher",
        "name",
        "params",
        "use_case",
        "score",
        "quant",
        "mode",
        "fit",
        "download",
    ]


def test_column_keys_for_width_compact_layout():
    assert column_keys_for_width(95) == ["inst", "source", "name", "score", "download"]


def test_column_keys_for_width_compact_mode_layout():
    assert column_keys_for_width(120, compact=True) == [
        "inst",
        "name",
        "params",
        "score",
        "quant",
        "fit",
        "use_case",
    ]


def test_column_keys_for_width_compact_mode_wide_layout():
    assert column_keys_for_width(180, compact=True) == [
        "inst",
        "source",
        "name",
        "params",
        "score",
        "quant",
        "fit",
        "mode",
        "use_case",
        "download",
    ]


def test_compute_column_widths_preserves_minimums():
    keys = ["inst", "source", "name", "score", "download"]
    widths = compute_column_widths(keys, available_width=95)
    assert widths["inst"] >= 8
    assert widths["source"] >= 10
    assert widths["name"] >= 18
    assert widths["score"] >= 8
    assert widths["download"] >= 4


def test_compute_column_widths_distributes_extra_to_expandables():
    keys = ["inst", "source", "publisher", "name", "download"]
    widths = compute_column_widths(keys, available_width=180)
    assert widths["name"] > 18
    assert widths["publisher"] > 8
    assert widths["download"] > 4


def test_compute_column_widths_compact_caps_name_growth():
    keys = ["inst", "name", "params", "score", "quant", "fit", "use_case"]
    widths = compute_column_widths(keys, available_width=220, compact=True)
    assert widths["name"] <= 28
    assert widths["fit"] >= 8


def test_compute_column_widths_compact_consumes_extra_space():
    keys = ["inst", "name", "params", "score", "quant", "fit", "use_case"]
    available_width = 150
    widths = compute_column_widths(keys, available_width=available_width, compact=True)
    separator_budget = len(keys) + 2
    target_content_width = max(40, available_width - separator_budget)
    assert sum(widths.values()) == target_content_width
