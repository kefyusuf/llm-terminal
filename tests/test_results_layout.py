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
