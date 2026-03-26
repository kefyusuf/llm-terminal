from results_text import (
    align_plain_cell,
    blank_result_row,
    format_header_label,
    truncate_cell,
    truncate_plain_cell,
)


def test_truncate_cell_short_text_unchanged():
    assert truncate_cell("abc", 5) == "abc"


def test_truncate_cell_long_text_uses_ellipsis():
    assert truncate_cell("abcdef", 5) == "ab..."


def test_truncate_plain_cell_strips_markup():
    assert truncate_plain_cell("[green]Done[/green]", 10) == "Done"


def test_format_header_label_pads_to_width():
    assert format_header_label("Model", 8) == "Model   "


def test_align_plain_cell_right_alignment():
    assert align_plain_cell("42", 4, "right") == "  42"


def test_blank_result_row_contains_expected_keys():
    row = blank_result_row()
    assert row["name"] == "-"
    assert row["download"] == "-"
    assert len(row) == 11
