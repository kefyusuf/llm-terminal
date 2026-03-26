from results_presenter import (
    download_cell_markup,
    fit_cell_markup,
    installed_cell_markup,
    mode_cell_markup,
    score_cell_markup,
    source_cell_markup,
    use_case_cell_markup,
)


def _truncate_plain(value, max_len):
    text = str(value or "-")
    return text[:max_len]


def _align_plain(value, width, align="left"):
    text = str(value or "-")
    if align == "right":
        return text.rjust(width)
    if align == "center":
        return text.center(width)
    return text.ljust(width)


def test_fit_cell_markup_perfect_uses_green():
    out = fit_cell_markup(
        "Perfect", width=7, truncate_plain=_truncate_plain, align_plain=_align_plain
    )
    assert "#4fe08a" in out


def test_mode_cell_markup_gpu_uses_green():
    out = mode_cell_markup("GPU", width=8, truncate_plain=_truncate_plain, align_plain=_align_plain)
    assert "#4fe08a" in out


def test_installed_cell_markup_detects_installed_marker():
    out = installed_cell_markup(
        "installed",
        width=3,
        truncate_plain=_truncate_plain,
        align_plain=_align_plain,
    )
    assert "●" in out


def test_source_cell_markup_hf_uses_provider_color():
    out = source_cell_markup(
        "Hugging Face",
        width=12,
        truncate_plain=_truncate_plain,
        align_plain=_align_plain,
    )
    assert "#b6a3ff" in out


def test_score_cell_markup_million_uses_green_band():
    out = score_cell_markup(
        "1.2M", width=8, truncate_plain=_truncate_plain, align_plain=_align_plain
    )
    assert "#4fe08a" in out


def test_use_case_cell_markup_coding_uses_coding_color():
    out = use_case_cell_markup(
        "Coding",
        width=8,
        truncate_plain=_truncate_plain,
        align_plain=_align_plain,
    )
    assert "#86a7ff" in out


def test_download_cell_markup_failed_uses_error_color():
    out = download_cell_markup(
        "Failed",
        width=8,
        truncate_plain=_truncate_plain,
        align_plain=_align_plain,
    )
    assert "#ff7f8f" in out
