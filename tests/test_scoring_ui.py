"""Tests for scoring-related presentation and sorting."""

from results_presenter import (
    composite_score_cell_markup,
    moe_cell_markup,
    score_bar_cell_markup,
    tok_s_cell_markup,
)
from results_text import align_plain_cell, truncate_cell
from results_view import filter_results_for_view

# ---------------------------------------------------------------------------
# Score Bar Presentation
# ---------------------------------------------------------------------------


class TestScoreBarCellMarkup:
    def test_renders_qsfac(self):
        result = {
            "score_quality": 72,
            "score_speed": 45,
            "score_fit": 88,
            "score_context": 60,
        }
        bar = score_bar_cell_markup(
            result,
            width=30,
            truncate_plain=truncate_cell,
            align_plain=align_plain_cell,
        )
        assert "Q72" in bar
        assert "S45" in bar
        assert "F88" in bar
        assert "C60" in bar

    def test_zero_scores(self):
        result = {
            "score_quality": 0,
            "score_speed": 0,
            "score_fit": 0,
            "score_context": 0,
        }
        bar = score_bar_cell_markup(
            result,
            width=30,
            truncate_plain=truncate_cell,
            align_plain=align_plain_cell,
        )
        assert "Q0" in bar
        assert "S0" in bar

    def test_missing_scores_defaults_to_zero(self):
        result = {}
        bar = score_bar_cell_markup(
            result,
            width=30,
            truncate_plain=truncate_cell,
            align_plain=align_plain_cell,
        )
        assert "Q0" in bar


class TestCompositeScoreCellMarkup:
    def test_high_score_green(self):
        result = {"score_composite": 85}
        cell = composite_score_cell_markup(
            result,
            width=5,
            truncate_plain=truncate_cell,
            align_plain=align_plain_cell,
        )
        assert "#4fe08a" in cell
        assert "85" in cell

    def test_low_score_red(self):
        result = {"score_composite": 15}
        cell = composite_score_cell_markup(
            result,
            width=5,
            truncate_plain=truncate_cell,
            align_plain=align_plain_cell,
        )
        assert "#ff7f8f" in cell


class TestTokSCellMarkup:
    def test_fast_green(self):
        result = {"estimated_tok_s": 150.0}
        cell = tok_s_cell_markup(
            result,
            width=10,
            truncate_plain=truncate_cell,
            align_plain=align_plain_cell,
        )
        assert "#4fe08a" in cell
        assert "150" in cell

    def test_zero_dash(self):
        result = {"estimated_tok_s": 0.0}
        cell = tok_s_cell_markup(
            result,
            width=10,
            truncate_plain=truncate_cell,
            align_plain=align_plain_cell,
        )
        assert "-" in cell


class TestMoeCellMarkup:
    def test_moe_model(self):
        result = {"is_moe": True, "total_experts": 8, "active_experts": 2}
        cell = moe_cell_markup(
            result,
            width=10,
            truncate_plain=truncate_cell,
            align_plain=align_plain_cell,
        )
        assert "MoE 2/8" in cell

    def test_non_moe_model(self):
        result = {"is_moe": False}
        cell = moe_cell_markup(
            result,
            width=10,
            truncate_plain=truncate_cell,
            align_plain=align_plain_cell,
        )
        assert "-" in cell


# ---------------------------------------------------------------------------
# Composite Score Sorting
# ---------------------------------------------------------------------------


class TestCompositeSort:
    def _make_results(self):
        return [
            {
                "source": "Ollama",
                "name": "alpha",
                "use_case_key": "chat",
                "score_composite": 50,
                "score_quality": 60,
                "score_speed": 40,
                "is_hidden_gem": False,
                "fit": "[green]Perfect[/green]",
            },
            {
                "source": "Ollama",
                "name": "beta",
                "use_case_key": "chat",
                "score_composite": 80,
                "score_quality": 70,
                "score_speed": 90,
                "is_hidden_gem": False,
                "fit": "[green]Perfect[/green]",
            },
            {
                "source": "Ollama",
                "name": "gamma",
                "use_case_key": "chat",
                "score_composite": 65,
                "score_quality": 85,
                "score_speed": 45,
                "is_hidden_gem": False,
                "fit": "[yellow]Partial[/yellow]",
            },
        ]

    def test_composite_sort_descending(self):
        results = self._make_results()
        sorted_r = filter_results_for_view(
            results,
            current_filter="Ollama",
            use_case_filter="all",
            hidden_gems_only=False,
            sort_mode="composite",
        )
        scores = [r["score_composite"] for r in sorted_r]
        assert scores == [80, 65, 50]

    def test_speed_sort_descending(self):
        results = self._make_results()
        sorted_r = filter_results_for_view(
            results,
            current_filter="Ollama",
            use_case_filter="all",
            hidden_gems_only=False,
            sort_mode="speed",
        )
        speeds = [r["score_speed"] for r in sorted_r]
        assert speeds == [90, 45, 40]

    def test_quality_sort_descending(self):
        results = self._make_results()
        sorted_r = filter_results_for_view(
            results,
            current_filter="Ollama",
            use_case_filter="all",
            hidden_gems_only=False,
            sort_mode="quality",
        )
        qualities = [r["score_quality"] for r in sorted_r]
        assert qualities == [85, 70, 60]
