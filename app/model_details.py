"""Model detail, hardware-plan, and comparison interactions."""

from __future__ import annotations

from app.modals import ComparisonModal, PlanModeModal
from app.responsive_modals import ModelDetailModal
from core.logging_ import get_logger
from core.model_intelligence import plan_hardware_for_model
from providers.hf_provider import enrich_hf_model_details
from results.results_view import result_unique_key

logger = get_logger(__name__)

MAX_COMPARISON = 4


class DetailsController:
    """Own the comparison set and open detail, plan, and comparison modals."""

    def __init__(self, app, state, *, hf_model_info_cache):
        self.app = app
        self.state = state
        self.hf_model_info_cache = hf_model_info_cache
        self.comparison_set: list[dict] = []

    def _status(self, text) -> None:
        self.app.status.update(text)

    def open_plan_mode(self) -> None:
        """Analyze hardware requirements for the selected model."""
        model = self.app.table.selected_model()
        if not model:
            self._status("No model selected for plan analysis.")
            return
        model_name = model.get("name", "")
        try:
            self.app.push_screen(PlanModeModal(model_name, plan_hardware_for_model(model_name)))
        except Exception as exc:
            logger.warning("Plan analysis failed for {}: {}", model_name, exc)
            self._status(f"Plan analysis failed: {exc}")

    def toggle_comparison(self) -> None:
        """Toggle the selected model in or out of the comparison set."""
        model = self.app.table.selected_model()
        if not model:
            self._status("No model selected for comparison.")
            return
        model_key = result_unique_key(model)
        for index, item in enumerate(self.comparison_set):
            if result_unique_key(item) == model_key:
                self.comparison_set.pop(index)
                count = len(self.comparison_set)
                self._status(f"Removed from comparison. ({count}/{MAX_COMPARISON} models)")
                return
        if len(self.comparison_set) >= MAX_COMPARISON:
            self._status(
                f"Comparison set full (max {MAX_COMPARISON}). "
                "Press C to view or remove a model first."
            )
            return
        self.comparison_set.append(model)
        count = len(self.comparison_set)
        self._status(f"Added to comparison. ({count}/{MAX_COMPARISON} models) Press C to view.")

    def show_comparison(self) -> None:
        if len(self.comparison_set) < 2:
            self._status(
                "Need at least 2 models for comparison. "
                f"({len(self.comparison_set)}/{MAX_COMPARISON} selected)"
            )
            return
        self.app.push_screen(ComparisonModal(self.comparison_set))

    def on_result_selected(self, row_key) -> None:
        model = self.app.table.model_for_key(row_key)
        if not model:
            return
        if model["source"] == "Hugging Face":
            self._status("Loading detailed model metadata...")
            self.open_hf_detail(model)
        else:
            self.open_detail_modal(model)

    def open_detail_modal(self, model) -> None:
        self.app.push_screen(ModelDetailModal(model))

    def open_hf_detail(self, model) -> None:
        self.app.run_worker(lambda: self._hf_detail_worker(model), thread=True, group="details")

    def _hf_detail_worker(self, model) -> None:
        try:
            specs = self.app.status.hardware_monitor().get_specs()
        except (OSError, ValueError, RuntimeError):
            self.app.call_from_thread(
                self._status, "Hardware detection unavailable. Retry model details."
            )
            return
        enriched = enrich_hf_model_details(model.copy(), specs, self.hf_model_info_cache)
        try:
            enriched["download_plan"] = self.app.dl.preview_job(enriched)
        except (OSError, ValueError, RuntimeError):
            enriched["download_plan"] = {"status": "service_unavailable", "allowed": False}
        self.app.call_from_thread(self.on_hf_detail_ready, enriched)

    def on_hf_detail_ready(self, enriched_model) -> None:
        model_id = enriched_model.get("id")
        if model_id:
            results = self.state.results
            for index, item in enumerate(results):
                if item.get("source") == "Hugging Face" and item.get("id") == model_id:
                    results[index] = enriched_model
                    break
            self.app.table.refresh()
        fetch_status = enriched_model.get("metadata_fetch_status")
        if fetch_status == "available":
            self._status("Detailed metadata loaded.")
        elif fetch_status == "cached":
            self._status("Showing cached detailed metadata.")
        else:
            self._status("Detailed metadata unavailable. Showing retained facts and estimates.")
        self.open_detail_modal(enriched_model)
