"""Model detail, hardware-plan, comparison, and download-row interactions."""

from textual import work
from textual.widgets import (
    DataTable,
)

from app.modals import (
    ComparisonModal,
    DownloadJobModal,
    ModelDetailModal,
    PlanModeModal,
)
from core.logging_ import get_logger
from core.model_intelligence import plan_hardware_for_model
from downloads.download_history import (
    cancel_model_payload,
    fallback_entry_from_target,
    is_external_entry,
)
from downloads.download_status import is_active_state
from providers.hf_provider import enrich_hf_model_details
from results.results_view import result_unique_key

logger = get_logger(__name__)


class ModelDetailsMixin:
    """Model detail, hardware-plan, comparison, and download-row interactions.

    Methods use the viewer's shared state and Textual APIs, preserving subclass hooks.
    """

    def action_open_plan_mode(self) -> None:
        """Open plan mode to analyze hardware requirements for the selected model."""
        model = self._get_selected_model()
        if not model:
            self.update_status("No model selected for plan analysis.")
            return

        model_name = model.get("name", "")

        try:
            plans = plan_hardware_for_model(model_name)
            self.push_screen(PlanModeModal(model_name, plans))
        except Exception as exc:
            logger.warning("Plan analysis failed for {}: {}", model_name, exc)
            self.update_status(f"Plan analysis failed: {exc}")

    def action_toggle_comparison(self) -> None:
        """Toggle the selected model in/out of the comparison set (max 4)."""
        model = self._get_selected_model()
        if not model:
            self.update_status("No model selected for comparison.")
            return

        model_key = result_unique_key(model)

        # Check if already in comparison set
        for i, m in enumerate(self.comparison_set):
            if result_unique_key(m) == model_key:
                self.comparison_set.pop(i)
                count = len(self.comparison_set)
                self.update_status(f"Removed from comparison. ({count}/4 models)")
                return

        # Add to comparison set
        if len(self.comparison_set) >= 4:
            self.update_status(
                "Comparison set full (max 4). Press C to view or remove a model first."
            )
            return

        self.comparison_set.append(model)
        count = len(self.comparison_set)
        self.update_status(f"Added to comparison. ({count}/4 models) Press C to view.")

    def action_show_comparison(self) -> None:
        """Show the comparison modal for selected models."""
        if len(self.comparison_set) < 2:
            self.update_status(
                f"Need at least 2 models for comparison. ({len(self.comparison_set)}/4 selected)"
            )
            return

        self.push_screen(ComparisonModal(self.comparison_set))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        data_table = getattr(event, "data_table", None)
        if data_table is not None and data_table.id == "download-history-table":
            target_id = str(event.row_key.value)
            entry = self.dl.download_registry.get(target_id)

            if not entry:
                entry = fallback_entry_from_target(target_id)

            cursor_column = data_table.cursor_column
            if cursor_column == 5:
                state = entry.get("state", "idle")

                if is_external_entry(entry):
                    self.update_status("External download; management unavailable in this app.")
                    return

                if is_active_state(state):
                    self.cancel_model_download(cancel_model_payload(target_id, entry))
                else:
                    self.delete_download_entry(target_id, delete_data=False)
            else:
                self.push_screen(DownloadJobModal(entry))
            return

        if data_table is not None and data_table.id != "results-table":
            return
        row_key = str(event.row_key.value)
        selected_model = next(
            (item for item in self.all_results if result_unique_key(item) == row_key),
            None,
        )
        if selected_model:
            if selected_model["source"] == "Hugging Face":
                self.update_status("Loading detailed model metadata...")
                self.open_hf_detail_worker(selected_model)
            else:
                self.open_model_detail_modal(selected_model)

    def open_model_detail_modal(self, model):
        self.push_screen(ModelDetailModal(model))

    @work(thread=True)
    def open_hf_detail_worker(self, selected_model):
        try:
            specs = self._get_hardware_monitor().get_specs()
        except (OSError, ValueError, RuntimeError):
            self.call_from_thread(
                self.update_status, "Hardware detection unavailable. Retry model details."
            )
            return
        enriched = enrich_hf_model_details(
            selected_model.copy(),
            specs,
            self.hf_model_info_cache,
        )
        try:
            enriched["download_plan"] = self.dl.preview_job(enriched)
        except (OSError, ValueError, RuntimeError):
            enriched["download_plan"] = {"status": "service_unavailable", "allowed": False}
        self.call_from_thread(self.on_hf_detail_ready, enriched)

    def on_hf_detail_ready(self, enriched_model):
        model_id = enriched_model.get("id")
        if model_id:
            for idx, item in enumerate(self.all_results):
                if item.get("source") == "Hugging Face" and item.get("id") == model_id:
                    self.all_results[idx] = enriched_model
                    break
            self.refresh_table()
        if enriched_model.get("metadata_fetch_status") == "available":
            self.update_status("Detailed metadata loaded.")
        elif enriched_model.get("metadata_fetch_status") == "cached":
            self.update_status("Showing cached detailed metadata.")
        else:
            self.update_status("Detailed metadata unavailable. Showing retained facts and estimates.")
        self.open_model_detail_modal(enriched_model)
