from __future__ import annotations

from types import SimpleNamespace

from app.viewer import AIModelViewer
from results.results_view import result_unique_key


def test_get_selected_model_uses_datatable_row_key_not_first_cell():
    model = {
        "source": "Ollama",
        "name": "qwen2.5:7b",
        "id": "qwen2.5:7b",
    }
    expected_key = result_unique_key(model)

    row_key = SimpleNamespace(value=expected_key)
    column_key = SimpleNamespace(value="inst")

    class _Table:
        cursor_row = 0
        row_count = 1
        cursor_coordinate = SimpleNamespace(row=0, column=0)

        def get_row_at(self, _row_index):
            # DataTable.get_row_at() returns cell values, not the row key.
            return ["-", "Ollama", "qwen2.5:7b"]

        def coordinate_to_cell_key(self, _coordinate):
            return row_key, column_key

    viewer = SimpleNamespace(
        all_results=[model],
        query_one=lambda *_args, **_kwargs: _Table(),
    )

    selected = AIModelViewer._get_selected_model(viewer)

    assert selected is model
