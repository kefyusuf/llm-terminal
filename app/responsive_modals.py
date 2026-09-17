"""Responsive runtime modal variants for constrained terminal sizes."""

from __future__ import annotations

from app.modals import ModelDetailModal as BaseModelDetailModal


class ModelDetailModal(BaseModelDetailModal):
    """Keep model-detail actions visible while long metadata scrolls."""

    CSS = BaseModelDetailModal.CSS + """
    #modal-container {
        max-height: 90%;
        overflow-y: auto;
        padding: 1 2 4 2;
    }
    #button-row {
        dock: bottom;
        background: #0f141f;
    }
    """
