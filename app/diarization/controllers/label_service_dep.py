"""Class-based FastAPI dependency providing the label service."""
from __future__ import annotations

from fastapi import Request

from app.diarization.services.label_service import LabelService


class LabelServiceDep:
    """Exposes the process-wide :class:`LabelService` via ``self.value``."""

    def __init__(self, request: Request):
        self.value: LabelService = request.app.state.label_service
