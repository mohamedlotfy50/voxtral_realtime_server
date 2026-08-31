"""Class-based FastAPI dependency providing the batch service."""
from __future__ import annotations

from fastapi import Request

from app.diarization.services.batch_service import BatchService


class BatchServiceDep:
    """Exposes the process-wide :class:`BatchService` via ``self.value``."""

    def __init__(self, request: Request):
        self.value: BatchService = request.app.state.batch_service
