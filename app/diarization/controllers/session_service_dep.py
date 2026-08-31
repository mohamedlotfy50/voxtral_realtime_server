"""Class-based FastAPI dependency providing the session service.

FastAPI instantiates :class:`SessionServiceDep` per request, resolving
``request`` as a sub-dependency; the service is then read from
``app.state`` (populated by the app lifespan in ``app.server``).
"""
from __future__ import annotations

from fastapi import Request

from app.diarization.services.session_service import SessionService


class SessionServiceDep:
    """Exposes the process-wide :class:`SessionService` via ``self.value``."""

    def __init__(self, request: Request):
        self.value: SessionService = request.app.state.session_service
