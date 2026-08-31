"""In-memory session repository (data-access layer, no HTTP concerns).

Owns the thread-safe storage of diarization sessions. It is constructed with
a ``labeler_factory`` so it stays agnostic of configuration: the service
layer injects a factory built from :class:`DiarizationSettings`.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Callable

from app.diarization.domain.pyannote_labeler import PyannoteLabeler
from app.diarization.repositories.session import Session
from app.diarization.repositories.session_not_found import SessionNotFound


class SessionRepository:
    """Thread-safe in-memory store of :class:`Session` objects."""

    _log = logging.getLogger("app.diarization.repositories.session_repository")

    def __init__(self, labeler_factory: Callable[[], PyannoteLabeler]):
        self._labeler_factory = labeler_factory
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self) -> str:
        sid = uuid.uuid4().hex
        labeler = self._labeler_factory()
        now = time.time()
        with self._lock:
            self._sessions[sid] = Session(sid, labeler, now, now)
        return sid

    def get(self, sid: str) -> Session:
        with self._lock:
            s = self._sessions.get(sid)
            if s is None:
                raise SessionNotFound(sid)
            s.last_used = time.time()
            return s

    def delete(self, sid: str) -> bool:
        with self._lock:
            return self._sessions.pop(sid, None) is not None

    def sweep(self, ttl_s: float) -> int:
        cutoff = time.time() - ttl_s
        with self._lock:
            stale = [sid for sid, s in self._sessions.items() if s.last_used < cutoff]
            for sid in stale:
                self._sessions.pop(sid, None)
        if stale:
            self._log.info("expired %d idle sessions", len(stale))
        return len(stale)

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)
