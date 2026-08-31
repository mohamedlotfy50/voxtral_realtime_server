"""Session lifecycle use-case: create / get / delete / sweep / count.

Translates domain/repository errors into HTTP errors, keeping the
repository free of FastAPI concerns.
"""
from __future__ import annotations

from fastapi import HTTPException

from app.diarization.repositories.session import Session
from app.diarization.repositories.session_not_found import SessionNotFound
from app.diarization.repositories.session_repository import SessionRepository


class SessionService:
    def __init__(self, repository: SessionRepository, session_ttl_s: float):
        self._repo = repository
        self.session_ttl_s = session_ttl_s

    def create(self) -> str:
        return self._repo.create()

    def get(self, sid: str) -> Session:
        try:
            return self._repo.get(sid)
        except SessionNotFound as e:
            raise HTTPException(status_code=404, detail="session not found") from e

    def delete(self, sid: str) -> bool:
        return self._repo.delete(sid)

    def sweep(self) -> int:
        return self._repo.sweep(self.session_ttl_s)

    def count(self) -> int:
        return self._repo.count()
