"""Domain exception raised when a session id is unknown to the repository."""
from __future__ import annotations


class SessionNotFound(Exception):
    """Raised when a session id is unknown to the repository."""

    def __init__(self, session_id: str):
        super().__init__(f"session not found: {session_id}")
        self.session_id = session_id
