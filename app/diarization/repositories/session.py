"""Session entity held by the session repository."""
from __future__ import annotations

from dataclasses import dataclass

from app.diarization.domain.pyannote_labeler import PyannoteLabeler


@dataclass
class Session:
    """A diarization session: id + its labeler + lifecycle timestamps."""

    id: str
    labeler: PyannoteLabeler
    created_at: float
    last_used: float
