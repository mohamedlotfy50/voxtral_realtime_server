"""Aggregate configuration handed to the app factory."""
from __future__ import annotations

from dataclasses import dataclass, field

from app.shared.config.diarization_settings import DiarizationSettings
from app.shared.config.server_settings import ServerSettings
from app.shared.config.voxtral_settings import VoxtralSettings


@dataclass(frozen=True)
class AppConfig:
    """Aggregate configuration handed to the app factory.

    ``voxtral`` defaults to an empty :class:`VoxtralSettings` so the
    diarization-only deployment path (``uvicorn app.server:app``) boots even
    when the root ``config.yaml`` is absent.
    """

    diarization: DiarizationSettings = field(default_factory=DiarizationSettings)
    server: ServerSettings = field(default_factory=ServerSettings)
    voxtral: VoxtralSettings = field(default_factory=VoxtralSettings)
