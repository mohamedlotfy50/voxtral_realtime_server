"""Frozen transport settings for the diarization HTTP server."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ServerSettings:
    """Transport settings for the diarization HTTP server."""

    host: str = "0.0.0.0"
    port: int = 8001
    log_level: str = "info"

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
