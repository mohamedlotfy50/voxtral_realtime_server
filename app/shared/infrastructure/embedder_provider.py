"""Process-singleton provider for the ECAPA speech embedder.

``EmbedderProvider.get()`` is a class-level lazy singleton (double-checked
locking) so the model is loaded exactly once per process and shared across
all requests. Used as a FastAPI dependency via ``Depends(EmbedderProvider.get)``
and called directly from the app lifespan.
"""
from __future__ import annotations

import threading

from app.shared.config.loader import ConfigService
from app.shared.infrastructure.embedder import SpeechEmbedder


class EmbedderProvider:
    """Lazy process-wide singleton for :class:`SpeechEmbedder`."""

    _instance: SpeechEmbedder | None = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> SpeechEmbedder:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    settings = ConfigService.get().settings()
                    cls._instance = SpeechEmbedder(settings.device)
        return cls._instance
