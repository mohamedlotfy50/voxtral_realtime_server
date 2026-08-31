"""Process-singleton provider for the continuous-batching embedding worker.

``BatcherProvider.get()`` is a class-level lazy singleton (double-checked
locking) so the worker thread is started exactly once. ``shutdown()`` stops
the worker on app shutdown. Used as a FastAPI dependency via
``Depends(BatcherProvider.get)`` and called directly from the app lifespan.
"""
from __future__ import annotations

import threading

from app.shared.config.loader import ConfigService
from app.shared.infrastructure.embedder_provider import EmbedderProvider
from app.shared.infrastructure.embedding_batcher import EmbeddingBatcher


class BatcherProvider:
    """Lazy process-wide singleton for :class:`EmbeddingBatcher`."""

    _instance: EmbeddingBatcher | None = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> EmbeddingBatcher:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    settings = ConfigService.get().settings()
                    cls._instance = EmbeddingBatcher(
                        EmbedderProvider.get(),
                        max_batch_size=settings.max_batch_size,
                        max_wait_ms=settings.max_wait_ms,
                    )
        return cls._instance

    @classmethod
    def shutdown(cls) -> None:
        """Stop the background worker; safe to call on shutdown."""
        if cls._instance is not None:
            with cls._lock:
                if cls._instance is not None:
                    cls._instance.stop()
                    cls._instance = None
