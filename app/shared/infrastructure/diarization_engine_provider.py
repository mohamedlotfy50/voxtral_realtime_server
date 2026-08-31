"""Process-singleton provider for the diarization inference engine.

``DiarizationEngineProvider.get()`` is a class-level lazy singleton
(double-checked locking) so the worker pool and its pipelines are started
exactly once. ``shutdown()`` stops the workers and releases the pipelines
on app shutdown. Used by the app lifespan and by :class:`PyannoteLabeler`.
"""
from __future__ import annotations

import threading

from app.shared.config.loader import ConfigService
from app.shared.infrastructure.diarization_engine import DiarizationEngine
from app.shared.infrastructure.pyannote_pipeline_provider import PyannotePipelineProvider


class DiarizationEngineProvider:
    """Lazy process-wide singleton for :class:`DiarizationEngine`."""

    _instance: DiarizationEngine | None = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> DiarizationEngine:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    settings = ConfigService.get().settings()
                    pipelines = [
                        PyannotePipelineProvider.get(worker_id=i)
                        for i in range(settings.engine_workers)
                    ]
                    cls._instance = DiarizationEngine(pipelines)
        return cls._instance

    @classmethod
    def shutdown(cls) -> None:
        """Stop workers + release pipelines; safe to call on shutdown."""
        if cls._instance is not None:
            with cls._lock:
                if cls._instance is not None:
                    cls._instance.stop()
                    cls._instance = None
        PyannotePipelineProvider.shutdown()
