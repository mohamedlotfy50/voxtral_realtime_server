"""Per-worker provider for pyannote.audio speaker-diarization pipeline instances.

Loads ``pyannote/speaker-diarization-3.1`` once per engine worker on the
configured device. Each :class:`DiarizationEngine` worker thread owns a
dedicated pipeline instance (``get(worker_id=...)``) so concurrent workers
never share a pipeline — the pyannote pipeline is not guaranteed thread-safe
for concurrent GPU access. Instances also get their native GPU batching
knobs (``segmentation_batch_size`` / ``embedding_batch_size``) applied so
every forward pass processes windows/crops in batches instead of one-by-one.

Requires a HuggingFace access token with accepted terms for:
  * ``pyannote/speaker-diarization-3.1``
  * ``pyannote/segmentation-3.0``

Set via ``hf_token`` in ``diarization.yaml`` or the ``HF_TOKEN`` env var.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from app.shared.config.loader import ConfigService
from app.shared.infrastructure.tf32_patch import Tf32Patch
from app.shared.infrastructure.warning_filter import PyannoteWarningFilter


class PyannotePipelineProvider:
    """Lazily-built, process-wide pool of pyannote diarization pipelines."""

    _instances: dict[int, object] = {}  # worker_id -> pyannote Pipeline
    _lock = threading.Lock()
    _log = logging.getLogger("app.shared.infrastructure.pyannote_pipeline_provider")

    @classmethod
    def get(cls, worker_id: int = 0) -> object:
        if worker_id not in cls._instances:
            with cls._lock:
                if worker_id not in cls._instances:
                    cls._instances[worker_id] = cls._load(worker_id)
        return cls._instances[worker_id]

    @classmethod
    def count(cls) -> int:
        """Number of loaded pipeline instances."""
        with cls._lock:
            return len(cls._instances)

    @classmethod
    def _load(cls, worker_id: int) -> object:
        import torch
        from pyannote.audio import Pipeline

        settings = ConfigService.get().load().diarization
        token = settings.hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "HuggingFace token required for pyannote.audio. "
                "Set 'hf_token' in diarization.yaml or the HF_TOKEN env var. "
                "Create a token at https://huggingface.co/settings/tokens and "
                "accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1"
            )

        # one-time runtime tuning (idempotent): TF32 re-enable + benign
        # warning suppression, applied before any inference happens
        Tf32Patch.apply(settings.tf32, settings.device)
        PyannoteWarningFilter.apply()

        cls._log.info("Loading pyannote pipeline '%s' on %s (worker %d) ...",
                      settings.pyannote_model, settings.device, worker_id)
        pipeline = Pipeline.from_pretrained(
            settings.pyannote_model,
            token=token,
            cache_dir="hf_cache/pyannote",
        )
        try:
            pipeline.to(torch.device(settings.device))
        except Exception as e:
            cls._log.warning("Failed to move pipeline to %s (%s); using CPU",
                             settings.device, e)

        # Native GPU batching: both attributes exist on the SpeakerDiarization
        # pipeline (a property with setter + a plain attribute respectively).
        # Guard with hasattr so alternative pipeline models still boot.
        for attr in ("segmentation_batch_size", "embedding_batch_size"):
            value = getattr(settings, attr, None)
            if value is not None and hasattr(pipeline, attr):
                setattr(pipeline, attr, value)

        # Server-wide hyperparameter defaults (e.g. tuned clustering
        # threshold), applied to every worker before any inference.
        if settings.pipeline_params:
            pipeline.instantiate(dict(settings.pipeline_params))
            cls._log.info("pipeline params overridden: %s",
                          settings.pipeline_params)
        cls._log.info(
            "pyannote pipeline ready on %s (worker %d, seg_batch=%s, emb_batch=%s).",
            settings.device, worker_id,
            getattr(pipeline, "segmentation_batch_size", "?"),
            getattr(pipeline, "embedding_batch_size", "?"),
        )
        return pipeline

    @classmethod
    def shutdown(cls) -> None:
        """Release all pipeline instances (GPU memory freed on process exit)."""
        with cls._lock:
            n = len(cls._instances)
            cls._instances.clear()
        if n:
            cls._log.info("released %d pyannote pipeline instance(s).", n)
