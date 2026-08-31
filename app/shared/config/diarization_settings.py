"""Frozen diarization tuning parameters."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional


@dataclass(frozen=True)
class DiarizationSettings:
    """Tunable parameters for the pyannote.audio diarization pipeline.

    Instances are immutable; use ``replace()`` or :meth:`with_overrides` to
    produce a modified copy (the ``configure()`` override hook does this).
    """

    # pyannote pipeline
    device: str = "cuda:1"
    hf_token: Optional[str] = None
    pyannote_model: str = "pyannote/speaker-diarization-3.1"
    sample_rate: int = 16000
    pipeline_every_n: int = 2
    pipeline_min_audio_s: float = 2.0
    max_speakers: Optional[int] = None

    # GPU batching
    segmentation_batch_size: int = 32   # windows per segmentation forward pass
    embedding_batch_size: int = 32      # crops per embedding forward pass
    engine_workers: int = 2             # parallel pipeline workers on the device
    tf32: bool = True                   # re-enable TF32 matmuls (throughput > accuracy)

    # Server-wide pipeline hyperparameter defaults applied to every worker
    # at load time, e.g. {"clustering": {"threshold": 0.65,
    # "min_cluster_size": 6}}. Per-request ``params`` (batch items) override.
    pipeline_params: Optional[dict[str, Any]] = None

    # session lifecycle
    session_ttl_s: int = 3600

    def with_overrides(self, **overrides: Any) -> "DiarizationSettings":
        """Return a new frozen settings with the given fields replaced."""
        unknown = set(overrides) - set(self.__dataclass_fields__)
        if unknown:
            raise TypeError(f"unknown diarization settings: {sorted(unknown)}")
        return replace(self, **overrides)

    def labeler_kwargs(self) -> dict:
        """Keyword args for :class:`PyannoteLabeler`."""
        return dict(
            sample_rate=self.sample_rate,
            pipeline_every_n=self.pipeline_every_n,
            pipeline_min_audio_s=self.pipeline_min_audio_s,
            max_speakers=self.max_speakers,
        )

    def public_view(self) -> dict:
        """Subset exposed by the /health endpoint."""
        return {k: getattr(self, k) for k in (
            "device", "hf_token", "pyannote_model", "sample_rate",
            "pipeline_every_n", "pipeline_min_audio_s", "max_speakers",
            "segmentation_batch_size", "embedding_batch_size", "engine_workers",
            "tf32", "pipeline_params", "session_ttl_s",
        )}
