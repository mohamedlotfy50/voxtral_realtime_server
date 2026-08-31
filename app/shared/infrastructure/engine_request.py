"""Request object enqueued to the diarization engine workers."""
from __future__ import annotations

from concurrent.futures import Future
from typing import Any, Optional

import torch


class EngineRequest:
    """A single pipeline run awaiting execution by an engine worker.

    Carries the full-audio waveform, its sample rate, the optional speaker
    count hints (``max_speakers`` / ``min_speakers`` / ``num_speakers``),
    optional per-request pipeline hyperparameter overrides (``params``)
    and the :class:`concurrent.futures.Future` the caller blocks on. The
    worker resolves it with a pyannote ``Annotation`` (or the exception
    raised by the pipeline).
    """

    __slots__ = ("waveform", "sample_rate", "max_speakers", "min_speakers",
                 "num_speakers", "params", "future")

    def __init__(self, waveform: torch.Tensor, sample_rate: int,
                 max_speakers: Optional[int], future: Future,
                 min_speakers: Optional[int] = None,
                 num_speakers: Optional[int] = None,
                 params: Optional[dict[str, Any]] = None) -> None:
        self.waveform = waveform
        self.sample_rate = sample_rate
        self.max_speakers = max_speakers
        self.min_speakers = min_speakers
        self.num_speakers = num_speakers
        self.params = params
        self.future: Future = future
