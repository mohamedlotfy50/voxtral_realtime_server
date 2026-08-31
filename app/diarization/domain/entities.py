"""Pure-domain entities for the diarization clustering core.

No imports from FastAPI, torch or speechbrain — only numpy + stdlib — so the
clustering algorithms stay independently unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Speaker:
    """A tracked speaker cluster with an L2-normalized centroid."""

    idx: int
    centroid: np.ndarray
    talk_time_s: float = 0.0
    utterances: int = 0
    last_active_s: float = -1.0
