"""Per-session embedding store."""
from __future__ import annotations

from typing import Any

import numpy as np


class EmbeddingStore:
    """Holds every sentence embedding + meta for a session.

    `labels` always reflects the *current* (post-consolidation) cluster id
    of each embedding, contiguous 0..K-1 in first-appearance order.
    """

    def __init__(self):
        self.embeddings: list[np.ndarray] = []
        self.labels: list[int] = []
        self.meta: list[dict[str, Any]] = []  # {start_s, end_s, duration_s}

    def add(self, vec: np.ndarray, label: int, meta: dict[str, Any]) -> None:
        self.embeddings.append(vec.astype(np.float64))
        self.labels.append(label)
        self.meta.append(meta)

    def __len__(self) -> int:
        return len(self.embeddings)

    def matrix(self) -> np.ndarray:
        if not self.embeddings:
            return np.zeros((0, 192), dtype=np.float64)
        return np.stack(self.embeddings)
