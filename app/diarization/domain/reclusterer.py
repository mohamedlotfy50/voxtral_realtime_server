"""Agglomerative consolidation of speaker clusters (cosine, average linkage)."""
from __future__ import annotations

import logging
from collections import Counter, defaultdict

import numpy as np

from app.diarization.domain.embedding_store import EmbeddingStore


class Reclusterer:
    """Consolidates all embeddings of a session.

    ``consolidate(store)`` clusters all embeddings with
    ``distance_threshold = 1 - merge_threshold`` (cosine >= merge_threshold
    merges), then maps old cluster ids -> new stable ids (first-appearance
    order) and updates ``store.labels`` in place.
    """

    _log = logging.getLogger("app.diarization.domain.reclusterer")

    def __init__(self, merge_threshold: float):
        self.merge_threshold = merge_threshold

    def consolidate(self, store: EmbeddingStore) -> dict[int, int]:
        """Return {old_label: new_label} mapping (may be identity/empty)."""
        n = len(store)
        if n < 2:
            return {}
        X = store.matrix()
        # L2 normalize (embeddings already normalized, but be safe)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xn = X / norms
        distance_threshold = max(1e-6, 1.0 - self.merge_threshold)
        try:
            from sklearn.cluster import AgglomerativeClustering
            ac = AgglomerativeClustering(
                metric="cosine", linkage="average",
                n_clusters=None, distance_threshold=distance_threshold,
            )
            raw = ac.fit_predict(Xn)
        except Exception as e:
            self._log.warning("agglomerative failed (%s); skipping", e)
            return {}
        # stable remap raw -> 0..K-1 by first appearance
        seen: dict[int, int] = {}
        new_raw = []
        for r in raw:
            if r not in seen:
                seen[r] = len(seen)
            new_raw.append(seen[r])
        old_labels = list(store.labels)
        # old_label -> new_label by majority vote of its embeddings
        votes: dict[int, Counter] = defaultdict(Counter)
        for o, nw in zip(old_labels, new_raw):
            votes[o][nw] += 1
        old_to_new = {o: votes[o].most_common(1)[0][0] for o in votes}
        # apply
        store.labels = [old_to_new[o] for o in old_labels]
        return old_to_new
