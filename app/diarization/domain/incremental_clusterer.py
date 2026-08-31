"""Stateful per-session speaker clusterer.

Two layers:
  1. Fast per-sentence assignment (stick / switch / recency / new).
  2. Periodic agglomerative consolidation that merges spurious clusters
     and emits a relabel mapping the client applies to prior segments.

``assign()`` returns the (consolidated) label for this utterance plus an
optional ``relabel`` event so the client can rewrite previously printed
speakers live.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import numpy as np

from app.diarization.domain.embedding_store import EmbeddingStore
from app.diarization.domain.entities import Speaker
from app.diarization.domain.reclusterer import Reclusterer


class IncrementalClusterer:
    def __init__(self, stick_threshold: float, switch_threshold: float,
                 min_segment_s: float, recency_window_s: float,
                 centroid_ema: float, merge_threshold: float,
                 recluster_every: int):
        self.stick_threshold = stick_threshold
        self.switch_threshold = switch_threshold
        self.min_segment_s = min_segment_s
        self.recency_window_s = recency_window_s
        self.ema = centroid_ema
        self.recluster_every = max(1, recluster_every)
        self.store = EmbeddingStore()
        self.reclusterer = Reclusterer(merge_threshold)
        self.speakers: list[Speaker] = []
        # maps any previously-emitted idx -> current internal idx (so the
        # client's prev_speaker hint stays valid across consolidations)
        self.idx_remap: dict[int, int] = {}
        self._since_consolidate = 0

    def _cos(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _rebuild_speakers(self) -> None:
        """Reconstruct speakers list from the store (post-consolidation)."""
        if not self.store:
            self.speakers = []
            return
        labels = self.store.labels
        by_label: dict[int, list[int]] = defaultdict(list)
        for i, lab in enumerate(labels):
            by_label[lab].append(i)
        X = self.store.matrix()
        self.speakers = []
        for lab in sorted(by_label):
            idxs = by_label[lab]
            embs = X[idxs]
            centroid = embs.mean(axis=0)
            nn = np.linalg.norm(centroid)
            centroid = centroid / nn if nn > 0 else centroid
            talk = float(sum(self.store.meta[i]["duration_s"] for i in idxs))
            utt = len(idxs)
            last = float(max(self.store.meta[i]["end_s"] for i in idxs))
            self.speakers.append(Speaker(idx=lab, centroid=centroid,
                                         talk_time_s=talk, utterances=utt,
                                         last_active_s=last))

    def _consolidate(self) -> dict[int, int]:
        """Run agglomerative consolidation. Returns {old_idx: new_idx}."""
        old_to_new = self.reclusterer.consolidate(self.store)
        if not old_to_new:
            return {}
        # compose idx_remap: ancient -> current_old -> new
        new_remap: dict[int, int] = {}
        for k, v in self.idx_remap.items():
            new_remap[k] = old_to_new.get(v, v)
        for o, nw in old_to_new.items():
            new_remap[o] = nw
        self.idx_remap = new_remap
        self._rebuild_speakers()
        return old_to_new

    def assign(self, vec: np.ndarray, duration_s: float, end_s: float,
               start_s: float, prev_speaker: Optional[int]) -> dict[str, Any]:
        vec = vec.astype(np.float64)
        n = np.linalg.norm(vec)
        if n > 0:
            vec = vec / n

        # map client's prev_speaker hint through idx_remap
        prev = None
        if prev_speaker is not None:
            prev = self.idx_remap.get(prev_speaker, prev_speaker)

        assigned: Optional[int] = None
        sim: Optional[float] = None
        method = "new"

        # 1. stickiness
        if prev is not None and 0 <= prev < len(self.speakers):
            s_prev = self._cos(vec, self.speakers[prev].centroid)
            if s_prev >= self.stick_threshold:
                assigned = prev
                sim = s_prev
                method = "stick"

        # 2. switch
        if assigned is None and self.speakers:
            sims = [self._cos(vec, sp.centroid) for sp in self.speakers]
            best = int(np.argmax(sims))
            if sims[best] >= self.switch_threshold:
                assigned = best
                sim = sims[best]
                method = "switch"

        # 3. recency fallback for short sentences
        if assigned is None and duration_s < self.min_segment_s and self.speakers:
            recent = [sp for sp in self.speakers
                      if sp.last_active_s >= 0
                      and (end_s - sp.last_active_s) <= self.recency_window_s]
            if recent:
                recent.sort(key=lambda sp: sp.last_active_s, reverse=True)
                assigned = recent[0].idx
                sim = None
                method = "recency"

        # 4. create new speaker
        if assigned is None:
            sp = Speaker(idx=len(self.speakers), centroid=vec.copy())
            self.speakers.append(sp)
            assigned = sp.idx
            sim = None
            method = "new"

        # store before consolidation (consolidation uses the full set)
        self.store.add(vec, assigned,
                       {"start_s": start_s, "end_s": end_s,
                        "duration_s": duration_s})

        self._since_consolidate += 1
        relabel = None
        consolidated = False
        if self._since_consolidate >= self.recluster_every:
            old_to_new = self._consolidate()
            self._since_consolidate = 0
            if old_to_new:
                consolidated = True
                assigned = old_to_new.get(assigned, assigned)
                relabel = {
                    "mapping": {f"SPEAKER_{k:02d}": f"SPEAKER_{v:02d}"
                                for k, v in old_to_new.items()},
                    "num_speakers": len(self.speakers),
                    "consolidation": len(self.store),
                }
                # stats already rebuilt from store; skip per-utterance update
                return {
                    "speaker_idx": assigned,
                    "speaker": f"SPEAKER_{assigned:02d}",
                    "sim": (round(float(sim), 4) if sim is not None else None),
                    "num_speakers": len(self.speakers),
                    "method": method,
                    "consolidated": True,
                    "relabel": relabel,
                }

        # update centroid (EMA) + stats (non-consolidation path)
        sp = self.speakers[assigned]
        a = self.ema
        new_c = (1 - a) * sp.centroid + a * vec
        nn = np.linalg.norm(new_c)
        sp.centroid = new_c / nn if nn > 0 else new_c
        sp.talk_time_s += duration_s
        sp.utterances += 1
        sp.last_active_s = end_s

        return {
            "speaker_idx": assigned,
            "speaker": f"SPEAKER_{assigned:02d}",
            "sim": (round(float(sim), 4) if sim is not None else None),
            "num_speakers": len(self.speakers),
            "method": method,
            "consolidated": consolidated,
            "relabel": relabel,
        }

    def summary(self) -> list[dict]:
        return [
            {"id": f"SPEAKER_{sp.idx:02d}",
             "talk_time_s": round(sp.talk_time_s, 3),
             "utterances": sp.utterances}
            for sp in self.speakers
        ]
