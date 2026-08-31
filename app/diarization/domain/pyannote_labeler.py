"""Stateful per-session speaker labeler backed by the pyannote.audio pipeline.

Replaces :class:`IncrementalClusterer` with the same response contract
(``assign`` / ``consolidate`` / ``summary``) so the controller and client need
no changes.

The pipeline runs on the **full accumulated audio** for each session every
``pipeline_every_n`` segments (throttled). Between runs, segments keep their
last-known label (or ``prev_speaker`` for new segments). The final
``consolidate()`` forces a full pass to absorb the tail.

Label stability protocol (see :class:`SegmentRegistry.update_labels`):

* Speaker IDs are **stable** across pipeline passes: a raw pyannote label
  matched to a confirmed internal label keeps that label's ID; brand-new
  speakers get fresh, never-reused IDs. There is no per-pass renumbering.
* **No-split rule**: every confirmed label maps to exactly one target label
  per pass (majority vote over its segments). Renames are therefore safe
  for the client to apply as a bulk ``{old: new}`` mapping.
* Newly confirmed (previously pending) segments are corrected per-segment
  via the ``segments`` patch in the relabel event — a bulk rename cannot
  express those corrections.
"""
from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from typing import Any, Optional

import numpy as np

from app.diarization.domain.segment_registry import SegmentRegistry
from app.diarization.domain.session_audio_buffer import SessionAudioBuffer
from app.shared.infrastructure.diarization_engine_provider import DiarizationEngineProvider


class PyannoteLabeler:
    """Per-session labeler wrapping the global pyannote pipeline."""

    _log = logging.getLogger("app.diarization.domain.pyannote_labeler")

    def __init__(self, sample_rate: int, pipeline_every_n: int,
                 pipeline_min_audio_s: float,
                 max_speakers: Optional[int]) -> None:
        self._sr = sample_rate
        self._pipeline_every_n = max(1, pipeline_every_n)
        self._pipeline_min_audio_s = pipeline_min_audio_s
        self._max_speakers = max_speakers
        self._buffer = SessionAudioBuffer(sample_rate)
        self._segments = SegmentRegistry()
        self._since_pipeline = 0
        self._idx_remap: dict[int, int] = {}
        self._num_speakers = 0
        self._next_id = 0  # next fresh stable speaker ID (never reused)

    # -- public API (same contract as IncrementalClusterer) ---------------- #

    def assign(self, pcm: np.ndarray, start_s: float, end_s: float,
               prev_speaker: Optional[int]) -> dict[str, Any]:
        self._buffer.add(pcm, start_s, end_s)
        self._segments.add(start_s, end_s)
        self._since_pipeline += 1

        should_run = (
            self._since_pipeline >= self._pipeline_every_n
            and self._buffer.duration_s >= self._pipeline_min_audio_s
        )

        if should_run:
            return self._run_pipeline()

        # Throttled: assign temporary label from prev_speaker or default 0
        prev = None
        if prev_speaker is not None:
            prev = self._idx_remap.get(prev_speaker, prev_speaker)
        temp_label = prev if prev is not None else 0
        self._segments.segments[-1].label_idx = temp_label
        if temp_label + 1 > self._num_speakers:
            self._num_speakers = temp_label + 1

        return {
            "speaker_idx": temp_label,
            "speaker": f"SPEAKER_{temp_label:02d}",
            "sim": None,
            "num_speakers": self._num_speakers,
            "method": "pending",
            "consolidated": False,
            "relabel": None,
        }

    def consolidate(self) -> dict[str, Any]:
        """Force a final pipeline pass on all accumulated audio."""
        if self._buffer.duration_s < 0.1:
            return {"relabel": None,
                    "num_speakers": self._num_speakers,
                    "speakers": self.summary()}
        result = self._run_pipeline(force=True)
        return {
            "relabel": result.get("relabel"),
            "num_speakers": self._num_speakers,
            "speakers": self.summary(),
        }

    def summary(self) -> list[dict]:
        by_spk: dict[int, dict] = defaultdict(
            lambda: {"talk_time_s": 0.0, "utterances": 0})
        for seg in self._segments.segments:
            if seg.label_idx >= 0:
                by_spk[seg.label_idx]["talk_time_s"] += seg.end_s - seg.start_s
                by_spk[seg.label_idx]["utterances"] += 1
        return [
            {"id": f"SPEAKER_{idx:02d}",
             "talk_time_s": round(d["talk_time_s"], 3),
             "utterances": d["utterances"]}
            for idx, d in sorted(by_spk.items())
        ]

    # -- internals --------------------------------------------------------- #

    def _run_pipeline(self, force: bool = False) -> dict[str, Any]:
        waveform = self._buffer.waveform()
        if waveform.shape[1] < int(self._sr * 0.1):
            return self._pending_result()

        t0 = time.monotonic()
        try:
            annotation = DiarizationEngineProvider.get().submit(
                waveform, self._sr, self._max_speakers, realtime=True,
            ).result()
        except Exception as e:
            self._log.exception("pyannote pipeline failed: %s", e)
            return self._pending_result()

        elapsed = time.monotonic() - t0
        self._log.debug("pipeline ran in %.2fs on %.1fs audio (%d turns)",
                        elapsed, self._buffer.duration_s,
                        len(list(annotation.itertracks())))

        old_labels = [seg.label_idx for seg in self._segments.segments]
        confirmed = [seg.confirmed for seg in self._segments.segments]
        raw_labels = [
            self._dominant_speaker(annotation, seg.start_s, seg.end_s)
            for seg in self._segments.segments
        ]
        new_labels = self._match_and_remap(old_labels, raw_labels, confirmed)

        relabel_map, segment_patch = self._segments.update_labels(new_labels)
        self._since_pipeline = 0
        self._num_speakers = len(set(
            seg.label_idx for seg in self._segments.segments
            if seg.label_idx >= 0))

        # Update idx_remap so client's prev_speaker stays valid
        for old, new in relabel_map.items():
            for k, v in list(self._idx_remap.items()):
                if v == old:
                    self._idx_remap[k] = new
            self._idx_remap[old] = new

        current_label = self._segments.segments[-1].label_idx
        if current_label < 0:
            current_label = 0
        client_relabel = None
        if relabel_map or segment_patch:
            client_relabel = {
                "mapping": {f"SPEAKER_{k:02d}": f"SPEAKER_{v:02d}"
                            for k, v in relabel_map.items()},
                "segments": [
                    {"start_s": p["start_s"], "end_s": p["end_s"],
                     "speaker": f"SPEAKER_{p['speaker_idx']:02d}"}
                    for p in segment_patch
                ],
                "num_speakers": self._num_speakers,
                "consolidation": len(self._segments),
            }

        return {
            "speaker_idx": current_label,
            "speaker": f"SPEAKER_{current_label:02d}",
            "sim": None,
            "num_speakers": self._num_speakers,
            "method": "consolidate" if force else "pipeline",
            "consolidated": True,
            "relabel": client_relabel,
        }

    def _pending_result(self) -> dict[str, Any]:
        label = self._segments.segments[-1].label_idx
        if label < 0:
            label = 0
        return {
            "speaker_idx": label,
            "speaker": f"SPEAKER_{label:02d}",
            "sim": None,
            "num_speakers": self._num_speakers,
            "method": "pending",
            "consolidated": False,
            "relabel": None,
        }

    def _dominant_speaker(self, annotation, start_s: float,
                          end_s: float) -> Any:
        """Return the pyannote speaker label with the most speech in [start, end]."""
        from pyannote.core import Segment as PyannoteSegment
        seg = PyannoteSegment(start=start_s, end=end_s)
        cropped = annotation.crop(seg)
        durations: dict[Any, float] = defaultdict(float)
        for turn, _, label in cropped.itertracks(yield_label=True):
            clip_start = max(turn.start, start_s)
            clip_end = min(turn.end, end_s)
            durations[label] += max(0.0, clip_end - clip_start)
        if not durations:
            return None
        return max(durations, key=durations.get)

    def _match_and_remap(self, old_labels: list[int],
                         raw_labels: list,
                         confirmed: list[bool]) -> list[int]:
        """Map this pass's raw pyannote labels to stable internal labels.

        Guarantees the no-split invariant: every confirmed old label maps to
        exactly one target label per pass, so the client's bulk rename
        ``{old: new}`` can never merge two distinct speakers.

        1. Raw pyannote label strings -> first-appearance ints.
        2. Votes: each confirmed segment votes (old_label, raw) — pending
           segments don't vote (their old label is a borrowed temporary).
        3. Greedy 1-1 raw<->old matching by vote strength.
        4. Matched raws keep their old stable ID; unseen raws get fresh,
           never-reused IDs.
        5. Confirmed segments take their OLD label's matched target (the
           whole label moves together); pending segments take their own
           raw's ID (recency fallback when the raw is None/silence).
        """
        # 1. First-appearance remap of raw pyannote label strings
        seen: dict[Any, int] = {}
        fa: list[int] = []
        for label in raw_labels:
            if label is None:
                fa.append(-1)
                continue
            if label not in seen:
                seen[label] = len(seen)
            fa.append(seen[label])

        # 2. Votes from confirmed segments only
        votes: dict[int, Counter] = defaultdict(Counter)
        for old, f, conf in zip(old_labels, fa, confirmed):
            if conf and old >= 0 and f >= 0:
                votes[old][f] += 1

        # 3. Greedy 1-1 matching, strongest vote first (deterministic ties)
        candidates = sorted(
            ((cnt, old, f)
             for old, counter in votes.items()
             for f, cnt in counter.items()),
            key=lambda x: (-x[0], x[1], x[2]),
        )
        old_to_raw: dict[int, int] = {}
        used_raws: set[int] = set()
        for _, old, f in candidates:
            if old not in old_to_raw and f not in used_raws:
                old_to_raw[old] = f
                used_raws.add(f)

        # 4. Raw -> stable internal ID (matched raws keep the old ID;
        #    unmatched raws get fresh IDs)
        raw_to_id: dict[int, int] = {f: old for old, f in old_to_raw.items()}
        for f in sorted(set(fa)):
            if f >= 0 and f not in raw_to_id:
                raw_to_id[f] = self._next_id
                self._next_id += 1

        # 5. Assign new labels
        result: list[int] = []
        last_valid = 0
        for old, f, conf in zip(old_labels, fa, confirmed):
            if conf:
                # Whole-label move: confirmed segments follow their old
                # label's matched target (keeps the mapping a function).
                if old in old_to_raw:
                    label = raw_to_id[old_to_raw[old]]
                else:
                    # Old label has no matched raw this pass: keep it.
                    label = old
            else:
                # Pending segment: take its own raw's label directly.
                label = raw_to_id[f] if f >= 0 else last_valid
            result.append(label)
            if label >= 0:
                last_valid = label
        return result
