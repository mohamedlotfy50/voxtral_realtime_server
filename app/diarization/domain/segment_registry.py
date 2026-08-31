"""Per-session segment registry tracking all labeled segments.

Each :class:`SegmentRecord` maps a time range ``[start_s, end_s]`` to a
speaker label index. Segments start out ``confirmed=False`` (pending): they
carry a borrowed temporary label until the next pipeline pass assigns their
real label.

``update_labels()`` applies a new label assignment and separates the
changes into two protocol pieces returned to the client:

* ``relabel`` — ``{old_idx: new_idx}`` for CONFIRMED labels that moved.
  The no-split matching rule in :class:`PyannoteLabeler` guarantees each
  old label maps to exactly one new label, so clients can safely apply it
  as a bulk rename over all stored segments.
* ``patch`` — per-segment corrections for newly confirmed segments. A bulk
  rename cannot express these (the same borrowed temporary label may
  belong to different real speakers once confirmed), so the exact segments
  are listed with their final labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SegmentRecord:
    """A single diarized utterance time range and its current label."""

    start_s: float
    end_s: float
    label_idx: int = -1   # -1 = not yet labeled
    confirmed: bool = False  # True once a pipeline pass assigned this label


class SegmentRegistry:
    """Ordered list of segments with label-update + relabel generation."""

    def __init__(self) -> None:
        self._segments: list[SegmentRecord] = []

    def add(self, start_s: float, end_s: float) -> SegmentRecord:
        seg = SegmentRecord(start_s, end_s)
        self._segments.append(seg)
        return seg

    def update_labels(
        self, new_labels: list[int]
    ) -> tuple[dict[int, int], list[dict]]:
        """Apply ``new_labels`` (one per segment, in order).

        Returns:
          relabel: ``{old_idx: new_idx}`` for confirmed labels that moved
            (safe as a bulk rename; each old idx appears once).
          patch: ``[{start_s, end_s, speaker_idx}, ...]`` for segments that
            were pending and just got confirmed (per-segment corrections).
        """
        relabel: dict[int, int] = {}
        patch: list[dict] = []
        for i, new in enumerate(new_labels):
            seg = self._segments[i]
            old = seg.label_idx
            if not seg.confirmed:
                if new >= 0:
                    patch.append({"start_s": seg.start_s,
                                  "end_s": seg.end_s,
                                  "speaker_idx": new})
                    seg.confirmed = True
            elif old >= 0 and old != new:
                if old not in relabel:
                    relabel[old] = new
                elif relabel[old] != new:
                    # No-split invariant violated upstream; keep the first
                    # mapping so the client protocol stays consistent.
                    pass
            seg.label_idx = new
        return relabel, patch

    @property
    def segments(self) -> list[SegmentRecord]:
        return self._segments

    def __len__(self) -> int:
        return len(self._segments)