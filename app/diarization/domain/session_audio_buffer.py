"""Per-session audio accumulation buffer for pyannote pipeline input.

Reconstructs the full audio timeline (speech + silence) so the pyannote
pipeline can run its own VAD and segmentation on realistic audio.
"""
from __future__ import annotations

import numpy as np
import torch


class SessionAudioBuffer:
    """Accumulates PCM chunks placed at their timeline positions.

    ``add(pcm, start_s, end_s)`` writes the chunk at ``start_s`` in a growing
    internal buffer, zero-padding any gap. ``waveform()`` returns a
    ``(1, samples)`` float32 tensor suitable for ``pyannote.audio``.
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        self._sr = sample_rate
        self._buf = np.zeros(0, dtype=np.float32)

    def add(self, pcm: np.ndarray, start_s: float, end_s: float) -> None:
        start_sample = int(start_s * self._sr)
        end_sample = int(end_s * self._sr)
        if end_sample <= 0:
            return
        if end_sample > len(self._buf):
            self._buf = np.pad(
                self._buf, (0, end_sample - len(self._buf)),
                mode="constant",
            )
        n = min(len(pcm), end_sample - start_sample)
        if n > 0:
            self._buf[start_sample:start_sample + n] = pcm[:n].astype(np.float32)

    @property
    def duration_s(self) -> float:
        return len(self._buf) / self._sr

    def waveform(self) -> torch.Tensor:
        """Return ``(1, samples)`` tensor on CPU (pipeline moves it to GPU)."""
        if len(self._buf) == 0:
            return torch.zeros(1, 0, dtype=torch.float32)
        return torch.from_numpy(self._buf.copy()).float().unsqueeze(0)
