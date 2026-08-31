"""Audio decoding helper (base64 PCM16 -> float32)."""
from __future__ import annotations

import base64

import numpy as np


class AudioDecoder:
    """Stateless decoder for base64-encoded int16 PCM (@16k mono) buffers."""

    @staticmethod
    def pcm16_b64_to_float(b64str: str) -> np.ndarray:
        """Decode to float32 in [-1, 1)."""
        raw = base64.b64decode(b64str)
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
