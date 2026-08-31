"""ECAPA speech embedder (loaded once on the configured device)."""
from __future__ import annotations

import logging

import numpy as np
import torch


class SpeechEmbedder:
    """Wraps speechbrain ECAPA encoder. Batched forward with rel-lengths."""

    _log = logging.getLogger("app.shared.infrastructure.embedder")
    SAMPLE_RATE = 16000

    def __init__(self, device: str):
        from speechbrain.inference.speaker import EncoderClassifier
        self.device = device
        self._log.info("Loading ECAPA spkrec-ecapa-voxceleb on %s ...", device)
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="hf_cache/ecapa",
            run_opts={"device": device},
        )
        self.dim = 192  # ECAPA spkrec-ecapa-voxceleb output dim
        self._log.info("ECAPA ready (embedding dim=%s).", self.dim)

    @torch.no_grad()
    def encode_batch(self, waveforms: list[np.ndarray]) -> torch.Tensor:
        """Encode a list of variable-length 1D float PCM arrays.

        Returns: (N, dim) tensor on `device`, L2-normalized.
        """
        if not waveforms:
            return torch.zeros(0, self.dim, device=self.device)
        lens = [int(w.shape[0]) for w in waveforms]
        maxlen = max(lens)
        batch = torch.zeros(len(waveforms), maxlen, device=self.device, dtype=torch.float32)
        rel = torch.zeros(len(waveforms), device=self.device, dtype=torch.float32)
        for i, w in enumerate(waveforms):
            n = min(len(w), maxlen)
            batch[i, :n] = torch.from_numpy(w[:n]).float()
            rel[i] = lens[i] / maxlen
        emb = self.model.encode_batch(batch, rel)  # (N, 1, dim)
        emb = emb.squeeze(1)                       # (N, dim)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return emb
