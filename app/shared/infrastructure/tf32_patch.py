"""Runtime patch re-enabling TensorFloat-32 (TF32) for pyannote inference.

pyannote disables TF32 at the start of *every* pipeline call
(``fix_reproducibility`` in ``pyannote.audio.utils.reproducibility``,
triggered from ``Pipeline.__call__`` and ``Inference`` — see
pyannote-audio#1370) at the cost of matmul/cudnn throughput on Ampere+
GPUs. Since diarization here prioritizes throughput, this patch no-ops the
two module-level ``fix_reproducibility`` references (``pipeline`` and
``inference`` hold their own imported copies) and sets the TF32 flags on.

Controlled by the ``tf32`` setting (``true`` = patch applied, ``false`` =
pyannote's default accuracy-first behavior is preserved). Idempotent:
applying it more than once is a no-op.
"""
from __future__ import annotations

import logging


class Tf32Patch:
    """Re-enables TF32 and neutralizes pyannote's per-call disabler."""

    _log = logging.getLogger("app.shared.infrastructure.tf32_patch")
    _applied = False

    @classmethod
    def apply(cls, enabled: bool, device: str) -> None:
        """Apply the patch once; safe (and cheap) to call per worker load."""
        if not enabled:
            cls._log.info("tf32 disabled by config; keeping pyannote defaults")
            return
        if "cuda" not in device.lower():
            cls._log.info("tf32 patch skipped for non-CUDA device %r", device)
            return
        if cls._applied:
            return

        import torch
        import pyannote.audio.core.inference as pyannote_inference
        import pyannote.audio.core.pipeline as pyannote_pipeline

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Both modules bound their own reference to fix_reproducibility at
        # import time, so each must be replaced individually.
        pyannote_pipeline.fix_reproducibility = cls._no_fix
        pyannote_inference.fix_reproducibility = cls._no_fix
        cls._applied = True
        cls._log.info(
            "TF32 re-enabled for %s (pyannote per-call disabler neutralized)",
            device,
        )

    @staticmethod
    def _no_fix(device) -> None:
        """No-op replacement for ``fix_reproducibility`` — keeps TF32 on."""
        return None
