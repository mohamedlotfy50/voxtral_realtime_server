"""Suppresses known-benign pyannote runtime warnings.

The ``std(): degrees of freedom is <= 0`` UserWarning from
``pyannote.audio.models.blocks.pooling`` fires on very short or degenerate
audio regions (e.g. trailing silence at segment borders). It does not
affect results — the warning is silenced so server logs stay readable.
Applied once at pipeline load; idempotent.
"""
from __future__ import annotations

import logging
import warnings


class PyannoteWarningFilter:
    """Targeted warnings filter for benign pyannote internals."""

    _log = logging.getLogger("app.shared.infrastructure.warning_filter")
    _applied = False

    _BENIGN_PATTERNS = (
        # stats pooling on very short/degenerate sequences — no effect on output
        r"std\(\): degrees of freedom is <= 0\..*",
    )

    @classmethod
    def apply(cls) -> None:
        if cls._applied:
            return
        for pattern in cls._BENIGN_PATTERNS:
            warnings.filterwarnings("ignore", message=pattern)
        cls._applied = True
        cls._log.info("benign pyannote warnings suppressed: %s",
                      [p[:40] for p in cls._BENIGN_PATTERNS])
