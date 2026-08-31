"""Request schema for one audio item inside a /batch request."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class BatchItem(BaseModel):
    audio: str                  # base64 PCM16 @16k mono
    max_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    num_speakers: Optional[int] = None
    # Nested pipeline hyperparameter overrides for this item, e.g.
    # {"clustering": {"threshold": 0.65}} — applied before the run
    # (benchmark sweeps / experiments; sticky per worker until changed).
    params: Optional[dict[str, Any]] = None
