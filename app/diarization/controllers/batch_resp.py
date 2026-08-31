"""Response schema for one item of a /batch response."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class BatchItemResp(BaseModel):
    turns: list[dict] = []          # [{start_s, end_s, speaker}]
    num_speakers: int = 0
    error: Optional[str] = None     # set if this item failed
