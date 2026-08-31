"""Request schema for the /label endpoint."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class LabelReq(BaseModel):
    audio: str           # base64 PCM16 @16k mono
    start_s: float
    end_s: float
    prev_speaker: Optional[int] = None
