"""Request schema for the /batch endpoint."""
from __future__ import annotations

from pydantic import BaseModel

from app.diarization.controllers.batch_item import BatchItem


class BatchReq(BaseModel):
    items: list[BatchItem]
