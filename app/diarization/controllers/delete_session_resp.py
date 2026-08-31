"""Response schema for session deletion."""
from __future__ import annotations

from pydantic import BaseModel


class DeleteSessionResp(BaseModel):
    deleted: str
