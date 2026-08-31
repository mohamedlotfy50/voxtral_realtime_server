"""Response schema for session creation."""
from __future__ import annotations

from pydantic import BaseModel


class CreateSessionResp(BaseModel):
    session_id: str
