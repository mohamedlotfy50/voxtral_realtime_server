"""Label use-case: decode PCM -> labeler.assign -> label response.

Encapsulates the per-utterance pipeline that used to live inline in the
``label`` FastAPI route, so the controller stays a thin adapter.
"""
from __future__ import annotations

import asyncio

from fastapi import HTTPException

from app.diarization.services.session_service import SessionService
from app.shared.infrastructure.audio_decoder import AudioDecoder
from app.shared.infrastructure.embedder import SpeechEmbedder


class LabelService:
    def __init__(self, sessions: SessionService):
        self._sessions = sessions

    async def label(self, sid: str, audio_b64: str, start_s: float,
                    end_s: float, prev_speaker: int | None) -> dict:
        session = self._sessions.get(sid)
        pcm = AudioDecoder.pcm16_b64_to_float(audio_b64)
        if pcm.shape[0] < int(SpeechEmbedder.SAMPLE_RATE * 0.1):
            raise HTTPException(status_code=400, detail="audio too short (<0.1s)")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            session.labeler.assign,
            pcm, start_s, end_s, prev_speaker,
        )

    def consolidate(self, sid: str) -> dict:
        """Force a final pipeline pass on all accumulated audio.

        Returns the relabel mapping (old->new) so the client can rewrite any
        prior segments emitted since the last pipeline run. Use at end of
        stream to absorb the tail (< pipeline_every_n) utterances.
        """
        session = self._sessions.get(sid)
        return session.labeler.consolidate()
