"""Batch use-case: decode items -> submit offline engine runs -> collect turns.

Full-file one-shot diarization without session state. Items are submitted to
the :class:`DiarizationEngine` at **offline priority** so they never delay
live realtime sessions, and all futures are awaited concurrently. Per-item
failures (bad base64, too short, pipeline error) are isolated into that
item's ``error`` field instead of failing the whole request.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import Future
from typing import Optional, Union

import torch

from app.diarization.controllers.batch_item import BatchItem
from app.diarization.controllers.batch_resp import BatchItemResp
from app.shared.config.loader import ConfigService
from app.shared.infrastructure.audio_decoder import AudioDecoder
from app.shared.infrastructure.diarization_engine import DiarizationEngine


class BatchService:
    def __init__(self, engine: DiarizationEngine):
        self._engine = engine
        self._sample_rate = ConfigService.get().settings().sample_rate
        self._min_samples = int(self._sample_rate * 0.1)

    async def diarize(self, items: list[BatchItem]) -> dict:
        """Diarize every item concurrently; per-item failures are isolated."""
        entries: list[Union[Future, str]] = [self._submit(i) for i in items]
        futures = [e for e in entries if not isinstance(e, str)]
        loop = asyncio.get_event_loop()
        outcomes = await loop.run_in_executor(None, self._collect, futures) \
            if futures else []
        outcome_iter = iter(outcomes)
        results = [
            self._to_resp(e) if isinstance(e, str) else self._to_resp(next(outcome_iter))
            for e in entries
        ]
        return {"results": [r.model_dump() for r in results]}

    # -- internals --------------------------------------------------------- #

    def _submit(self, item: BatchItem) -> Union[Future, str]:
        """Decode + enqueue one offline run; a str is a per-item error."""
        try:
            pcm = AudioDecoder.pcm16_b64_to_float(item.audio)
        except Exception as e:
            return f"invalid base64 audio: {e}"
        if pcm.shape[0] < self._min_samples:
            return "audio too short (<0.1s)"
        waveform = torch.from_numpy(pcm.copy()).float().unsqueeze(0)
        return self._engine.submit(
            waveform, self._sample_rate, item.max_speakers, realtime=False,
            min_speakers=item.min_speakers, num_speakers=item.num_speakers,
            params=item.params,
        )

    @staticmethod
    def _collect(futures: list[Future]) -> list:
        outcomes = []
        for f in futures:
            try:
                outcomes.append(f.result())
            except Exception as e:
                outcomes.append(e)
        return outcomes

    @staticmethod
    def _to_resp(outcome: Union[str, Exception, object]) -> BatchItemResp:
        if isinstance(outcome, (str, Exception)):
            return BatchItemResp(error=str(outcome))
        turns = [
            {"start_s": round(float(turn.start), 3),
             "end_s": round(float(turn.end), 3),
             "speaker": str(label)}
            for turn, _, label in outcome.itertracks(yield_label=True)
        ]
        return BatchItemResp(
            turns=turns,
            num_speakers=len({t["speaker"] for t in turns}),
        )
