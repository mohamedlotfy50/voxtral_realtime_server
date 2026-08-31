"""vLLM ``RealtimeConnection`` patch emitting token_ids in delta events.

Moved verbatim from the former root ``voxtral_patch.py``. Kept isolated under
``app.voxtral`` so the diarization app (``app.server``) never imports vLLM.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from uuid import uuid4

from fastapi import WebSocket
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.serve.utils.api_utils import sanitize_message
from vllm.entrypoints.speech_to_text.realtime import connection as conn_mod
from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection
from vllm.entrypoints.speech_to_text.realtime.protocol import (
    TranscriptionDelta,
    TranscriptionDone,
)
from vllm.entrypoints.speech_to_text.realtime.serving import OpenAIServingRealtime
from vllm.sampling_params import RequestOutputKind, SamplingParams


class RealtimeMonkyPatch(RealtimeConnection):
    def __init__(self, websocket: WebSocket, serving: OpenAIServingRealtime) -> None:
        super().__init__(websocket, serving)

    async def _run_generation(
        self,
        streaming_input_gen: AsyncGenerator,
        input_stream: asyncio.Queue[list[int]],
    ) -> None:
        request_id = f"rt-{self.connection_id}-{uuid4()}"
        full_text = ""
        prompt_token_ids_len = 0
        completion_tokens_len = 0

        try:

            sampling_params = SamplingParams.from_optional(
                temperature=0.0,
                max_tokens=self.serving.model_cls.realtime_max_tokens,
                output_kind=RequestOutputKind.DELTA,
                skip_clone=True,
            )

            result_gen = self.serving.engine_client.generate(
                prompt=streaming_input_gen,
                sampling_params=sampling_params,
                request_id=request_id,
            )

            async for output in result_gen:
                if output.outputs and len(output.outputs) > 0:
                    if not prompt_token_ids_len and output.prompt_token_ids:
                        prompt_token_ids_len = len(output.prompt_token_ids)

                    delta = output.outputs[0].text
                    token_ids = list(output.outputs[0].token_ids)
                    full_text += delta

                    input_stream.put_nowait(token_ids)

                    # KEY CHANGE: include token_ids in the delta event
                    await self.send(
                        TranscriptionDelta(delta=delta, token_ids=token_ids)
                    )

                    completion_tokens_len += len(token_ids)

                if not self._is_connected:
                    break

            usage = UsageInfo(
                prompt_tokens=prompt_token_ids_len,
                completion_tokens=completion_tokens_len,
                total_tokens=prompt_token_ids_len + completion_tokens_len,
            )

            await self.send(TranscriptionDone(text=full_text, usage=usage))

            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

        except Exception as e:
            conn_mod.logger.exception("Error in generation: %s", e)
            await self.send_error(
                sanitize_message(str(e)), "processing_error"
            )
