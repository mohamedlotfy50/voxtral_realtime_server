"""Request object enqueued to the embedding batcher worker."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


class BatcherRequest:
    """A single embedding request awaiting batched inference.

    Carries the PCM payload + the asyncio future the caller awaits, bound to
    the event loop that submitted it so the worker thread can resolve it
    thread-safely via ``loop.call_soon_threadsafe``.
    """

    __slots__ = ("pcm", "future", "loop")

    def __init__(self, pcm: np.ndarray, loop: asyncio.AbstractEventLoop):
        self.pcm = pcm
        self.future: asyncio.Future = loop.create_future()
        self.loop = loop
