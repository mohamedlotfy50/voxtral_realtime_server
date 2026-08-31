"""Continuous-batching embedding worker (global, iteration-level scheduling)."""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from typing import Optional

import numpy as np

from app.shared.infrastructure.batcher_request import BatcherRequest
from app.shared.infrastructure.embedder import SpeechEmbedder


class EmbeddingBatcher:
    """Global continuously-batched embedding worker.

    submit() returns a future; the background worker thread loops, draining
    up to max_batch_size pending requests each iteration (with a short wait
    for the first request), running one batched ECAPA forward and resolving
    every request's future immediately. New arrivals mid-iteration are
    picked up next iteration — no batch barrier.
    """

    _log = logging.getLogger("app.shared.infrastructure.batcher")

    def __init__(self, embedder: SpeechEmbedder, max_batch_size: int,
                 max_wait_ms: int):
        self.embedder = embedder
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_ms / 1000.0
        self._q: queue.Queue[Optional[BatcherRequest]] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="embed-batcher",
                                        daemon=True)
        # metrics
        self._lock = threading.Lock()
        self._batches = 0
        self._total_reqs = 0
        self._total_fill = 0  # sum of batch sizes for avg fill ratio
        self._last_batch_size = 0
        self._last_batch_t = 0.0
        self._thread.start()

    @staticmethod
    def _resolve(req: BatcherRequest, emb: np.ndarray) -> None:
        """Resolve a request's asyncio.Future from a worker thread."""
        def _set(fut):
            if not fut.done():
                fut.set_result(emb)
        try:
            req.loop.call_soon_threadsafe(_set, req.future)
        except RuntimeError:
            # loop closed; drop silently
            pass

    def submit(self, pcm: np.ndarray) -> asyncio.Future:
        loop = asyncio.get_event_loop()
        req = BatcherRequest(pcm.astype(np.float32, copy=False), loop)
        self._q.put(req)
        return req.future

    def stop(self):
        self._stop.set()
        self._q.put(None)

    def _run(self):
        dev = self.embedder.device
        while not self._stop.is_set():
            try:
                first = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            if first is None or self._stop.is_set():
                break
            reqs = [first]
            # non-blocking drain up to max_batch_size
            while len(reqs) < self.max_batch_size:
                try:
                    r = self._q.get_nowait()
                except queue.Empty:
                    break
                if r is None:
                    break
                reqs.append(r)
            # optional short wait to grow the batch (continuous batching
            # still fires immediately if the queue is already populated)
            if len(reqs) < self.max_batch_size and self.max_wait_s > 0:
                deadline = time.monotonic() + self.max_wait_s
                while len(reqs) < self.max_batch_size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        r = self._q.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if r is None:
                        break
                    reqs.append(r)
            waveforms = [r.pcm for r in reqs]
            try:
                embs = self.embedder.encode_batch(waveforms)
                embs = embs.cpu().numpy() if hasattr(embs, "cpu") else np.asarray(embs)
            except Exception as e:
                self._log.exception("encode_batch failed: %s", e)
                embs = np.zeros((len(reqs), self.embedder.dim), dtype=np.float32)
            for r, e in zip(reqs, embs):
                self._resolve(r, e)
            with self._lock:
                self._batches += 1
                self._total_reqs += len(reqs)
                self._total_fill += len(reqs)
                self._last_batch_size = len(reqs)
                self._last_batch_t = time.monotonic()

    def metrics(self) -> dict:
        with self._lock:
            avg_fill = (self._total_fill / self._batches) if self._batches else 0.0
            return {
                "batches": self._batches,
                "total_requests": self._total_reqs,
                "avg_batch_size": round(avg_fill, 2),
                "last_batch_size": self._last_batch_size,
                "last_batch_ago_s": round(time.monotonic() - self._last_batch_t, 3)
                    if self._last_batch_t else None,
                "queue_depth": self._q.qsize(),
                "max_batch_size": self.max_batch_size,
            }
