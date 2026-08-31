"""Cross-session diarization inference engine (background worker pool).

N worker threads, each owning a dedicated pyannote pipeline instance (see
:class:`PyannotePipelineProvider`), drain two priority queues:

* **realtime** — session ``label`` / ``consolidate`` runs, taken first;
* **offline** — full-file batch requests, taken only when realtime is idle.

``submit()`` returns a :class:`concurrent.futures.Future` resolved with the
pyannote ``Annotation`` (or the pipeline exception). Together with the
pipeline-native batch sizes (``segmentation_batch_size`` /
``embedding_batch_size``) this keeps the GPU continuously fed: batches fill
each forward pass and parallel workers overlap CUDA kernels across sessions.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future
from typing import Optional

import torch

from app.shared.infrastructure.engine_request import EngineRequest


class DiarizationEngine:
    """Worker pool executing pyannote pipeline runs for all sessions."""

    _log = logging.getLogger("app.shared.infrastructure.diarization_engine")
    _POLL_S = 0.5  # realtime-queue poll timeout (bounds shutdown latency)

    def __init__(self, pipelines: list) -> None:
        if not pipelines:
            raise ValueError("engine requires at least one pipeline")
        self._pipelines = pipelines
        self._realtime_q: queue.Queue[Optional[EngineRequest]] = queue.Queue()
        self._offline_q: queue.Queue[Optional[EngineRequest]] = queue.Queue()
        self._stop = threading.Event()
        # metrics
        self._lock = threading.Lock()
        self._runs = 0
        self._failures = 0
        self._total_run_s = 0.0
        self._last_run_s = 0.0
        self._last_run_t = 0.0
        self._workers = [
            threading.Thread(target=self._run, args=(i,),
                             name=f"diar-engine-{i}", daemon=True)
            for i in range(len(pipelines))
        ]
        for t in self._workers:
            t.start()
        self._log.info("engine started with %d worker(s)", len(self._workers))

    # -- public API -------------------------------------------------------- #

    def submit(self, waveform: torch.Tensor, sample_rate: int,
               max_speakers: Optional[int] = None,
               realtime: bool = True,
               min_speakers: Optional[int] = None,
               num_speakers: Optional[int] = None,
               params: Optional[dict] = None) -> Future:
        """Enqueue a pipeline run; realtime requests jump ahead of offline.

        ``params`` is an optional nested dict of pipeline hyperparameter
        overrides (e.g. ``{"clustering": {"threshold": 0.65}}``) applied to
        the worker's pipeline via ``pipeline.instantiate`` before the run —
        used by the batch benchmark to sweep configurations. Values are
        sticky on that worker until the next override; realtime sessions
        rely on the server-wide defaults from ``diarization.yaml``.
        """
        req = EngineRequest(
            waveform, sample_rate, max_speakers, Future(),
            min_speakers=min_speakers, num_speakers=num_speakers,
            params=params,
        )
        (self._realtime_q if realtime else self._offline_q).put(req)
        return req.future

    def stop(self) -> None:
        """Stop workers; fail any still-pending requests."""
        self._stop.set()
        for t in self._workers:
            t.join(timeout=self._POLL_S * 4)
        for q_ in (self._realtime_q, self._offline_q):
            while True:
                try:
                    req = q_.get_nowait()
                except queue.Empty:
                    break
                if req is not None and not req.future.done():
                    req.future.set_exception(RuntimeError("engine stopped"))
        self._log.info("engine stopped (%s)", self.metrics())

    def metrics(self) -> dict:
        with self._lock:
            avg = (self._total_run_s / self._runs) if self._runs else 0.0
            return {
                "workers": len(self._workers),
                "runs": self._runs,
                "failures": self._failures,
                "avg_run_s": round(avg, 3),
                "last_run_s": round(self._last_run_s, 3),
                "last_run_ago_s": round(time.monotonic() - self._last_run_t, 3)
                    if self._last_run_t else None,
                "realtime_queue_depth": self._realtime_q.qsize(),
                "offline_queue_depth": self._offline_q.qsize(),
            }

    # -- internals --------------------------------------------------------- #

    def _run(self, worker_id: int) -> None:
        pipeline = self._pipelines[worker_id]
        while not self._stop.is_set():
            req = self._next_request()
            if req is None:
                continue
            self._execute(pipeline, req)

    def _next_request(self) -> Optional[EngineRequest]:
        """Realtime first (blocking poll); offline only when realtime is idle."""
        try:
            return self._realtime_q.get(timeout=self._POLL_S)
        except queue.Empty:
            pass
        try:
            return self._offline_q.get_nowait()
        except queue.Empty:
            return None

    def _execute(self, pipeline, req: EngineRequest) -> None:
        kwargs: dict = {}
        if req.max_speakers is not None:
            kwargs["max_speakers"] = req.max_speakers
        if req.min_speakers is not None:
            kwargs["min_speakers"] = req.min_speakers
        if req.num_speakers is not None:
            kwargs["num_speakers"] = req.num_speakers
        if req.params:
            pipeline.instantiate(req.params)
        t0 = time.monotonic()
        try:
            output = pipeline(
                {"waveform": req.waveform, "sample_rate": req.sample_rate},
                **kwargs,
            )
            # speaker-diarization-3.1 returns an Annotation; some pipeline
            # variants wrap it — unwrap defensively.
            annotation = getattr(output, "speaker_diarization", output)
            req.future.set_result(annotation)
            elapsed = time.monotonic() - t0
        except Exception as e:
            req.future.set_exception(e)
            elapsed = time.monotonic() - t0
            self._log.exception("pipeline run failed after %.2fs: %s", elapsed, e)
        with self._lock:
            self._runs += 1
            self._failures += 0 if req.future.exception() is None else 1
            self._total_run_s += elapsed
            self._last_run_s = elapsed
            self._last_run_t = time.monotonic()
