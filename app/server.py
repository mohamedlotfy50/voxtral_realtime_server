"""FastAPI application factory + lifespan for the diarization service.

All logic lives on :class:`AppFactory` as class/staticmethods — there are no
module-level functions. The lifespan eagerly loads the pyannote pipeline,
builds the service singletons, runs the idle-session sweeper, and shuts the
pipeline down cleanly on exit. The module-level ``app`` variable is the
uvicorn entrypoint (``uvicorn app.server:app``) — it is a value, not a method.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.diarization.controllers.diarization_controller import DiarizationController
from app.diarization.domain.pyannote_labeler import PyannoteLabeler
from app.diarization.repositories.session_repository import SessionRepository
from app.diarization.services.batch_service import BatchService
from app.diarization.services.label_service import LabelService
from app.diarization.services.session_service import SessionService
from app.shared.config.loader import ConfigService
from app.shared.infrastructure.diarization_engine_provider import DiarizationEngineProvider


class AppFactory:
    """Builds the diarization FastAPI application and owns its lifecycle."""

    _log = logging.getLogger("app.server")
    SWEEPER_INTERVAL_S = 60

    @staticmethod
    def _labeler_factory() -> PyannoteLabeler:
        settings = ConfigService.get().load().diarization
        return PyannoteLabeler(**settings.labeler_kwargs())

    @classmethod
    def _build_services(cls) -> tuple[SessionService, LabelService, BatchService]:
        repository = SessionRepository(cls._labeler_factory)
        settings = ConfigService.get().load().diarization
        session_service = SessionService(repository, settings.session_ttl_s)
        label_service = LabelService(session_service)
        batch_service = BatchService(DiarizationEngineProvider.get())
        return session_service, label_service, batch_service

    @staticmethod
    async def _sweeper(session_service: SessionService) -> None:
        while True:
            await asyncio.sleep(AppFactory.SWEEPER_INTERVAL_S)
            try:
                session_service.sweep()
            except Exception:  # noqa: BLE001 - keep sweeper alive
                AppFactory._log.exception("sweeper iteration failed")

    @staticmethod
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # startup: eagerly load engine (pipelines + workers), build services
        engine = DiarizationEngineProvider.get()
        session_service, label_service, batch_service = AppFactory._build_services()
        app.state.session_service = session_service
        app.state.label_service = label_service
        app.state.batch_service = batch_service
        app.state._sweeper_task = asyncio.create_task(AppFactory._sweeper(session_service))
        AppFactory._log.info(
            "diarization service ready (%s, engine=%s)",
            ConfigService.get().server(), engine.metrics(),
        )
        try:
            yield
        finally:
            # shutdown: cancel sweeper, stop engine + release pipelines
            task = getattr(app.state, "_sweeper_task", None)
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            DiarizationEngineProvider.shutdown()
            AppFactory._log.info("diarization service stopped")

    @classmethod
    def create(cls) -> FastAPI:
        """Build and return the configured FastAPI application."""
        app = FastAPI(title="Realtime Speaker Diarization", lifespan=cls.lifespan)
        app.include_router(DiarizationController.build_router())
        return app


# uvicorn entrypoint: `uvicorn app.server:app`
app = AppFactory.create()
