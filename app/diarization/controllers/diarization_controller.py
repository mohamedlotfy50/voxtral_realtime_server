"""Diarization HTTP controller (class-based; routes are bound methods).

FastAPI does not ship class-based views, so :meth:`DiarizationController.build_router`
constructs an :class:`APIRouter` and registers the instance's bound methods as
route handlers. The handlers stay thin: decode request -> call service ->
return DTO. Dependencies are class-based (``SessionServiceDep`` /
``LabelServiceDep``) — no module-level functions.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.diarization.controllers.batch_req import BatchReq
from app.diarization.controllers.batch_service_dep import BatchServiceDep
from app.diarization.controllers.create_session_resp import CreateSessionResp
from app.diarization.controllers.delete_session_resp import DeleteSessionResp
from app.diarization.controllers.label_req import LabelReq
from app.diarization.controllers.label_service_dep import LabelServiceDep
from app.diarization.controllers.session_service_dep import SessionServiceDep
from app.shared.config.loader import ConfigService
from app.shared.infrastructure.diarization_engine_provider import DiarizationEngineProvider


class DiarizationController:
    """Thin controller exposing the diarization HTTP endpoints."""

    @classmethod
    def build_router(cls) -> APIRouter:
        """Build and return an :class:`APIRouter` with all routes bound."""
        router = APIRouter(prefix="/v1/diarization", tags=["diarization"])
        handler = cls()
        router.post("/sessions", response_model=CreateSessionResp)(handler.create_session)
        router.post("/sessions/{sid}/label")(handler.label)
        router.get("/sessions/{sid}")(handler.session_summary)
        router.post("/sessions/{sid}/consolidate")(handler.consolidate_session)
        router.delete("/sessions/{sid}", response_model=DeleteSessionResp)(handler.delete_session)
        router.post("/batch")(handler.batch)
        router.get("/health")(handler.health)
        return router

    async def create_session(self, dep: SessionServiceDep = Depends(SessionServiceDep)) -> dict:
        sid = dep.value.create()
        return {"session_id": sid}

    async def label(self, sid: str, req: LabelReq,
                    dep: LabelServiceDep = Depends(LabelServiceDep)) -> dict:
        return await dep.value.label(sid, req.audio, req.start_s, req.end_s,
                                     req.prev_speaker)

    async def session_summary(self, sid: str,
                              dep: SessionServiceDep = Depends(SessionServiceDep)) -> dict:
        s = dep.value.get(sid)
        return {"session_id": sid, "speakers": s.labeler.summary(),
                "num_speakers": len(set(
                    seg.label_idx for seg in s.labeler._segments.segments
                    if seg.label_idx >= 0))}

    async def consolidate_session(self, sid: str,
                                  dep: LabelServiceDep = Depends(LabelServiceDep)) -> dict:
        return dep.value.consolidate(sid)

    async def delete_session(self, sid: str,
                             dep: SessionServiceDep = Depends(SessionServiceDep)) -> dict:
        ok = dep.value.delete(sid)
        if not ok:
            raise HTTPException(status_code=404, detail="session not found")
        return {"deleted": sid}

    async def batch(self, req: BatchReq,
                    dep: BatchServiceDep = Depends(BatchServiceDep)) -> dict:
        """Offline one-shot diarization of multiple full audios."""
        return await dep.value.diarize(req.items)

    async def health(self, dep: SessionServiceDep = Depends(SessionServiceDep)) -> dict:
        return {
            "sessions": dep.value.count(),
            "engine": DiarizationEngineProvider.get().metrics(),
            "config": ConfigService.get().settings().public_view(),
        }
