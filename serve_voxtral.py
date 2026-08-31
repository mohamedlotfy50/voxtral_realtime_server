"""Entrypoint for the vLLM Voxtral realtime STT server.

Run with: ``python serve_voxtral.py``

Applies the :class:`RealtimeMonkyPatch` to vLLM's realtime connection/router
modules so transcription delta events carry ``token_ids``, builds the serve
argv from :class:`VoxtralSettings` (root ``config.yaml``) and launches vLLM
in-process. All logic lives on :class:`VoxtralServerLauncher`; there are no
module-level functions.
"""
import logging
import os
import sys

from vllm.entrypoints.cli.main import main
from vllm.entrypoints.speech_to_text.realtime import api_router as router_mod
from vllm.entrypoints.speech_to_text.realtime import connection as conn_mod

from app.shared.config.loader import ConfigService
from app.voxtral.realtime_patch import RealtimeMonkyPatch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")


class VoxtralServerLauncher:
    """Launches the patched vLLM Voxtral realtime server in-process."""

    def __init__(self) -> None:
        # env defaults — kept here (after vLLM import) to match prior behavior
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
        os.environ.setdefault("VLLM_USE_MODELSCOPE", "False")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    def run(self) -> None:
        conn_mod.RealtimeConnection = RealtimeMonkyPatch
        router_mod.RealtimeConnection = RealtimeMonkyPatch
        argv = ConfigService.get().voxtral().build_argv()
        sys.argv = argv
        print("Running (in-process with token_ids patch):", " ".join(argv))
        main()


if __name__ == "__main__":
    VoxtralServerLauncher().run()
