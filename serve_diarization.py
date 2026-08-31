"""Entrypoint for the realtime speaker-diarization HTTP server.

Run with: ``python serve_diarization.py``  (or ``uvicorn app.server:app``)

Starts a uvicorn server hosting the FastAPI app built by
:class:`AppFactory`. Host/port/log level come from :class:`ServerSettings`
(``diarization.yaml``). All logic lives on :class:`DiarizationServerLauncher`;
there are no module-level functions.
"""
import logging

import uvicorn

from app.shared.config.loader import ConfigService

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")


class DiarizationServerLauncher:
    """Launches the diarization FastAPI server via uvicorn."""

    def run(self) -> None:
        server_cfg = ConfigService.get().server()
        config = uvicorn.Config(
            app="app.server:app",
            host=server_cfg.host,
            port=server_cfg.port,
            log_level=server_cfg.log_level,
        )
        print(f"Starting diarization server on {server_cfg.host}:{server_cfg.port}")
        uvicorn.Server(config).run()


if __name__ == "__main__":
    DiarizationServerLauncher().run()
