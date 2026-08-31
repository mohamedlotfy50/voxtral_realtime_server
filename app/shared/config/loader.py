"""Configuration loading + override service.

``ConfigService`` is the single source of truth for all three settings groups:

* :class:`DiarizationSettings` + :class:`ServerSettings` — read from
  ``diarization.yaml`` (shipped with the package; path overridable via the
  ``DIARIZATION_CONFIG`` env var).
* :class:`VoxtralSettings` — read from the root ``config.yaml`` (path
  overridable via the ``VOXTRAL_CONFIG`` env var). This file is **optional**:
  the diarization-only deployment path (``uvicorn app.server:app``) boots
  fine without it; vLLM launch is driven by ``serve_voxtral.py``.

The cached :class:`AppConfig` is built lazily on first access; calling
:meth:`configure` clears the cache so the next access rebuilds it with the
overrides applied. This keeps behavior compatible with the previous
``configure(**kw)`` global mutation pattern while the live instance stays
immutable.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml

from app.shared.config.app_config import AppConfig
from app.shared.config.diarization_settings import DiarizationSettings
from app.shared.config.server_settings import ServerSettings
from app.shared.config.voxtral_settings import VoxtralSettings


class ConfigService:
    """Loads and caches the application configuration."""

    _log = logging.getLogger("app.shared.config")
    _DEFAULT_DIARIZATION_YAML = Path(__file__).with_name("diarization.yaml")
    _DEFAULT_VOXTRAL_YAML = Path.cwd() / "config.yaml"

    _instance: Optional["ConfigService"] = None

    def __init__(self, diarization_yaml: Optional[Path] = None,
                 voxtral_yaml: Optional[Path] = None) -> None:
        self._diarization_yaml = diarization_yaml or self._DEFAULT_DIARIZATION_YAML
        self._voxtral_yaml = voxtral_yaml or self._DEFAULT_VOXTRAL_YAML
        self._overrides: dict[str, Any] = {}
        self._cache: Optional[AppConfig] = None

    # -- singleton access ------------------------------------------------- #
    @classmethod
    def get(cls) -> "ConfigService":
        if cls._instance is None:
            cls._instance = ConfigService()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear the global singleton (mainly for tests)."""
        cls._instance = None

    # -- public API -------------------------------------------------------- #
    def configure(self, **overrides: Any) -> None:
        """Apply overrides, clearing the cached config.

        ``overrides`` may contain any field from :class:`DiarizationSettings`,
        :class:`ServerSettings` or :class:`VoxtralSettings`. Unknown keys
        raise ``TypeError`` to surface typos early.
        """
        if overrides:
            self._log.debug("applying config overrides: %s", overrides)
            self._overrides.update(overrides)
            self._cache = None

    def load(self) -> AppConfig:
        """Return the cached config, building it on first call."""
        if self._cache is None:
            self._cache = self._build()
        return self._cache

    def settings(self) -> DiarizationSettings:
        return self.load().diarization

    def server(self) -> ServerSettings:
        return self.load().server

    def voxtral(self) -> VoxtralSettings:
        return self.load().voxtral

    # -- internals --------------------------------------------------------- #
    def _build(self) -> AppConfig:
        from dataclasses import fields
        dia_fields = {f.name: getattr(DiarizationSettings(), f.name)
                      for f in fields(DiarizationSettings)}
        srv_fields = {f.name: getattr(ServerSettings(), f.name)
                      for f in fields(ServerSettings)}
        vox_fields = {f.name: getattr(VoxtralSettings(), f.name)
                      for f in fields(VoxtralSettings)}

        # 1. yaml files (lowest precedence)
        dia_path = Path(os.environ.get("DIARIZATION_CONFIG", self._diarization_yaml))
        if dia_path.exists():
            with open(dia_path) as f:
                raw = yaml.safe_load(f) or {}
            dia_fields.update(raw.get("diarization", {}))
            srv_fields.update(raw.get("server", {}))

        vox_path = Path(os.environ.get("VOXTRAL_CONFIG", self._voxtral_yaml))
        if vox_path.exists():
            with open(vox_path) as f:
                raw = yaml.safe_load(f) or {}
            # config.yaml is flat (voxtral fields at top level)
            vox_fields.update({k: v for k, v in raw.items()
                               if k in vox_fields})

        # 2. explicit overrides (highest precedence)
        for key, value in self._overrides.items():
            if key in dia_fields:
                dia_fields[key] = value
            elif key in srv_fields:
                srv_fields[key] = value
            elif key in vox_fields:
                vox_fields[key] = value
            else:
                raise TypeError(f"unknown config key: {key!r}")

        return AppConfig(
            diarization=DiarizationSettings(**dia_fields),
            server=ServerSettings(**srv_fields),
            voxtral=VoxtralSettings(**vox_fields),
        )
