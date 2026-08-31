"""Frozen configuration for the vLLM Voxtral realtime STT server.

Mirrors the fields of the root ``config.yaml``. All fields are optional
(``None`` defaults) so the diarization-only deployment path still boots when
``config.yaml`` is absent — vLLM launch is driven by ``main.py``, not by the
diarization app lifespan.

Note: deliberately does NOT use ``from __future__ import annotations`` so
that ``ClassVar`` is detected as a real type (not a string) by the dataclass
machinery on Python 3.10, keeping ``_VALUE_ARGS`` / ``_FLAG_ARGS`` out of
``__dataclass_fields__``.
"""
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Optional


@dataclass(frozen=True)
class VoxtralSettings:
    """vLLM serve arguments for the Voxtral realtime model.

    ``build_argv()`` expands the settings into the CLI argument vector
    consumed by ``vllm serve``. Unknown override keys raise ``TypeError``.
    """

    model_path: Optional[str] = None
    served_model_name: Optional[str] = None
    api_key: Optional[str] = None
    host: Optional[str] = None
    port: Optional[str] = None
    trust_remote_code: Optional[bool] = None
    enforce_eager: Optional[bool] = None
    tensor_parallel_size: Optional[str] = None
    max_model_len: Optional[str] = None
    max_num_batched_tokens: Optional[str] = None
    max_num_seqs: Optional[str] = None
    gpu_memory_utilization: Optional[str] = None
    attention_backend: Optional[str] = None
    compilation_config: Optional[str] = None

    # (attr, cli flag) pairs for value arguments
    _VALUE_ARGS: ClassVar[tuple] = (
        ("served_model_name", "--served-model-name"),
        ("api_key", "--api-key"),
        ("host", "--host"),
        ("port", "--port"),
        ("tensor_parallel_size", "--tensor-parallel-size"),
        ("max_model_len", "--max-model-len"),
        ("max_num_batched_tokens", "--max-num-batched-tokens"),
        ("max_num_seqs", "--max-num-seqs"),
        ("gpu_memory_utilization", "--gpu-memory-utilization"),
        ("attention_backend", "--attention-backend"),
        ("compilation_config", "--compilation-config"),
    )

    # (attr, cli flag) pairs for boolean flag arguments
    _FLAG_ARGS: ClassVar[tuple] = (
        ("trust_remote_code", "--trust-remote-code"),
        ("enforce_eager", "--enforce-eager"),
    )

    def with_overrides(self, **overrides: Any) -> "VoxtralSettings":
        """Return a new frozen settings with the given fields replaced."""
        unknown = set(overrides) - set(self.__dataclass_fields__)
        if unknown:
            raise TypeError(f"unknown voxtral settings: {sorted(unknown)}")
        return replace(self, **overrides)

    def build_argv(self) -> list[str]:
        """Expand into the ``vllm serve`` CLI argument vector."""
        argv = [
            "vllm", "serve", self.model_path,
            "--tokenizer-mode", "mistral",
            "--config-format", "mistral",
            "--load-format", "mistral",
        ]
        for attr, flag in self._VALUE_ARGS:
            value = getattr(self, attr)
            if value is not None:
                argv += [flag, str(value)]
        for attr, flag in self._FLAG_ARGS:
            if getattr(self, attr):
                argv += [flag]
        return argv
