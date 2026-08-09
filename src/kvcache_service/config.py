"""Environment based application settings without framework lock-in."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .errors import BackendConfigurationError


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise BackendConfigurationError(f"{name} must be true or false, got {raw!r}")


def _int_env(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise BackendConfigurationError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise BackendConfigurationError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    backend: str = "transformers"
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    model_revision: str = "main"
    device: str = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = False
    local_files_only: bool = False
    store_dir: Path = Path("data/kv-cache")
    verify_checksum: bool = True
    max_context_tokens: int = 0
    default_chunk_size: int = 512
    max_new_tokens: int = 2048
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "info"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            backend=os.getenv("KVCACHE_BACKEND", "transformers"),
            model_id=os.getenv("KVCACHE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
            model_revision=os.getenv("KVCACHE_MODEL_REVISION", "main"),
            device=os.getenv("KVCACHE_DEVICE", "auto"),
            dtype=os.getenv("KVCACHE_DTYPE", "auto"),
            trust_remote_code=_bool_env("KVCACHE_TRUST_REMOTE_CODE", False),
            local_files_only=_bool_env("KVCACHE_LOCAL_FILES_ONLY", False),
            store_dir=Path(os.getenv("KVCACHE_STORE_DIR", "data/kv-cache")).expanduser(),
            verify_checksum=_bool_env("KVCACHE_VERIFY_CHECKSUM", True),
            max_context_tokens=_int_env("KVCACHE_MAX_CONTEXT_TOKENS", 0),
            default_chunk_size=_int_env("KVCACHE_DEFAULT_CHUNK_SIZE", 512, 1),
            max_new_tokens=_int_env("KVCACHE_MAX_NEW_TOKENS", 2048, 1),
            host=os.getenv("KVCACHE_HOST", "0.0.0.0"),
            port=_int_env("KVCACHE_PORT", 8080, 1),
            log_level=os.getenv("KVCACHE_LOG_LEVEL", "info"),
        )
