"""Environment based application settings without framework lock-in."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

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


def _float_env(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise BackendConfigurationError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise BackendConfigurationError(f"{name} must be >= {minimum}, got {value}")
    return value


def _csv_env(name: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    values = tuple(item.strip().rstrip("/") for item in raw.split(",") if item.strip())
    if not values:
        raise BackendConfigurationError(f"{name} must contain at least one value")
    return values


def _api_keys_env(name: str) -> Dict[str, str]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BackendConfigurationError(f"{name} must be a JSON object") from exc
    if not isinstance(payload, dict) or not payload:
        raise BackendConfigurationError(f"{name} must be a non-empty JSON object")
    normalized = {str(tenant).strip(): str(key) for tenant, key in payload.items()}
    if any(not tenant or not key for tenant, key in normalized.items()):
        raise BackendConfigurationError(f"{name} tenant names and keys must not be empty")
    return normalized


def _optional_env(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return value.strip()


@dataclass(frozen=True)
class Settings:
    backend: str = "transformers"
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    model_revision: str = "main"
    device: str = "auto"
    dtype: str = "auto"
    model_fingerprint: Optional[str] = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    store_dir: Path = Path("data/kv-cache")
    verify_checksum: bool = True
    cache_ttl_seconds: int = 0
    max_store_bytes: int = 0
    max_context_tokens: int = 0
    default_chunk_size: int = 512
    max_new_tokens: int = 2048
    api_key: Optional[str] = field(default=None, repr=False)
    api_keys: Dict[str, str] = field(default_factory=dict, repr=False)
    admin_api_key: Optional[str] = field(default=None, repr=False)
    tenant_header: str = "X-Tenant-ID"
    request_timeout_seconds: float = 600.0
    admission_queue_timeout_seconds: float = 5.0
    shutdown_grace_seconds: float = 30.0
    max_concurrent_requests: int = 256
    max_concurrent_per_tenant: int = 32
    rate_limit_tokens_per_minute: int = 0
    rate_limit_burst_tokens: int = 0
    metrics_enabled: bool = True
    vllm_endpoints: Tuple[str, ...] = ("http://127.0.0.1:8000",)
    vllm_api_key: Optional[str] = field(default=None, repr=False)
    vllm_timeout_seconds: float = 600.0
    vllm_max_retries: int = 1
    vllm_circuit_breaker_failures: int = 3
    vllm_circuit_breaker_cooldown_seconds: float = 15.0
    vllm_affinity_weight: float = 10.0
    lmcache_controller_url: Optional[str] = None
    redis_url: Optional[str] = field(default=None, repr=False)
    redis_key_prefix: str = "kvcache"
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
            model_fingerprint=_optional_env("KVCACHE_MODEL_FINGERPRINT"),
            trust_remote_code=_bool_env("KVCACHE_TRUST_REMOTE_CODE", False),
            local_files_only=_bool_env("KVCACHE_LOCAL_FILES_ONLY", False),
            store_dir=Path(os.getenv("KVCACHE_STORE_DIR", "data/kv-cache")).expanduser(),
            verify_checksum=_bool_env("KVCACHE_VERIFY_CHECKSUM", True),
            cache_ttl_seconds=_int_env("KVCACHE_CACHE_TTL_SECONDS", 0),
            max_store_bytes=_int_env("KVCACHE_MAX_STORE_BYTES", 0),
            max_context_tokens=_int_env("KVCACHE_MAX_CONTEXT_TOKENS", 0),
            default_chunk_size=_int_env("KVCACHE_DEFAULT_CHUNK_SIZE", 512, 1),
            max_new_tokens=_int_env("KVCACHE_MAX_NEW_TOKENS", 2048, 1),
            api_key=_optional_env("KVCACHE_API_KEY"),
            api_keys=_api_keys_env("KVCACHE_API_KEYS"),
            admin_api_key=_optional_env("KVCACHE_ADMIN_API_KEY"),
            tenant_header=os.getenv("KVCACHE_TENANT_HEADER", "X-Tenant-ID").strip(),
            request_timeout_seconds=_float_env("KVCACHE_REQUEST_TIMEOUT_SECONDS", 600.0, 0.1),
            admission_queue_timeout_seconds=_float_env(
                "KVCACHE_ADMISSION_QUEUE_TIMEOUT_SECONDS", 5.0, 0.1
            ),
            shutdown_grace_seconds=_float_env("KVCACHE_SHUTDOWN_GRACE_SECONDS", 30.0),
            max_concurrent_requests=_int_env("KVCACHE_MAX_CONCURRENT_REQUESTS", 256, 1),
            max_concurrent_per_tenant=_int_env("KVCACHE_MAX_CONCURRENT_PER_TENANT", 32, 1),
            rate_limit_tokens_per_minute=_int_env("KVCACHE_RATE_LIMIT_TOKENS_PER_MINUTE", 0),
            rate_limit_burst_tokens=_int_env("KVCACHE_RATE_LIMIT_BURST_TOKENS", 0),
            metrics_enabled=_bool_env("KVCACHE_METRICS_ENABLED", True),
            vllm_endpoints=_csv_env("KVCACHE_VLLM_ENDPOINTS", ("http://127.0.0.1:8000",)),
            vllm_api_key=_optional_env("KVCACHE_VLLM_API_KEY"),
            vllm_timeout_seconds=_float_env("KVCACHE_VLLM_TIMEOUT_SECONDS", 600.0, 0.1),
            vllm_max_retries=_int_env("KVCACHE_VLLM_MAX_RETRIES", 1),
            vllm_circuit_breaker_failures=_int_env("KVCACHE_VLLM_CIRCUIT_BREAKER_FAILURES", 3, 1),
            vllm_circuit_breaker_cooldown_seconds=_float_env(
                "KVCACHE_VLLM_CIRCUIT_BREAKER_COOLDOWN_SECONDS", 15.0
            ),
            vllm_affinity_weight=_float_env("KVCACHE_VLLM_AFFINITY_WEIGHT", 10.0),
            lmcache_controller_url=_optional_env("KVCACHE_LMCACHE_CONTROLLER_URL"),
            redis_url=_optional_env("KVCACHE_REDIS_URL"),
            redis_key_prefix=os.getenv("KVCACHE_REDIS_KEY_PREFIX", "kvcache").strip(),
            host=os.getenv("KVCACHE_HOST", "0.0.0.0"),
            port=_int_env("KVCACHE_PORT", 8080, 1),
            log_level=os.getenv("KVCACHE_LOG_LEVEL", "info"),
        )
