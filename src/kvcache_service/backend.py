"""Backend protocol and loader for local, private, or future cloud providers."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .config import Settings
from .domain import BuildCacheCommand, CacheInfo, CompletionCommand, CompletionResult
from .errors import BackendConfigurationError


class KVCacheBackend(ABC):
    """Stable boundary between the HTTP API and an inference/cache implementation."""

    name = "unknown"

    @abstractmethod
    def health(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def build_cache(self, command: BuildCacheCommand) -> CacheInfo:
        raise NotImplementedError

    @abstractmethod
    def get_cache(self, cache_id: str) -> CacheInfo:
        raise NotImplementedError

    @abstractmethod
    def list_caches(self) -> List[CacheInfo]:
        raise NotImplementedError

    @abstractmethod
    def delete_cache(self, cache_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def complete(self, command: CompletionCommand) -> CompletionResult:
        raise NotImplementedError


def load_backend(settings: Settings) -> KVCacheBackend:
    """Load the built-in backend or a user supplied ``module:factory`` plugin."""

    if settings.backend == "transformers":
        from .transformers_backend import TransformersBackend

        return TransformersBackend(settings)

    if ":" not in settings.backend:
        raise BackendConfigurationError(
            "KVCACHE_BACKEND must be 'transformers' or a 'python.module:factory' reference"
        )

    module_name, factory_name = settings.backend.rsplit(":", 1)
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, factory_name)
        backend = factory(settings)
    except (ImportError, AttributeError, TypeError) as exc:
        raise BackendConfigurationError(
            f"Cannot load backend factory {settings.backend!r}: {exc}"
        ) from exc
    if not isinstance(backend, KVCacheBackend):
        raise BackendConfigurationError(
            f"Backend factory {settings.backend!r} must return KVCacheBackend"
        )
    return backend
