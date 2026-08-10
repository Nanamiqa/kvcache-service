"""Backend protocol and loader for local, private, or future cloud providers."""

from __future__ import annotations

import asyncio
import importlib
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

from .config import Settings
from .domain import (
    BuildCacheCommand,
    CacheInfo,
    CompletionChunk,
    CompletionCommand,
    CompletionResult,
)
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
    def get_cache(self, cache_id: str, tenant_id: str = "default") -> CacheInfo:
        raise NotImplementedError

    @abstractmethod
    def list_caches(self, tenant_id: Optional[str] = None) -> List[CacheInfo]:
        raise NotImplementedError

    @abstractmethod
    def delete_cache(self, cache_id: str, tenant_id: str = "default") -> bool:
        raise NotImplementedError

    def cache_stats(self, tenant_id: Optional[str] = None) -> Dict[str, int]:
        """Return backend cache storage counters when supported."""

        return {
            "cache_count": 0,
            "tensor_bytes": 0,
            "disk_bytes": 0,
            "max_store_bytes": 0,
            "ttl_seconds": 0,
        }

    def prune_caches(self, tenant_id: Optional[str] = None) -> Dict[str, int]:
        """Remove expired/over-quota entries when supported."""

        return {"removed_count": 0, "freed_bytes": 0}

    @abstractmethod
    def complete(self, command: CompletionCommand) -> CompletionResult:
        raise NotImplementedError

    async def ahealth(self) -> Dict[str, Any]:
        return await asyncio.to_thread(self.health)

    async def abuild_cache(self, command: BuildCacheCommand) -> CacheInfo:
        return await asyncio.to_thread(self.build_cache, command)

    async def aget_cache(self, cache_id: str, tenant_id: str = "default") -> CacheInfo:
        return await asyncio.to_thread(self.get_cache, cache_id, tenant_id)

    async def alist_caches(self, tenant_id: Optional[str] = None) -> List[CacheInfo]:
        return await asyncio.to_thread(self.list_caches, tenant_id)

    async def adelete_cache(self, cache_id: str, tenant_id: str = "default") -> bool:
        return await asyncio.to_thread(self.delete_cache, cache_id, tenant_id)

    async def acache_stats(self, tenant_id: Optional[str] = None) -> Dict[str, int]:
        return await asyncio.to_thread(self.cache_stats, tenant_id)

    async def aprune_caches(self, tenant_id: Optional[str] = None) -> Dict[str, int]:
        return await asyncio.to_thread(self.prune_caches, tenant_id)

    async def acomplete(self, command: CompletionCommand) -> CompletionResult:
        return await asyncio.to_thread(self.complete, command)

    async def astream_complete(self, command: CompletionCommand) -> AsyncIterator[CompletionChunk]:
        """Stream completion events; synchronous backends emit one terminal event."""

        result = await self.acomplete(command)
        yield CompletionChunk(
            text=result.text,
            model=result.model,
            token_ids=result.token_ids,
            finish_reason=result.finish_reason,
            prompt_tokens=result.prompt_tokens,
            cached_tokens=result.cached_tokens,
            completion_tokens=(
                result.completion_tokens
                if result.completion_tokens is not None
                else len(result.token_ids)
            ),
            timings_ms=result.timings_ms,
        )

    async def aclose(self) -> None:
        """Release backend resources during graceful shutdown."""

        return None


def load_backend(settings: Settings) -> KVCacheBackend:
    """Load the built-in backend or a user supplied ``module:factory`` plugin."""

    if settings.backend == "transformers":
        from .transformers_backend import TransformersBackend

        return TransformersBackend(settings)

    if settings.backend == "vllm":
        from .vllm_backend import VLLMBackend

        return VLLMBackend(settings)

    if ":" not in settings.backend:
        raise BackendConfigurationError(
            "KVCACHE_BACKEND must be 'transformers', 'vllm', or a 'python.module:factory' reference"
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
