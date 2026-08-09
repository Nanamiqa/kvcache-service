"""Template for a private inference service or cloud prompt-cache adapter.

Put real credentials in environment variables or a secret manager, never in cache metadata.
"""

from __future__ import annotations

from typing import Any, Dict, List

from kvcache_service.backend import KVCacheBackend
from kvcache_service.config import Settings
from kvcache_service.domain import (
    BuildCacheCommand,
    CacheInfo,
    CompletionCommand,
    CompletionResult,
)


class CustomBackend(KVCacheBackend):
    name = "custom"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Initialize an HTTP client, provider SDK, DB, or private inference client here.

    def health(self) -> Dict[str, Any]:
        return {"status": "ok", "backend": self.name, "model": self.settings.model_id}

    def build_cache(self, command: BuildCacheCommand) -> CacheInfo:
        # 1. Raw-KV engine: call its prefill/store operation.
        # 2. Cloud API: store a logical prefix and provider-side cache key.
        raise NotImplementedError("Map cache creation to the target provider")

    def get_cache(self, cache_id: str) -> CacheInfo:
        raise NotImplementedError("Read cache metadata from the target provider or DB")

    def list_caches(self) -> List[CacheInfo]:
        raise NotImplementedError("List provider or DB cache records")

    def delete_cache(self, cache_id: str) -> bool:
        raise NotImplementedError("Delete or invalidate the provider-side cache")

    def complete(self, command: CompletionCommand) -> CompletionResult:
        # Resolve command.cache_id, append command.prompt/input_ids, call the remote inference API,
        # and map provider usage (including cached tokens when available) into CompletionResult.
        raise NotImplementedError("Map completion to the target provider")


def create_backend(settings: Settings) -> KVCacheBackend:
    return CustomBackend(settings)
