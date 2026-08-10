"""Asynchronous, cache-affine gateway backend for one or more vLLM replicas."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple, Union

import httpx

from .backend import KVCacheBackend
from .config import Settings
from .domain import (
    BuildCacheCommand,
    CacheInfo,
    CompletionChunk,
    CompletionCommand,
    CompletionResult,
)
from .errors import (
    BackendUnavailableError,
    CacheCompatibilityError,
    CacheExpiredError,
    CacheNotFoundError,
    KVCacheError,
    UpstreamAPIError,
)
from .logical_store import LogicalCacheRecord, LogicalCacheStore, PrefixValue
from .metrics import (
    BACKEND_DURATION,
    BACKEND_REQUESTS,
    CACHE_OPERATIONS,
    CACHE_TOKENS,
    COMPLETION_TOKENS,
    LMCACHE_HEALTHY,
)
from .router import ReplicaRouter, ReplicaState


class VLLMBackend(KVCacheBackend):
    """Production data-plane adapter; vLLM/LMCache own the physical KV blocks."""

    name = "vllm"

    def __init__(
        self,
        settings: Settings,
        *,
        client: Optional[httpx.AsyncClient] = None,
        store: Optional[LogicalCacheStore] = None,
    ) -> None:
        self.settings = settings
        self.router = ReplicaRouter(
            settings.vllm_endpoints,
            failure_threshold=settings.vllm_circuit_breaker_failures,
            cooldown_seconds=settings.vllm_circuit_breaker_cooldown_seconds,
            affinity_weight=settings.vllm_affinity_weight,
        )
        headers = {"User-Agent": "kvcache-service/0.3"}
        if settings.vllm_api_key:
            headers["Authorization"] = f"Bearer {settings.vllm_api_key}"
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.vllm_timeout_seconds),
            headers=headers,
            limits=httpx.Limits(max_connections=1024, max_keepalive_connections=256),
        )
        self._owns_client = client is None
        if store is not None:
            self.store = store
        elif settings.redis_url:
            from .redis_store import RedisLogicalCacheStore

            self.store = RedisLogicalCacheStore(
                settings.redis_url,
                key_prefix=settings.redis_key_prefix,
                ttl_seconds=settings.cache_ttl_seconds,
            )
        else:
            self.store = LogicalCacheStore(
                settings.store_dir / "logical-cache.sqlite3",
                ttl_seconds=settings.cache_ttl_seconds,
            )
        identity = settings.model_fingerprint or f"{settings.model_id}@{settings.model_revision}"
        self._model_fingerprint = hashlib.sha256(identity.encode("utf-8")).hexdigest()

    @staticmethod
    def _sync_error() -> KVCacheError:
        return KVCacheError("The vLLM backend must be used through its asynchronous API")

    def health(self) -> Dict[str, Any]:
        return {
            "status": "configured",
            "backend": self.name,
            "model": self.settings.model_id,
            "replica_count": len(self.router.endpoints),
            "lmcache_controller": self.settings.lmcache_controller_url,
        }

    def build_cache(self, command: BuildCacheCommand) -> CacheInfo:
        del command
        raise self._sync_error()

    def get_cache(self, cache_id: str, tenant_id: str = "default") -> CacheInfo:
        return self.store.get(cache_id, tenant_id).info

    def list_caches(self, tenant_id: Optional[str] = None) -> List[CacheInfo]:
        return self.store.list(tenant_id)

    def delete_cache(self, cache_id: str, tenant_id: str = "default") -> bool:
        return self.store.delete(cache_id, tenant_id)

    def cache_stats(self, tenant_id: Optional[str] = None) -> Dict[str, int]:
        return self.store.stats(tenant_id)

    def prune_caches(self, tenant_id: Optional[str] = None) -> Dict[str, int]:
        del tenant_id
        return self.store.prune()

    def complete(self, command: CompletionCommand) -> CompletionResult:
        del command
        raise self._sync_error()

    async def ahealth(self) -> Dict[str, Any]:
        async def probe(endpoint: str) -> bool:
            try:
                response = await self.client.get(f"{endpoint}/health", timeout=5.0)
                response.raise_for_status()
            except Exception as exc:
                await self.router.record_probe(endpoint, False, str(exc)[:500])
                return False
            await self.router.record_probe(endpoint, True)
            return True

        results = await asyncio.gather(*(probe(endpoint) for endpoint in self.router.endpoints))
        healthy = sum(results)
        lmcache: Dict[str, Any] = {"configured": False, "healthy": None}
        if self.settings.lmcache_controller_url:
            lmcache = {"configured": True, "healthy": False}
            try:
                response = await self.client.get(
                    f"{self.settings.lmcache_controller_url.rstrip('/')}/healthcheck",
                    headers={"Authorization": ""},
                    timeout=5.0,
                )
                response.raise_for_status()
                lmcache = {"configured": True, "healthy": True}
                LMCACHE_HEALTHY.set(1)
            except Exception as exc:
                lmcache["error"] = str(exc)[:500]
                LMCACHE_HEALTHY.set(0)
        status = "ok" if healthy and lmcache.get("healthy") is not False else "unavailable"
        return {
            "status": status,
            "backend": self.name,
            "model": self.settings.model_id,
            "healthy_replicas": healthy,
            "replica_count": len(results),
            "replicas": await self.router.snapshot(),
            "lmcache": lmcache,
        }

    @staticmethod
    def _prefix_value(command: BuildCacheCommand) -> PrefixValue:
        if command.text is not None:
            return command.text
        if command.input_ids is not None:
            return [int(value) for value in command.input_ids]
        raise KVCacheError("Exactly one of text or input_ids is required")

    def _cache_identity(self, tenant_id: str, prefix: PrefixValue) -> Tuple[str, str]:
        encoded = json.dumps(prefix, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        prefix_sha = hashlib.sha256(encoded).hexdigest()
        digest = hashlib.sha256()
        for value in (tenant_id, self._model_fingerprint, prefix_sha):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        return prefix_sha, digest.hexdigest()

    @staticmethod
    def _upstream_error(response: httpx.Response) -> UpstreamAPIError:
        try:
            payload = response.json()
            error = payload.get("error", payload)
            message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        except (ValueError, TypeError):
            message = response.text[:1000] or f"HTTP {response.status_code}"
        status = response.status_code if 400 <= response.status_code < 500 else 502
        return UpstreamAPIError(f"vLLM rejected the request: {message}", status)

    async def _post_json(
        self,
        payload: Dict[str, Any],
        *,
        affinity_key: str,
        operation: str,
        request_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str]:
        excluded: Set[str] = set()
        attempts = min(len(self.router.endpoints), self.settings.vllm_max_retries + 1)
        last_error: Optional[Exception] = None
        for _ in range(attempts):
            state = await self.router.select(affinity_key, excluded)
            excluded.add(state.endpoint)
            started = time.perf_counter()
            headers = {"X-Request-ID": request_id} if request_id else None
            try:
                async with self.router.lease(state):
                    response = await self.client.post(
                        f"{state.endpoint}/v1/completions",
                        json=payload,
                        headers=headers,
                    )
                if response.status_code >= 400:
                    error = self._upstream_error(response)
                    if response.status_code < 500:
                        BACKEND_REQUESTS.labels(
                            endpoint=state.endpoint, operation=operation, outcome="rejected"
                        ).inc()
                        raise error
                    await self.router.record_failure(state, error)
                    raise BackendUnavailableError(str(error))
                data = response.json()
                BACKEND_REQUESTS.labels(
                    endpoint=state.endpoint, operation=operation, outcome="success"
                ).inc()
                BACKEND_DURATION.labels(endpoint=state.endpoint, operation=operation).observe(
                    time.perf_counter() - started
                )
                return data, state.endpoint
            except UpstreamAPIError:
                raise
            except (httpx.HTTPError, BackendUnavailableError, ValueError) as exc:
                last_error = exc
                BACKEND_REQUESTS.labels(
                    endpoint=state.endpoint, operation=operation, outcome="failure"
                ).inc()
        raise BackendUnavailableError(f"All vLLM replicas failed: {last_error}")

    async def abuild_cache(self, command: BuildCacheCommand) -> CacheInfo:
        prefix = self._prefix_value(command)
        prefix_sha, cache_id = self._cache_identity(command.tenant_id, prefix)
        try:
            return (await asyncio.to_thread(self.store.get, cache_id, command.tenant_id)).info
        except (CacheNotFoundError, CacheCompatibilityError, CacheExpiredError):
            pass

        lock = None
        acquire_lock = getattr(self.store, "acquire_build_lock", None)
        release_lock = getattr(self.store, "release_build_lock", None)
        try:
            if acquire_lock is not None:
                lock = await asyncio.to_thread(acquire_lock, cache_id, command.tenant_id)
                try:
                    return (
                        await asyncio.to_thread(self.store.get, cache_id, command.tenant_id)
                    ).info
                except (CacheNotFoundError, CacheCompatibilityError, CacheExpiredError):
                    pass

            payload: Dict[str, Any] = {
                "model": self.settings.model_id,
                "prompt": prefix,
                "max_tokens": 1,
                "temperature": 0,
                "stream": False,
            }
            data, endpoint = await self._post_json(
                payload,
                affinity_key=cache_id,
                operation="warm",
                request_id=command.request_id,
            )
            usage = data.get("usage") or {}
            token_count = int(usage.get("prompt_tokens") or 0)
            created = datetime.now(timezone.utc)
            expires = (
                created + timedelta(seconds=self.settings.cache_ttl_seconds)
                if self.settings.cache_ttl_seconds
                else None
            )
            info = CacheInfo(
                cache_id=cache_id,
                backend=self.name,
                model=self.settings.model_id,
                model_fingerprint=self._model_fingerprint,
                prefix_sha256=prefix_sha,
                token_count=token_count,
                layer_count=0,
                dtype="logical",
                tensor_bytes=0,
                created_at=created.isoformat(),
                chunk_size=command.chunk_size,
                expires_at=expires.isoformat() if expires else None,
                tenant_id=command.tenant_id,
                affinity_key=cache_id,
                storage=f"vllm/lmcache@{endpoint}",
            )
            await asyncio.to_thread(self.store.save, info, prefix)
            CACHE_OPERATIONS.labels(operation="warm", outcome="success").inc()
            return info
        finally:
            if lock is not None and release_lock is not None:
                await asyncio.to_thread(release_lock, lock)

    async def aget_cache(self, cache_id: str, tenant_id: str = "default") -> CacheInfo:
        return (await asyncio.to_thread(self.store.get, cache_id, tenant_id)).info

    async def alist_caches(self, tenant_id: Optional[str] = None) -> List[CacheInfo]:
        return await asyncio.to_thread(self.store.list, tenant_id)

    async def adelete_cache(self, cache_id: str, tenant_id: str = "default") -> bool:
        deleted = await asyncio.to_thread(self.store.delete, cache_id, tenant_id)
        CACHE_OPERATIONS.labels(operation="delete", outcome="success" if deleted else "miss").inc()
        return deleted

    async def acache_stats(self, tenant_id: Optional[str] = None) -> Dict[str, int]:
        return await asyncio.to_thread(self.store.stats, tenant_id)

    async def aprune_caches(self, tenant_id: Optional[str] = None) -> Dict[str, int]:
        del tenant_id
        result = await asyncio.to_thread(self.store.prune)
        CACHE_OPERATIONS.labels(operation="prune", outcome="success").inc(
            result["removed_count"] or 1
        )
        return result

    async def _resolve_prompt(
        self, command: CompletionCommand
    ) -> Tuple[Union[str, List[int]], str, int]:
        if not command.cache_id:
            if command.prompt is not None:
                value: Union[str, List[int]] = command.prompt
            elif command.input_ids is not None:
                value = command.input_ids
            else:
                raise KVCacheError("prompt or input_ids is required when kv_cache_id is absent")
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            affinity = hashlib.sha256(f"{command.tenant_id}\0{encoded}".encode()).hexdigest()
            return value, affinity, 0

        record: LogicalCacheRecord = await asyncio.to_thread(
            self.store.get, command.cache_id, command.tenant_id
        )
        prefix = record.prefix
        if isinstance(prefix, str):
            if command.input_ids is not None:
                raise CacheCompatibilityError("Text cache cannot be combined with token input_ids")
            supplied = command.prompt or ""
            if command.prompt_mode == "full":
                if not supplied.startswith(prefix):
                    raise CacheCompatibilityError(
                        "Full prompt does not start with the exact cached text prefix"
                    )
                full_prompt: Union[str, List[int]] = supplied
            else:
                full_prompt = prefix + supplied
        else:
            if command.prompt is not None:
                raise CacheCompatibilityError("Token cache cannot be combined with text prompt")
            supplied_ids = [int(value) for value in (command.input_ids or [])]
            if command.prompt_mode == "full":
                if supplied_ids[: len(prefix)] != prefix:
                    raise CacheCompatibilityError(
                        "Full input_ids do not start with the exact cached token prefix"
                    )
                full_prompt = supplied_ids
            else:
                full_prompt = prefix + supplied_ids
        return full_prompt, record.info.affinity_key or command.cache_id, record.info.token_count

    def _completion_payload(
        self, command: CompletionCommand, prompt: Union[str, List[int]], stream: bool
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.settings.model_id,
            "prompt": prompt,
            "max_tokens": command.max_tokens,
            "temperature": command.temperature,
            "top_p": command.top_p,
            "seed": command.seed,
            "stop": command.stop or None,
            "stream": stream,
        }
        if command.top_k:
            payload["top_k"] = command.top_k
        if command.stop_token_ids:
            payload["stop_token_ids"] = command.stop_token_ids
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return {key: value for key, value in payload.items() if value is not None}

    @staticmethod
    def _cached_tokens(usage: Dict[str, Any], fallback: int) -> int:
        details = usage.get("prompt_tokens_details") or {}
        return int(details.get("cached_tokens") or usage.get("cached_tokens") or fallback)

    async def acomplete(self, command: CompletionCommand) -> CompletionResult:
        started = time.perf_counter()
        prompt, affinity, _ = await self._resolve_prompt(command)
        data, _ = await self._post_json(
            self._completion_payload(command, prompt, False),
            affinity_key=affinity,
            operation="complete",
            request_id=command.request_id,
        )
        choices = data.get("choices") or []
        if not choices:
            raise UpstreamAPIError("vLLM returned no completion choice")
        choice = choices[0]
        usage = data.get("usage") or {}
        cached_tokens = self._cached_tokens(usage, 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        CACHE_TOKENS.labels(backend=self.name).inc(cached_tokens)
        COMPLETION_TOKENS.labels(backend=self.name).inc(completion_tokens)
        total_ms = (time.perf_counter() - started) * 1000
        return CompletionResult(
            text=str(choice.get("text") or ""),
            token_ids=[int(value) for value in choice.get("token_ids") or []],
            model=str(data.get("model") or self.settings.model_id),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            cached_tokens=cached_tokens,
            finish_reason=str(choice.get("finish_reason") or "stop"),
            timings_ms={
                "cache_load": 0.0,
                "input_processing": 0.0,
                "prefill": 0.0,
                "decode": 0.0,
                "total": round(total_ms, 3),
            },
            completion_tokens=completion_tokens,
        )

    async def astream_complete(self, command: CompletionCommand) -> AsyncIterator[CompletionChunk]:
        prompt, affinity, _ = await self._resolve_prompt(command)
        payload = self._completion_payload(command, prompt, True)
        state: ReplicaState = await self.router.select(affinity)
        headers = {"X-Request-ID": command.request_id} if command.request_id else None
        started = time.perf_counter()
        final_seen = False
        prompt_tokens = 0
        cached_tokens = 0
        completion_tokens = 0
        try:
            async with self.router.lease(state):
                async with self.client.stream(
                    "POST",
                    f"{state.endpoint}/v1/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        error = self._upstream_error(response)
                        if response.status_code >= 500:
                            raise BackendUnavailableError(str(error))
                        raise error
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if not raw or raw == "[DONE]":
                            continue
                        event = json.loads(raw)
                        usage = event.get("usage") or {}
                        if usage:
                            prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                            completion_tokens = int(
                                usage.get("completion_tokens") or completion_tokens
                            )
                            cached_tokens = self._cached_tokens(usage, cached_tokens)
                        choices = event.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0]
                        finish_reason = choice.get("finish_reason")
                        final_seen = final_seen or finish_reason is not None
                        token_ids = [int(value) for value in choice.get("token_ids") or []]
                        if token_ids:
                            completion_tokens += len(token_ids)
                        yield CompletionChunk(
                            text=str(choice.get("text") or ""),
                            model=str(event.get("model") or self.settings.model_id),
                            token_ids=token_ids,
                            finish_reason=finish_reason,
                            prompt_tokens=prompt_tokens,
                            cached_tokens=cached_tokens,
                            completion_tokens=completion_tokens,
                        )
            BACKEND_REQUESTS.labels(
                endpoint=state.endpoint, operation="stream", outcome="success"
            ).inc()
            BACKEND_DURATION.labels(endpoint=state.endpoint, operation="stream").observe(
                time.perf_counter() - started
            )
        except asyncio.CancelledError:
            BACKEND_REQUESTS.labels(
                endpoint=state.endpoint, operation="stream", outcome="cancelled"
            ).inc()
            raise
        except (httpx.HTTPError, ValueError) as exc:
            BACKEND_REQUESTS.labels(
                endpoint=state.endpoint, operation="stream", outcome="failure"
            ).inc()
            raise BackendUnavailableError(f"vLLM streaming request failed: {exc}") from exc
        except Exception:
            BACKEND_REQUESTS.labels(
                endpoint=state.endpoint, operation="stream", outcome="failure"
            ).inc()
            raise
        finally:
            CACHE_TOKENS.labels(backend=self.name).inc(cached_tokens)
            COMPLETION_TOKENS.labels(backend=self.name).inc(completion_tokens)
        if not final_seen:
            yield CompletionChunk(
                text="",
                model=self.settings.model_id,
                token_ids=[],
                finish_reason="stop",
                prompt_tokens=prompt_tokens,
                cached_tokens=cached_tokens,
                completion_tokens=completion_tokens,
            )

    async def aclose(self) -> None:
        await asyncio.to_thread(self.store.checkpoint)
        close = getattr(self.store, "close", None)
        if close is not None:
            await asyncio.to_thread(close)
        if self._owns_client:
            await self.client.aclose()
