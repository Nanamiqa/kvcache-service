"""Bounded concurrency and per-tenant token admission control."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Dict

from .config import Settings
from .errors import BackendUnavailableError, RequestOverloadedError
from .metrics import INFLIGHT_REQUESTS, REJECTED_REQUESTS


@dataclass
class _TokenBucket:
    tokens: float
    updated_at: float


class AdmissionController:
    """Admission controller that fails fast instead of allowing unbounded queues."""

    def __init__(self, settings: Settings) -> None:
        self._global = asyncio.Semaphore(settings.max_concurrent_requests)
        self._per_tenant_limit = settings.max_concurrent_per_tenant
        self._tenant_semaphores: Dict[str, asyncio.Semaphore] = {}
        self._tenant_lock = asyncio.Lock()
        self._bucket_lock = asyncio.Lock()
        self._buckets: Dict[str, _TokenBucket] = {}
        self._tokens_per_minute = settings.rate_limit_tokens_per_minute
        self._burst = settings.rate_limit_burst_tokens or self._tokens_per_minute
        self._queue_timeout = settings.admission_queue_timeout_seconds
        self._active = 0
        self._active_condition = asyncio.Condition()
        self._draining = False

    @property
    def draining(self) -> bool:
        return self._draining

    async def _tenant_semaphore(self, tenant_id: str) -> asyncio.Semaphore:
        async with self._tenant_lock:
            return self._tenant_semaphores.setdefault(
                tenant_id, asyncio.Semaphore(self._per_tenant_limit)
            )

    async def _consume_tokens(self, tenant_id: str, amount: int) -> None:
        if self._tokens_per_minute <= 0:
            return
        now = time.monotonic()
        async with self._bucket_lock:
            bucket = self._buckets.get(tenant_id)
            if bucket is None:
                bucket = _TokenBucket(tokens=float(self._burst), updated_at=now)
                self._buckets[tenant_id] = bucket
            elapsed = now - bucket.updated_at
            bucket.tokens = min(
                float(self._burst),
                bucket.tokens + elapsed * (self._tokens_per_minute / 60.0),
            )
            bucket.updated_at = now
            if amount > bucket.tokens:
                REJECTED_REQUESTS.labels(reason="token_budget").inc()
                raise RequestOverloadedError("Tenant token rate limit exceeded")
            bucket.tokens -= amount

    @asynccontextmanager
    async def admit(self, tenant_id: str, token_cost: int) -> AsyncIterator[None]:
        if self._draining:
            REJECTED_REQUESTS.labels(reason="draining").inc()
            raise BackendUnavailableError("Service is draining and not accepting new work")
        await self._consume_tokens(tenant_id, max(token_cost, 1))
        tenant = await self._tenant_semaphore(tenant_id)
        global_acquired = False
        tenant_acquired = False
        try:
            await asyncio.wait_for(self._global.acquire(), timeout=self._queue_timeout)
            global_acquired = True
            await asyncio.wait_for(tenant.acquire(), timeout=self._queue_timeout)
            tenant_acquired = True
        except asyncio.TimeoutError as exc:
            if tenant_acquired:
                tenant.release()
            if global_acquired:
                self._global.release()
            REJECTED_REQUESTS.labels(reason="concurrency").inc()
            raise RequestOverloadedError("Inference concurrency limit reached") from exc

        async with self._active_condition:
            self._active += 1
            INFLIGHT_REQUESTS.set(self._active)
        try:
            yield
        finally:
            if tenant_acquired:
                tenant.release()
            if global_acquired:
                self._global.release()
            async with self._active_condition:
                self._active -= 1
                INFLIGHT_REQUESTS.set(self._active)
                self._active_condition.notify_all()

    async def drain(self, timeout: float) -> bool:
        self._draining = True
        deadline = time.monotonic() + timeout
        async with self._active_condition:
            while self._active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                try:
                    await asyncio.wait_for(self._active_condition.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    return False
        return True
