"""Cache-affine, load-aware routing across vLLM replicas."""

from __future__ import annotations

import asyncio
import hashlib
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import AsyncIterator, Dict, Iterable, List, Optional, Set

from .errors import BackendUnavailableError, UpstreamAPIError
from .metrics import BACKEND_HEALTHY, BACKEND_INFLIGHT


@dataclass
class ReplicaState:
    endpoint: str
    active_requests: int = 0
    consecutive_failures: int = 0
    circuit_open_until: float = 0.0
    ewma_latency_ms: float = 0.0
    last_error: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.circuit_open_until <= time.monotonic()


class ReplicaRouter:
    """Rendezvous-affine router with load and circuit-breaker feedback."""

    def __init__(
        self,
        endpoints: Iterable[str],
        *,
        failure_threshold: int,
        cooldown_seconds: float,
        affinity_weight: float,
    ) -> None:
        states = [ReplicaState(endpoint=value.rstrip("/")) for value in endpoints]
        if not states:
            raise ValueError("At least one vLLM endpoint is required")
        self._states: Dict[str, ReplicaState] = {state.endpoint: state for state in states}
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._affinity_weight = affinity_weight
        self._lock = asyncio.Lock()
        for state in states:
            BACKEND_HEALTHY.labels(endpoint=state.endpoint).set(1)
            BACKEND_INFLIGHT.labels(endpoint=state.endpoint).set(0)

    @staticmethod
    def _affinity_score(key: str, endpoint: str) -> float:
        digest = hashlib.sha256(f"{key}\0{endpoint}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / float(2**64 - 1)

    async def select(self, affinity_key: str, excluded: Optional[Set[str]] = None) -> ReplicaState:
        excluded = excluded or set()
        async with self._lock:
            candidates = [
                state
                for state in self._states.values()
                if state.endpoint not in excluded and state.available
            ]
            if not candidates:
                raise BackendUnavailableError("No healthy vLLM replica is available")

            def route_score(state: ReplicaState) -> float:
                affinity = self._affinity_score(affinity_key, state.endpoint)
                latency_penalty = state.ewma_latency_ms / 1000.0
                return (
                    float(state.active_requests)
                    + latency_penalty
                    - self._affinity_weight * affinity
                )

            return min(candidates, key=route_score)

    @asynccontextmanager
    async def lease(self, state: ReplicaState) -> AsyncIterator[ReplicaState]:
        started = time.perf_counter()
        async with self._lock:
            state.active_requests += 1
            BACKEND_INFLIGHT.labels(endpoint=state.endpoint).set(state.active_requests)
        try:
            yield state
        except Exception as exc:
            # A caller-side 4xx is not evidence that the replica is unhealthy.
            if not isinstance(exc, UpstreamAPIError) or exc.status_code >= 500:
                await self.record_failure(state, exc)
            raise
        else:
            latency_ms = (time.perf_counter() - started) * 1000
            await self.record_success(state, latency_ms)
        finally:
            async with self._lock:
                state.active_requests = max(0, state.active_requests - 1)
                BACKEND_INFLIGHT.labels(endpoint=state.endpoint).set(state.active_requests)

    async def record_success(self, state: ReplicaState, latency_ms: float) -> None:
        async with self._lock:
            state.consecutive_failures = 0
            state.circuit_open_until = 0.0
            state.last_error = None
            if state.ewma_latency_ms == 0:
                state.ewma_latency_ms = latency_ms
            else:
                state.ewma_latency_ms = state.ewma_latency_ms * 0.8 + latency_ms * 0.2
            BACKEND_HEALTHY.labels(endpoint=state.endpoint).set(1)

    async def record_failure(self, state: ReplicaState, error: Exception) -> None:
        async with self._lock:
            state.consecutive_failures += 1
            state.last_error = str(error)[:500]
            if state.consecutive_failures >= self._failure_threshold:
                state.circuit_open_until = time.monotonic() + self._cooldown_seconds
                BACKEND_HEALTHY.labels(endpoint=state.endpoint).set(0)

    async def record_probe(self, endpoint: str, healthy: bool, error: Optional[str] = None) -> None:
        async with self._lock:
            state = self._states[endpoint]
            if healthy:
                state.consecutive_failures = 0
                state.circuit_open_until = 0.0
                state.last_error = None
                BACKEND_HEALTHY.labels(endpoint=endpoint).set(1)
            else:
                state.last_error = error
                state.consecutive_failures = max(
                    state.consecutive_failures + 1, self._failure_threshold
                )
                state.circuit_open_until = time.monotonic() + self._cooldown_seconds
                BACKEND_HEALTHY.labels(endpoint=endpoint).set(0)

    async def snapshot(self) -> List[dict]:
        async with self._lock:
            values = []
            now = time.monotonic()
            for state in self._states.values():
                payload = asdict(state)
                payload["healthy"] = state.circuit_open_until <= now
                payload["circuit_open_seconds"] = max(0.0, state.circuit_open_until - now)
                values.append(payload)
            return values

    @property
    def endpoints(self) -> List[str]:
        return list(self._states)
