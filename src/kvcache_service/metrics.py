"""Prometheus metrics shared by the API gateway and production backends."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter(
    "kvcache_http_requests_total",
    "HTTP requests handled by the gateway.",
    ("method", "route", "status"),
)
HTTP_DURATION = Histogram(
    "kvcache_http_request_duration_seconds",
    "HTTP request latency excluding streamed response consumption.",
    ("method", "route"),
)
INFLIGHT_REQUESTS = Gauge(
    "kvcache_inflight_requests",
    "Requests currently admitted for inference.",
)
REJECTED_REQUESTS = Counter(
    "kvcache_rejected_requests_total",
    "Requests rejected by admission control.",
    ("reason",),
)
BACKEND_REQUESTS = Counter(
    "kvcache_backend_requests_total",
    "Requests sent to an inference replica.",
    ("endpoint", "operation", "outcome"),
)
BACKEND_DURATION = Histogram(
    "kvcache_backend_request_duration_seconds",
    "Inference replica request latency.",
    ("endpoint", "operation"),
)
BACKEND_INFLIGHT = Gauge(
    "kvcache_backend_inflight_requests",
    "Active requests per inference replica.",
    ("endpoint",),
)
BACKEND_HEALTHY = Gauge(
    "kvcache_backend_healthy",
    "Whether an inference replica is available for routing.",
    ("endpoint",),
)
LMCACHE_HEALTHY = Gauge(
    "kvcache_lmcache_healthy",
    "Whether the configured LMCache MP HTTP frontend is healthy.",
)
CACHE_OPERATIONS = Counter(
    "kvcache_cache_operations_total",
    "Logical and physical cache operations.",
    ("operation", "outcome"),
)
CACHE_TOKENS = Counter(
    "kvcache_cached_tokens_total",
    "Prompt tokens reported as served from cache.",
    ("backend",),
)
COMPLETION_TOKENS = Counter(
    "kvcache_completion_tokens_total",
    "Completion tokens returned by the service.",
    ("backend",),
)
