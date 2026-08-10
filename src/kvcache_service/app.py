"""Production-capable FastAPI gateway for KV cache and completion backends."""

from __future__ import annotations

import asyncio
import json
import re
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import make_asgi_app

from . import __version__
from .admission import AdmissionController
from .api_models import (
    CacheCreateRequest,
    CacheInfoResponse,
    CacheListResponse,
    CachePruneResponse,
    CacheStoreStatsResponse,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    DeleteResponse,
    ModelItem,
    ModelListResponse,
    Usage,
)
from .backend import KVCacheBackend, load_backend
from .config import Settings
from .domain import BuildCacheCommand, CompletionChunk, CompletionCommand
from .errors import (
    CacheNotFoundError,
    KVCacheError,
    RequestTimeoutError,
)
from .metrics import HTTP_DURATION, HTTP_REQUESTS

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9._:@/-]{1,128}$")


def _error_response(status_code: int, code: str, message: str, error_type: str) -> JSONResponse:
    headers = {"Retry-After": "1"} if status_code == 429 else None
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={"error": {"code": code, "message": message, "type": error_type}},
    )


def _token_cost(request: CompletionRequest) -> int:
    if request.input_ids is not None:
        input_tokens = len(request.input_ids)
    elif request.prompt:
        input_tokens = max(1, len(request.prompt) // 4)
    else:
        input_tokens = 1
    return input_tokens + request.max_tokens


def _build_token_cost(request: CacheCreateRequest) -> int:
    if request.input_ids is not None:
        return max(1, len(request.input_ids))
    return max(1, len(request.text or "") // 4)


def _completion_command(request: CompletionRequest, context: Request) -> CompletionCommand:
    return CompletionCommand(
        cache_id=request.kv_cache_id,
        prompt=request.prompt,
        input_ids=request.input_ids,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        seed=request.seed,
        stop=request.normalized_stop(),
        stop_token_ids=request.stop_token_ids,
        prompt_mode=request.prompt_mode,
        tenant_id=context.state.tenant_id,
        request_id=context.state.request_id,
    )


def create_app(
    settings: Optional[Settings] = None, backend: Optional[KVCacheBackend] = None
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    limiter = AdmissionController(resolved_settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        yield
        await limiter.drain(resolved_settings.shutdown_grace_seconds)
        selected_backend = application.state.backend
        if selected_backend is not None:
            await selected_backend.aclose()

    application = FastAPI(
        title="KV Cache Service",
        version=__version__,
        description=(
            "Cache-aware, multi-tenant gateway for persistent prefixes and distributed LLM "
            "inference."
        ),
        lifespan=lifespan,
    )
    application.state.backend = backend
    application.state.backend_lock = threading.Lock()
    application.state.limiter = limiter

    def current_backend() -> KVCacheBackend:
        if application.state.backend is None:
            with application.state.backend_lock:
                if application.state.backend is None:
                    application.state.backend = load_backend(resolved_settings)
        return application.state.backend

    def authenticate(request: Request) -> Optional[JSONResponse]:
        supplied_tenant = request.headers.get(resolved_settings.tenant_header, "default").strip()
        if not _SAFE_IDENTIFIER.fullmatch(supplied_tenant):
            return _error_response(
                400, "invalid_tenant", "Invalid tenant identifier", "request_error"
            )

        authorization = request.headers.get("authorization", "")
        bearer = authorization[7:] if authorization.lower().startswith("bearer ") else ""
        supplied_key = bearer or request.headers.get("x-api-key", "")

        if resolved_settings.api_keys:
            expected = resolved_settings.api_keys.get(supplied_tenant)
            if (
                expected is None
                or not supplied_key
                or not secrets.compare_digest(supplied_key, expected)
            ):
                return _error_response(
                    401,
                    "invalid_api_key",
                    "A valid tenant API key is required",
                    "authentication_error",
                )
        elif resolved_settings.api_key:
            if not supplied_key or not secrets.compare_digest(
                supplied_key, resolved_settings.api_key
            ):
                return _error_response(
                    401,
                    "invalid_api_key",
                    "A valid API key is required",
                    "authentication_error",
                )

        request.state.tenant_id = supplied_tenant
        return None

    def authenticate_admin(request: Request) -> Optional[JSONResponse]:
        authorization = request.headers.get("authorization", "")
        bearer = authorization[7:] if authorization.lower().startswith("bearer ") else ""
        supplied_key = request.headers.get("x-admin-key", "") or bearer
        expected = resolved_settings.admin_api_key
        if (
            expected is None
            or not supplied_key
            or not secrets.compare_digest(supplied_key, expected)
        ):
            return _error_response(
                401,
                "invalid_admin_key",
                "A valid administrative API key is required",
                "authentication_error",
            )
        request.state.is_admin = True
        return None

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        started = time.perf_counter()
        supplied_request_id = request.headers.get("x-request-id", "").strip()
        request_id = (
            supplied_request_id
            if _SAFE_IDENTIFIER.fullmatch(supplied_request_id)
            else f"req-{uuid.uuid4().hex}"
        )
        request.state.request_id = request_id
        request.state.tenant_id = "default"
        request.state.is_admin = False

        if request.url.path.startswith("/v1/admin/"):
            rejection = authenticate_admin(request)
            if rejection is not None:
                rejection.headers["X-Request-ID"] = request_id
                return rejection
        elif request.url.path.startswith("/v1/"):
            rejection = authenticate(request)
            if rejection is not None:
                rejection.headers["X-Request-ID"] = request_id
                return rejection

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        HTTP_REQUESTS.labels(
            method=request.method, route=route_path, status=str(response.status_code)
        ).inc()
        HTTP_DURATION.labels(method=request.method, route=route_path).observe(
            time.perf_counter() - started
        )
        return response

    @application.exception_handler(KVCacheError)
    async def kv_cache_error_handler(_: Request, exc: KVCacheError) -> JSONResponse:
        error_type = "rate_limit_error" if exc.status_code == 429 else "invalid_request_error"
        return _error_response(exc.status_code, exc.code, str(exc), error_type)

    @application.exception_handler(asyncio.TimeoutError)
    async def timeout_error_handler(_: Request, exc: asyncio.TimeoutError) -> JSONResponse:
        del exc
        error = RequestTimeoutError("The request exceeded its configured deadline")
        return _error_response(error.status_code, error.code, str(error), "timeout_error")

    @application.get("/livez")
    async def livez() -> dict:
        return {"status": "ok", "version": __version__}

    @application.get("/health")
    async def health() -> dict:
        return await current_backend().ahealth()

    @application.get("/readyz")
    async def readyz() -> JSONResponse:
        if limiter.draining:
            return _error_response(503, "draining", "Service is draining", "service_error")
        health_data = await current_backend().ahealth()
        if health_data.get("status") not in {"ok", "configured"}:
            return _error_response(
                503, "backend_unavailable", "No inference backend is ready", "service_error"
            )
        return JSONResponse({"status": "ok", "backend": health_data})

    @application.get("/v1/models", response_model=ModelListResponse)
    async def models() -> ModelListResponse:
        health_data = await current_backend().ahealth()
        return ModelListResponse(data=[ModelItem(id=str(health_data["model"]))])

    @application.get("/v1/admin/status")
    async def admin_status() -> dict:
        selected_backend = current_backend()
        return {
            "version": __version__,
            "draining": limiter.draining,
            "backend": await selected_backend.ahealth(),
            "cache": await selected_backend.acache_stats(None),
            "limits": {
                "max_concurrent_requests": resolved_settings.max_concurrent_requests,
                "max_concurrent_per_tenant": resolved_settings.max_concurrent_per_tenant,
                "tokens_per_minute": resolved_settings.rate_limit_tokens_per_minute,
            },
        }

    @application.post("/v1/kv-caches", response_model=CacheInfoResponse, status_code=201)
    async def build_cache(request: CacheCreateRequest, context: Request) -> CacheInfoResponse:
        selected_backend = current_backend()
        command = BuildCacheCommand(
            text=request.text,
            input_ids=request.input_ids,
            chunk_size=request.chunk_size or resolved_settings.default_chunk_size,
            tenant_id=context.state.tenant_id,
            request_id=context.state.request_id,
        )
        async with limiter.admit(context.state.tenant_id, _build_token_cost(request)):
            try:
                info = await asyncio.wait_for(
                    selected_backend.abuild_cache(command),
                    timeout=resolved_settings.request_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise RequestTimeoutError("Cache warm request timed out") from exc
        return CacheInfoResponse(**info.to_dict())

    @application.get("/v1/kv-caches", response_model=CacheListResponse)
    async def list_caches(context: Request) -> CacheListResponse:
        items = await current_backend().alist_caches(context.state.tenant_id)
        return CacheListResponse(data=[CacheInfoResponse(**item.to_dict()) for item in items])

    @application.get("/v1/kv-caches/stats", response_model=CacheStoreStatsResponse)
    async def cache_stats(context: Request) -> CacheStoreStatsResponse:
        stats = await current_backend().acache_stats(context.state.tenant_id)
        return CacheStoreStatsResponse(**stats)

    @application.post("/v1/kv-caches/prune", response_model=CachePruneResponse)
    async def prune_caches(context: Request) -> CachePruneResponse:
        selected_backend = current_backend()
        result = await selected_backend.aprune_caches(context.state.tenant_id)
        stats = await selected_backend.acache_stats(context.state.tenant_id)
        return CachePruneResponse(**result, stats=CacheStoreStatsResponse(**stats))

    @application.get("/v1/kv-caches/{cache_id}", response_model=CacheInfoResponse)
    async def get_cache(cache_id: str, context: Request) -> CacheInfoResponse:
        info = await current_backend().aget_cache(cache_id, context.state.tenant_id)
        return CacheInfoResponse(**info.to_dict())

    @application.delete("/v1/kv-caches/{cache_id}", response_model=DeleteResponse)
    async def delete_cache(cache_id: str, context: Request) -> DeleteResponse:
        deleted = await current_backend().adelete_cache(cache_id, context.state.tenant_id)
        if not deleted:
            raise CacheNotFoundError(f"Cache {cache_id!r} does not exist")
        return DeleteResponse(id=cache_id, deleted=True)

    async def stream_events(
        context: Request,
        command: CompletionCommand,
        completion_id: str,
        created: int,
        token_cost: int,
    ) -> AsyncIterator[str]:
        try:
            async with limiter.admit(command.tenant_id, token_cost):
                iterator = current_backend().astream_complete(command).__aiter__()
                try:
                    deadline = time.monotonic() + resolved_settings.request_timeout_seconds
                    while True:
                        if await context.is_disconnected():
                            break
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise RequestTimeoutError("Streaming completion timed out")
                        try:
                            chunk: CompletionChunk = await asyncio.wait_for(
                                iterator.__anext__(), timeout=remaining
                            )
                        except StopAsyncIteration:
                            break
                        payload = {
                            "id": completion_id,
                            "object": "text_completion",
                            "created": created,
                            "model": chunk.model,
                            "choices": [
                                {
                                    "text": chunk.text,
                                    "index": 0,
                                    "logprobs": None,
                                    "finish_reason": chunk.finish_reason,
                                    "token_ids": chunk.token_ids,
                                }
                            ],
                        }
                        if chunk.finish_reason is not None:
                            payload["usage"] = {
                                "prompt_tokens": chunk.prompt_tokens,
                                "completion_tokens": chunk.completion_tokens,
                                "total_tokens": chunk.prompt_tokens + chunk.completion_tokens,
                                "cached_tokens": chunk.cached_tokens,
                            }
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                finally:
                    close_iterator = getattr(iterator, "aclose", None)
                    if close_iterator is not None:
                        await close_iterator()
        except KVCacheError as exc:
            error_payload = {
                "error": {"code": exc.code, "message": str(exc), "type": "stream_error"}
            }
            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
        except asyncio.TimeoutError:
            error = RequestTimeoutError("Streaming completion timed out")
            error_payload = {
                "error": {"code": error.code, "message": str(error), "type": "stream_error"}
            }
            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            raise
        yield "data: [DONE]\n\n"

    @application.post("/v1/completions")
    async def completions(request: CompletionRequest, context: Request):
        selected_backend = current_backend()
        backend_model = str(selected_backend.health()["model"])
        if request.model is not None and request.model != backend_model:
            raise KVCacheError(
                f"Requested model {request.model!r}, but this server hosts {backend_model!r}"
            )
        command = _completion_command(request, context)
        completion_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        cost = _token_cost(request)

        if request.stream:
            return StreamingResponse(
                stream_events(context, command, completion_id, created, cost),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        async with limiter.admit(context.state.tenant_id, cost):
            try:
                result = await asyncio.wait_for(
                    selected_backend.acomplete(command),
                    timeout=resolved_settings.request_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise RequestTimeoutError("Completion request timed out") from exc
        completion_tokens = (
            result.completion_tokens
            if result.completion_tokens is not None
            else len(result.token_ids)
        )
        return CompletionResponse(
            id=completion_id,
            created=created,
            model=result.model,
            choices=[
                CompletionChoice(
                    text=result.text,
                    finish_reason=result.finish_reason,
                    token_ids=result.token_ids,
                )
            ],
            usage=Usage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=result.prompt_tokens + completion_tokens,
                cached_tokens=result.cached_tokens,
            ),
            kv_cache_id=request.kv_cache_id,
            timings_ms=result.timings_ms,
        )

    if resolved_settings.metrics_enabled:
        application.mount("/metrics", make_asgi_app())

    return application


app = create_app()
