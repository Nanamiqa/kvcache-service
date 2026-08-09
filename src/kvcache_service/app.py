"""FastAPI application exposing cache lifecycle and OpenAI-style completions."""

from __future__ import annotations

import secrets
import threading
import time
import uuid
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import __version__
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
from .domain import BuildCacheCommand, CompletionCommand
from .errors import CacheNotFoundError, KVCacheError


def create_app(
    settings: Optional[Settings] = None, backend: Optional[KVCacheBackend] = None
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    application = FastAPI(
        title="KV Cache Service",
        version=__version__,
        description=(
            "Persist long-prefix KV tensors and reuse them through an OpenAI-style completion API."
        ),
    )
    application.state.backend = backend
    application.state.backend_lock = threading.Lock()

    def current_backend() -> KVCacheBackend:
        if application.state.backend is None:
            with application.state.backend_lock:
                if application.state.backend is None:
                    application.state.backend = load_backend(resolved_settings)
        return application.state.backend

    @application.middleware("http")
    async def api_key_authentication(request: Request, call_next):
        if resolved_settings.api_key and request.url.path.startswith("/v1/"):
            authorization = request.headers.get("authorization", "")
            bearer = authorization[7:] if authorization.lower().startswith("bearer ") else ""
            supplied = bearer or request.headers.get("x-api-key", "")
            if not supplied or not secrets.compare_digest(supplied, resolved_settings.api_key):
                return JSONResponse(
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                    content={
                        "error": {
                            "code": "invalid_api_key",
                            "message": "A valid API key is required",
                            "type": "authentication_error",
                        }
                    },
                )
        return await call_next(request)

    @application.exception_handler(KVCacheError)
    async def kv_cache_error_handler(_: Request, exc: KVCacheError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": str(exc),
                    "type": "invalid_request_error",
                }
            },
        )

    @application.get("/health")
    def health() -> dict:
        return current_backend().health()

    @application.get("/v1/models", response_model=ModelListResponse)
    def models() -> ModelListResponse:
        health_data = current_backend().health()
        return ModelListResponse(data=[ModelItem(id=str(health_data["model"]))])

    @application.post("/v1/kv-caches", response_model=CacheInfoResponse, status_code=201)
    def build_cache(request: CacheCreateRequest) -> CacheInfoResponse:
        info = current_backend().build_cache(
            BuildCacheCommand(
                text=request.text,
                input_ids=request.input_ids,
                chunk_size=request.chunk_size or resolved_settings.default_chunk_size,
            )
        )
        return CacheInfoResponse(**info.to_dict())

    @application.get("/v1/kv-caches", response_model=CacheListResponse)
    def list_caches() -> CacheListResponse:
        return CacheListResponse(
            data=[CacheInfoResponse(**item.to_dict()) for item in current_backend().list_caches()]
        )

    @application.get("/v1/kv-caches/stats", response_model=CacheStoreStatsResponse)
    def cache_stats() -> CacheStoreStatsResponse:
        return CacheStoreStatsResponse(**current_backend().cache_stats())

    @application.post("/v1/kv-caches/prune", response_model=CachePruneResponse)
    def prune_caches() -> CachePruneResponse:
        selected_backend = current_backend()
        result = selected_backend.prune_caches()
        return CachePruneResponse(
            **result,
            stats=CacheStoreStatsResponse(**selected_backend.cache_stats()),
        )

    @application.get("/v1/kv-caches/{cache_id}", response_model=CacheInfoResponse)
    def get_cache(cache_id: str) -> CacheInfoResponse:
        return CacheInfoResponse(**current_backend().get_cache(cache_id).to_dict())

    @application.delete("/v1/kv-caches/{cache_id}", response_model=DeleteResponse)
    def delete_cache(cache_id: str) -> DeleteResponse:
        deleted = current_backend().delete_cache(cache_id)
        if not deleted:
            raise CacheNotFoundError(f"Cache {cache_id!r} does not exist")
        return DeleteResponse(id=cache_id, deleted=True)

    @application.post("/v1/completions", response_model=CompletionResponse)
    def completions(request: CompletionRequest) -> CompletionResponse:
        selected_backend = current_backend()
        backend_model = str(selected_backend.health()["model"])
        if request.model is not None and request.model != backend_model:
            raise KVCacheError(
                f"Requested model {request.model!r}, but this server hosts {backend_model!r}"
            )
        result = selected_backend.complete(
            CompletionCommand(
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
            )
        )
        completion_tokens = len(result.token_ids)
        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
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

    return application


app = create_app()
