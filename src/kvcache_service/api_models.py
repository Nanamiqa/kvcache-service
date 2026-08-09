"""Pydantic models for the public REST contract."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CacheCreateRequest(StrictModel):
    text: Optional[str] = Field(default=None, description="Prefix text to tokenize and prefill")
    input_ids: Optional[List[int]] = Field(
        default=None, description="Exact prefix token ids; preferred when token boundaries matter"
    )
    chunk_size: Optional[int] = Field(default=None, ge=1, le=32768)

    @model_validator(mode="after")
    def exactly_one_input(self) -> CacheCreateRequest:
        if (self.text is None) == (self.input_ids is None):
            raise ValueError("Exactly one of text or input_ids is required")
        if self.text == "":
            raise ValueError("text must not be empty")
        return self


class CacheInfoResponse(StrictModel):
    cache_id: str
    backend: str
    model: str
    model_fingerprint: str
    prefix_sha256: str
    token_count: int
    layer_count: int
    dtype: str
    tensor_bytes: int
    created_at: str
    chunk_size: int
    expires_at: Optional[str] = None


class CacheListResponse(StrictModel):
    object: str = "list"
    data: List[CacheInfoResponse]


class CompletionRequest(StrictModel):
    model: Optional[str] = None
    prompt: Optional[str] = None
    input_ids: Optional[List[int]] = None
    kv_cache_id: Optional[str] = None
    max_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: List[int] = Field(default_factory=list)
    prompt_mode: Literal["suffix", "full"] = Field(
        default="suffix",
        description=(
            "suffix means prompt/input_ids only contains uncached tokens; full means it contains "
            "the cached prefix too and the server verifies/removes that exact token prefix"
        ),
    )
    stream: bool = False

    @model_validator(mode="after")
    def valid_input(self) -> CompletionRequest:
        if self.prompt is not None and self.input_ids is not None:
            raise ValueError("Only one of prompt or input_ids may be supplied")
        if self.kv_cache_id is None and self.prompt is None and self.input_ids is None:
            raise ValueError("prompt or input_ids is required when kv_cache_id is absent")
        if self.stream:
            raise ValueError("Streaming is not implemented by the reference backend")
        if any(stop == "" for stop in self.normalized_stop()):
            raise ValueError("stop strings must not be empty")
        if any(token_id < 0 for token_id in self.stop_token_ids):
            raise ValueError("stop_token_ids must not contain negative ids")
        return self

    def normalized_stop(self) -> List[str]:
        if self.stop is None:
            return []
        return [self.stop] if isinstance(self.stop, str) else self.stop


class CompletionChoice(StrictModel):
    text: str
    index: int = 0
    logprobs: None = None
    finish_reason: str
    token_ids: List[int]


class Usage(StrictModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int


class CompletionResponse(StrictModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage
    kv_cache_id: Optional[str] = None
    timings_ms: Dict[str, float]


class ModelItem(StrictModel):
    id: str
    object: str = "model"
    owned_by: str = "local"


class ModelListResponse(StrictModel):
    object: str = "list"
    data: List[ModelItem]


class DeleteResponse(StrictModel):
    id: str
    object: str = "kv_cache"
    deleted: bool


class CacheStoreStatsResponse(StrictModel):
    cache_count: int
    tensor_bytes: int
    disk_bytes: int
    max_store_bytes: int
    ttl_seconds: int


class CachePruneResponse(StrictModel):
    removed_count: int
    freed_bytes: int
    stats: CacheStoreStatsResponse
