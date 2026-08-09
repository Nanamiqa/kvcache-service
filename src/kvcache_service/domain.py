"""Backend-neutral request/result types."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class CacheInfo:
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BuildCacheCommand:
    text: Optional[str]
    input_ids: Optional[List[int]]
    chunk_size: int


@dataclass(frozen=True)
class CompletionCommand:
    cache_id: Optional[str]
    prompt: Optional[str]
    input_ids: Optional[List[int]]
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    seed: Optional[int]
    stop: List[str]
    stop_token_ids: List[int]
    prompt_mode: str = "suffix"


@dataclass(frozen=True)
class CompletionResult:
    text: str
    token_ids: List[int]
    model: str
    prompt_tokens: int
    cached_tokens: int
    finish_reason: str
    timings_ms: Dict[str, float]
