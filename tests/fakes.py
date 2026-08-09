from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List

from kvcache_service.backend import KVCacheBackend
from kvcache_service.domain import (
    BuildCacheCommand,
    CacheInfo,
    CompletionCommand,
    CompletionResult,
)
from kvcache_service.errors import CacheNotFoundError


class FakeConfig:
    max_position_embeddings = 128
    eos_token_id = None
    _commit_hash = "test-commit"

    def to_dict(self) -> Dict[str, Any]:
        return {"max_position_embeddings": self.max_position_embeddings, "model_type": "fake"}


class FakeTokenizer:
    vocab_size = 32
    eos_token_id = None

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = [2 + (ord(character) % 20) for character in text]
        return ([1] if add_special_tokens else []) + tokens

    def decode(self, token_ids: List[int], **_: Any) -> str:
        return "".join(chr(96 + (token_id % 26 or 26)) for token_id in token_ids)


class FakeCausalLM:
    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.config = FakeConfig()
        self._parameter = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self._parameter

    def to(self, device: Any) -> FakeCausalLM:
        self._parameter.data = self._parameter.data.to(device)
        return self

    def eval(self) -> FakeCausalLM:
        return self

    def __call__(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        past_key_values: Any,
        use_cache: bool,
        return_dict: bool,
    ) -> Any:
        del attention_mask, use_cache, return_dict
        new_state = input_ids.to(dtype=self.torch.float32).view(1, 1, -1, 1)
        if past_key_values is None:
            old_key = self.torch.empty((1, 1, 0, 1), device=input_ids.device)
            old_value = self.torch.empty((1, 1, 0, 1), device=input_ids.device)
        else:
            old_key, old_value = past_key_values[0]
        key = self.torch.cat((old_key, new_state), dim=-2)
        value = self.torch.cat((old_value, new_state + 0.5), dim=-2)
        logits = self.torch.zeros(
            (1, input_ids.shape[1], self.config_vocab_size), device=input_ids.device
        )
        for index, token_id in enumerate(input_ids[0].tolist()):
            logits[0, index, (int(token_id) + 1) % self.config_vocab_size] = 10
        return SimpleNamespace(past_key_values=((key, value),), logits=logits)

    @property
    def config_vocab_size(self) -> int:
        return 32


class FakeBackend(KVCacheBackend):
    name = "fake"

    def __init__(self) -> None:
        self._items: Dict[str, CacheInfo] = {}

    def health(self) -> Dict[str, Any]:
        return {"status": "ok", "backend": self.name, "model": "fake/model"}

    def build_cache(self, command: BuildCacheCommand) -> CacheInfo:
        source = command.text or ",".join(str(item) for item in command.input_ids or [])
        cache_id = hashlib.sha256(source.encode()).hexdigest()
        info = CacheInfo(
            cache_id=cache_id,
            backend=self.name,
            model="fake/model",
            model_fingerprint="f" * 64,
            prefix_sha256="e" * 64,
            token_count=len(source),
            layer_count=1,
            dtype="float32",
            tensor_bytes=128,
            created_at=datetime.now(timezone.utc).isoformat(),
            chunk_size=command.chunk_size,
        )
        self._items[cache_id] = info
        return info

    def get_cache(self, cache_id: str) -> CacheInfo:
        try:
            return self._items[cache_id]
        except KeyError as exc:
            raise CacheNotFoundError(cache_id) from exc

    def list_caches(self) -> List[CacheInfo]:
        return list(self._items.values())

    def delete_cache(self, cache_id: str) -> bool:
        return self._items.pop(cache_id, None) is not None

    def complete(self, command: CompletionCommand) -> CompletionResult:
        return CompletionResult(
            text="done",
            token_ids=[1, 2],
            model="fake/model",
            prompt_tokens=10,
            cached_tokens=8 if command.cache_id else 0,
            finish_reason="length",
        )
