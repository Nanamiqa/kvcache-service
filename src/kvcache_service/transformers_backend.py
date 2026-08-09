"""Single-model Transformers backend with persistent, reusable raw KV tensors."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .backend import KVCacheBackend
from .cache_codec import cache_to_tensors, tensors_to_cache
from .config import Settings
from .domain import BuildCacheCommand, CacheInfo, CompletionCommand, CompletionResult
from .errors import CacheCompatibilityError, ContextLengthError, KVCacheError
from .store import SafeTensorCacheStore


def _hash_tokens(model_fingerprint: str, token_ids: Sequence[int]) -> Tuple[str, str]:
    prefix_digest = hashlib.sha256()
    cache_digest = hashlib.sha256(model_fingerprint.encode("utf-8"))
    cache_digest.update(b"\0")
    for token_id in token_ids:
        encoded = int(token_id).to_bytes(8, byteorder="little", signed=True)
        prefix_digest.update(encoded)
        cache_digest.update(encoded)
    return prefix_digest.hexdigest(), cache_digest.hexdigest()


class TransformersBackend(KVCacheBackend):
    name = "transformers"

    def __init__(
        self,
        settings: Settings,
        *,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        store: Optional[SafeTensorCacheStore] = None,
    ) -> None:
        self.settings = settings
        self.store = store or SafeTensorCacheStore(
            settings.store_dir,
            verify_checksum=settings.verify_checksum,
            ttl_seconds=settings.cache_ttl_seconds,
            max_store_bytes=settings.max_store_bytes,
        )
        self._model = model
        self._tokenizer = tokenizer
        self._device: Optional[Any] = None
        self._torch: Optional[Any] = None
        self._model_fingerprint: Optional[str] = None
        self._context_limit: Optional[int] = None
        self._load_lock = threading.RLock()
        self._inference_lock = threading.RLock()

    def _ensure_loaded(self) -> None:
        if self._model_fingerprint is not None:
            return
        with self._load_lock:
            if self._model_fingerprint is not None:
                return
            try:
                import torch
            except ImportError as exc:
                raise KVCacheError(
                    "The Transformers backend needs the local extra: pip install -e '.[local]'"
                ) from exc
            self._torch = torch

            if self._model is None or self._tokenizer is None:
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                except ImportError as exc:
                    raise KVCacheError(
                        "The Transformers backend needs the local extra: pip install -e '.[local]'"
                    ) from exc
                self._device = self._select_device(torch)
                dtype = self._select_dtype(torch, self._device)
                load_options = {
                    "revision": self.settings.model_revision,
                    "trust_remote_code": self.settings.trust_remote_code,
                    "local_files_only": self.settings.local_files_only,
                }
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.settings.model_id, **load_options
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.settings.model_id,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    **load_options,
                )
                self._model.to(self._device)
                self._model.eval()
            else:
                self._device = self._select_device(torch)
                if hasattr(self._model, "to"):
                    self._model.to(self._device)
                if hasattr(self._model, "eval"):
                    self._model.eval()

            self._context_limit = self._detect_context_limit()
            self._model_fingerprint = self._fingerprint()

    def _select_device(self, torch: Any) -> Any:
        if self.settings.device != "auto":
            return torch.device(self.settings.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _select_dtype(self, torch: Any, device: Any) -> Any:
        if self.settings.dtype != "auto":
            dtype = getattr(torch, self.settings.dtype, None)
            if dtype is None:
                raise KVCacheError(f"Unknown torch dtype {self.settings.dtype!r}")
            return dtype
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if device.type == "mps":
            return torch.float16
        return torch.float32

    def _detect_context_limit(self) -> int:
        if self.settings.max_context_tokens:
            return self.settings.max_context_tokens
        config = self._model.config
        for attribute in ("max_position_embeddings", "max_sequence_length", "seq_length"):
            value = getattr(config, attribute, None)
            if isinstance(value, int) and 0 < value < 1_000_000_000:
                return value
        raise KVCacheError(
            "Cannot detect the model context window; set KVCACHE_MAX_CONTEXT_TOKENS explicitly"
        )

    def _fingerprint(self) -> str:
        config = self._model.config
        config_data = config.to_dict() if hasattr(config, "to_dict") else vars(config)
        dtype = (
            str(next(iter(self._model.parameters())).dtype)
            if hasattr(self._model, "parameters")
            else "unknown"
        )
        tokenizer_data = {
            "class": type(self._tokenizer).__name__,
            "vocab_size": getattr(self._tokenizer, "vocab_size", None),
        }
        payload = {
            "model_id": self.settings.model_id,
            "source": self._model_source_identity(),
            "dtype": dtype,
            "config": config_data,
            "tokenizer": tokenizer_data,
        }
        encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _model_source_identity(self) -> Dict[str, Any]:
        if self.settings.model_fingerprint:
            return {"kind": "explicit", "value": self.settings.model_fingerprint}

        config = self._model.config
        tokenizer_commit = getattr(self._tokenizer, "init_kwargs", {}).get("_commit_hash")
        commit = getattr(config, "_commit_hash", None) or tokenizer_commit
        if commit:
            return {"kind": "hub", "commit": commit}

        model_path = Path(self.settings.model_id).expanduser()
        if not model_path.is_dir():
            return {"kind": "revision", "revision": self.settings.model_revision}

        digest = hashlib.sha256()
        weight_suffixes = {".bin", ".pt", ".pth", ".safetensors"}
        content_suffixes = {".json", ".model", ".py", ".tiktoken", ".txt"}
        for path in sorted(item for item in model_path.rglob("*") if item.is_file()):
            relative = str(path.relative_to(model_path))
            stat = path.stat()
            digest.update(relative.encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            if path.suffix in content_suffixes and stat.st_size <= 16 * 1024 * 1024:
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
            elif path.suffix in weight_suffixes:
                digest.update(str(stat.st_mtime_ns).encode("ascii"))
        return {"kind": "local_manifest", "sha256": digest.hexdigest()}

    def _validate_ids(
        self, token_ids: Sequence[int], label: str, *, allow_empty: bool = False
    ) -> List[int]:
        if not token_ids and not allow_empty:
            raise KVCacheError(f"{label} must contain at least one token")
        normalized = [int(token_id) for token_id in token_ids]
        vocab_size = getattr(self._tokenizer, "vocab_size", None)
        if any(token_id < 0 for token_id in normalized):
            raise KVCacheError(f"{label} contains a negative token id")
        if isinstance(vocab_size, int) and any(token_id >= vocab_size for token_id in normalized):
            raise KVCacheError(f"{label} contains a token outside tokenizer vocabulary")
        return normalized

    def _encode(self, text: str, *, prefix: bool) -> List[int]:
        encoded = self._tokenizer.encode(text, add_special_tokens=prefix)
        return self._validate_ids(encoded, "Encoded text")

    def _check_context(self, token_count: int) -> None:
        if token_count > int(self._context_limit):
            raise ContextLengthError(
                f"Requested context is {token_count} tokens, but model limit is "
                f"{self._context_limit}. Chunked prefill reduces peak compute memory; it cannot "
                "extend the model's positional context window."
            )

    def _forward_chunk(
        self, token_ids: Sequence[int], past: Optional[Any], total_length: int
    ) -> Tuple[Any, Any]:
        torch = self._torch
        input_tensor = torch.tensor([list(token_ids)], dtype=torch.long, device=self._device)
        attention_mask = torch.ones((1, total_length), dtype=torch.long, device=self._device)
        with torch.inference_mode():
            output = self._model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
        if getattr(output, "past_key_values", None) is None:
            raise CacheCompatibilityError("The model did not return past_key_values")
        return output.past_key_values, output.logits[:, -1, :]

    def _prefill(
        self,
        token_ids: Sequence[int],
        chunk_size: int,
        *,
        past: Optional[Any] = None,
        past_length: int = 0,
    ) -> Tuple[Any, Any]:
        if chunk_size < 1:
            raise KVCacheError("chunk_size must be >= 1")
        current = past
        logits = None
        for start in range(0, len(token_ids), chunk_size):
            chunk = token_ids[start : start + chunk_size]
            current, logits = self._forward_chunk(chunk, current, past_length + start + len(chunk))
        return current, logits

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "backend": self.name,
            "model": self.settings.model_id,
            "model_loaded": self._model_fingerprint is not None,
            "store": str(self.store.root),
            "cache_policy": {
                "ttl_seconds": self.store.ttl_seconds,
                "max_store_bytes": self.store.max_store_bytes,
            },
        }

    def build_cache(self, command: BuildCacheCommand) -> CacheInfo:
        self._ensure_loaded()
        self.store.prune()
        if command.input_ids is not None:
            token_ids = self._validate_ids(command.input_ids, "input_ids")
        elif command.text is not None:
            token_ids = self._encode(command.text, prefix=True)
        else:
            raise KVCacheError("Exactly one of text or input_ids is required")
        self._check_context(len(token_ids))
        prefix_sha256, cache_id = _hash_tokens(self._model_fingerprint, token_ids)
        if self.store.exists(cache_id):
            return self.store.get_info(cache_id)

        with self._inference_lock:
            if self.store.exists(cache_id):
                return self.store.get_info(cache_id)
            cache, next_logits = self._prefill(token_ids, command.chunk_size)
            input_tensor = self._torch.tensor(token_ids, dtype=self._torch.long)
            tensors, dtype, tensor_bytes = cache_to_tensors(
                cache, input_tensor, next_logits.squeeze(0)
            )
            info = CacheInfo(
                cache_id=cache_id,
                backend=self.name,
                model=self.settings.model_id,
                model_fingerprint=self._model_fingerprint,
                prefix_sha256=prefix_sha256,
                token_count=len(token_ids),
                layer_count=sum(
                    name.endswith(".key") and name.startswith("layer.") for name in tensors
                ),
                dtype=dtype,
                tensor_bytes=tensor_bytes,
                created_at=datetime.now(timezone.utc).isoformat(),
                chunk_size=command.chunk_size,
            )
            return self.store.save(info, tensors)

    def get_cache(self, cache_id: str) -> CacheInfo:
        return self.store.get_info(cache_id)

    def list_caches(self) -> List[CacheInfo]:
        return self.store.list()

    def delete_cache(self, cache_id: str) -> bool:
        return self.store.delete(cache_id)

    def cache_stats(self) -> Dict[str, int]:
        return self.store.stats()

    def prune_caches(self) -> Dict[str, int]:
        return self.store.prune()

    def _eos_ids(self) -> List[int]:
        eos = getattr(self._model.config, "eos_token_id", None)
        if eos is None:
            eos = getattr(self._tokenizer, "eos_token_id", None)
        if eos is None:
            return []
        return [int(item) for item in eos] if isinstance(eos, (list, tuple)) else [int(eos)]

    def _sample(self, logits: Any, command: CompletionCommand, generator: Any) -> int:
        torch = self._torch
        scores = logits.detach().float().to("cpu")
        if command.temperature == 0:
            return int(torch.argmax(scores).item())
        scores = scores / command.temperature
        if command.top_k > 0 and command.top_k < scores.numel():
            threshold = torch.topk(scores, command.top_k).values[-1]
            scores[scores < threshold] = -float("inf")
        probabilities = torch.softmax(scores, dim=-1)
        if command.top_p < 1.0:
            sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
            cumulative = torch.cumsum(sorted_probabilities, dim=-1)
            remove = cumulative > command.top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            sorted_probabilities[remove] = 0
            probabilities = torch.zeros_like(probabilities).scatter(
                0, sorted_indices, sorted_probabilities
            )
            probabilities = probabilities / probabilities.sum()
        return int(torch.multinomial(probabilities, 1, generator=generator).item())

    def _decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(
            list(token_ids), skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def _completion_input_ids(
        self, command: CompletionCommand, cached_prefix_ids: Optional[List[int]]
    ) -> List[int]:
        has_explicit_input = command.input_ids is not None or command.prompt is not None
        if command.input_ids is not None:
            provided = self._validate_ids(
                command.input_ids,
                "input_ids",
                allow_empty=cached_prefix_ids is not None and command.prompt_mode == "suffix",
            )
        elif command.prompt is not None:
            add_special_tokens = cached_prefix_ids is None or command.prompt_mode == "full"
            if command.prompt == "" and not add_special_tokens:
                provided = []
            else:
                encoded = self._tokenizer.encode(
                    command.prompt, add_special_tokens=add_special_tokens
                )
                provided = self._validate_ids(
                    encoded,
                    "Encoded text",
                    allow_empty=cached_prefix_ids is not None and command.prompt_mode == "suffix",
                )
        else:
            provided = []

        if cached_prefix_ids is None or command.prompt_mode == "suffix":
            return provided
        if not has_explicit_input:
            return []
        prefix_length = len(cached_prefix_ids)
        if len(provided) < prefix_length or provided[:prefix_length] != cached_prefix_ids:
            mismatch = next(
                (
                    index
                    for index, (expected, actual) in enumerate(zip(cached_prefix_ids, provided))
                    if expected != actual
                ),
                min(len(provided), prefix_length),
            )
            raise CacheCompatibilityError(
                "Full prompt does not start with the exact cached token prefix; "
                f"first mismatch is at token index {mismatch}"
            )
        return provided[prefix_length:]

    def complete(self, command: CompletionCommand) -> CompletionResult:
        total_started = time.perf_counter()
        self._ensure_loaded()
        if command.max_tokens > self.settings.max_new_tokens:
            raise KVCacheError(f"max_tokens exceeds service limit {self.settings.max_new_tokens}")
        if command.prompt_mode not in {"suffix", "full"}:
            raise KVCacheError("prompt_mode must be 'suffix' or 'full'")

        with self._inference_lock:
            cache_load_ms = 0.0
            input_processing_ms = 0.0
            prefill_ms = 0.0
            if command.cache_id:
                load_started = time.perf_counter()
                info, tensors = self.store.load(command.cache_id)
                if info.model_fingerprint != self._model_fingerprint:
                    raise CacheCompatibilityError(
                        "Cache was built by a different model revision, tokenizer, or dtype"
                    )
                stored_prefix = tensors.get("prefix.input_ids")
                if stored_prefix is None or int(stored_prefix.numel()) != info.token_count:
                    raise CacheCompatibilityError(
                        "Stored prefix token ids are missing or inconsistent with metadata"
                    )
                cached_prefix_ids = [int(item) for item in stored_prefix.tolist()]
                past = tensors_to_cache(tensors, self._device)
                prefix_tokens = info.token_count
                logits = tensors["prefix.next_logits"].to(self._device)
                cache_load_ms = (time.perf_counter() - load_started) * 1000
                input_started = time.perf_counter()
                suffix_ids = self._completion_input_ids(command, cached_prefix_ids)
                input_processing_ms = (time.perf_counter() - input_started) * 1000
                prompt_tokens = info.token_count + len(suffix_ids)
                self._check_context(prompt_tokens + command.max_tokens)
                if suffix_ids:
                    prefill_started = time.perf_counter()
                    past, logits = self._prefill(
                        suffix_ids,
                        self.settings.default_chunk_size,
                        past=past,
                        past_length=prefix_tokens,
                    )
                    logits = logits.squeeze(0)
                    prefill_ms = (time.perf_counter() - prefill_started) * 1000
            else:
                input_started = time.perf_counter()
                suffix_ids = self._completion_input_ids(command, None)
                input_processing_ms = (time.perf_counter() - input_started) * 1000
                if not suffix_ids:
                    raise KVCacheError("prompt or input_ids is required when kv_cache_id is absent")
                prefix_tokens = 0
                prompt_tokens = len(suffix_ids)
                self._check_context(prompt_tokens + command.max_tokens)
                prefill_started = time.perf_counter()
                past, logits = self._prefill(suffix_ids, self.settings.default_chunk_size)
                logits = logits.squeeze(0)
                prefill_ms = (time.perf_counter() - prefill_started) * 1000

            prompt_tokens = prefix_tokens + len(suffix_ids)
            generated: List[int] = []
            finish_reason = "length"
            eos_ids = set(self._eos_ids())
            stop_ids = set(command.stop_token_ids)
            generator = self._torch.Generator(device="cpu")
            if command.seed is not None:
                generator.manual_seed(command.seed)
            else:
                generator.seed()

            decode_started = time.perf_counter()
            for index in range(command.max_tokens):
                token_id = self._sample(logits, command, generator)
                generated.append(token_id)
                if token_id in eos_ids or token_id in stop_ids:
                    finish_reason = "stop"
                    break
                decoded = self._decode(generated)
                if any(decoded.endswith(stop) for stop in command.stop):
                    finish_reason = "stop"
                    break
                if index + 1 < command.max_tokens:
                    past, next_logits = self._forward_chunk(
                        [token_id], past, prompt_tokens + len(generated)
                    )
                    logits = next_logits.squeeze(0)
            decode_ms = (time.perf_counter() - decode_started) * 1000

            text = self._decode(generated)
            for stop in command.stop:
                if stop and text.endswith(stop):
                    text = text[: -len(stop)]
                    break
            return CompletionResult(
                text=text,
                token_ids=generated,
                model=self.settings.model_id,
                prompt_tokens=prompt_tokens,
                cached_tokens=prefix_tokens,
                finish_reason=finish_reason,
                timings_ms={
                    "cache_load": round(cache_load_ms, 3),
                    "input_processing": round(input_processing_ms, 3),
                    "prefill": round(prefill_ms, 3),
                    "decode": round(decode_ms, 3),
                    "total": round((time.perf_counter() - total_started) * 1000, 3),
                },
            )
