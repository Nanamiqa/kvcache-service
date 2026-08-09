"""Compatibility helpers for Transformers legacy and DynamicCache representations."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .errors import CacheCompatibilityError


def _cache_pairs(cache: Any) -> List[Tuple[Any, Any]]:
    if isinstance(cache, (tuple, list)):
        return [(layer[0], layer[1]) for layer in cache]

    converter = getattr(cache, "to_legacy_cache", None)
    if callable(converter):
        legacy = converter()
        return [(layer[0], layer[1]) for layer in legacy]

    keys = getattr(cache, "key_cache", None)
    values = getattr(cache, "value_cache", None)
    if keys is not None and values is not None:
        return list(zip(keys, values))

    layers = getattr(cache, "layers", None)
    if layers is not None:
        pairs = []
        for layer in layers:
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if key is None or value is None:
                raise CacheCompatibilityError(
                    "This model uses non-K/V cache state; the Transformers backend only supports "
                    "decoder-only attention caches"
                )
            pairs.append((key, value))
        return pairs

    raise CacheCompatibilityError(
        f"Unsupported past_key_values type: {type(cache).__module__}.{type(cache).__name__}"
    )


def cache_to_tensors(
    cache: Any, input_ids: Any, next_logits: Any
) -> Tuple[Dict[str, Any], str, int]:
    pairs = _cache_pairs(cache)
    if not pairs:
        raise CacheCompatibilityError("The model returned an empty KV cache")

    tensors: Dict[str, Any] = {
        "prefix.input_ids": input_ids.detach().to("cpu").contiguous(),
        "prefix.next_logits": next_logits.detach().to("cpu").contiguous(),
    }
    token_count = int(input_ids.numel())
    tensor_bytes = 0
    dtype = str(pairs[0][0].dtype).removeprefix("torch.")
    for index, (key, value) in enumerate(pairs):
        if key.ndim < 3 or value.ndim < 3:
            raise CacheCompatibilityError(f"Layer {index} has unsupported KV tensor dimensions")
        if int(key.shape[-2]) != token_count or int(value.shape[-2]) != token_count:
            raise CacheCompatibilityError(
                "Sliding-window, hybrid, and compressed cache layouts are not serializable by "
                "this reference backend; use the vLLM + LMCache deployment instead"
            )
        tensors[f"layer.{index}.key"] = key
        tensors[f"layer.{index}.value"] = value
        tensor_bytes += key.numel() * key.element_size()
        tensor_bytes += value.numel() * value.element_size()
    tensor_bytes += input_ids.numel() * input_ids.element_size()
    tensor_bytes += next_logits.numel() * next_logits.element_size()
    return tensors, dtype, tensor_bytes


def tensors_to_cache(tensors: Mapping[str, Any], device: Any) -> Any:
    indices = sorted(
        int(name.split(".")[1])
        for name in tensors
        if name.startswith("layer.") and name.endswith(".key")
    )
    if indices != list(range(len(indices))):
        raise CacheCompatibilityError("Stored cache layer indexes are not contiguous")
    pairs: Sequence[Tuple[Any, Any]] = tuple(
        (
            tensors[f"layer.{index}.key"].to(device),
            tensors[f"layer.{index}.value"].to(device),
        )
        for index in indices
    )
    if not pairs:
        raise CacheCompatibilityError("Stored artifact does not contain KV layers")

    try:
        from transformers import DynamicCache

        converter = getattr(DynamicCache, "from_legacy_cache", None)
        if callable(converter):
            return converter(pairs)
    except (ImportError, TypeError, ValueError):
        pass
    return pairs


def get_layer_count(tensors: Mapping[str, Any]) -> int:
    return sum(1 for name in tensors if name.startswith("layer.") and name.endswith(".key"))
