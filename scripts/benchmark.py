#!/usr/bin/env python3
"""Compare uncached and persistent-KV completion latency through the public API."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional


def post(base_url: str, path: str, payload: Dict[str, Any], api_key: Optional[str]) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def timed_completion(
    base_url: str, payload: Dict[str, Any], api_key: Optional[str]
) -> tuple[float, dict]:
    started = time.perf_counter()
    response = post(base_url, "/v1/completions", payload, api_key)
    return (time.perf_counter() - started) * 1000, response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix_file", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key", default=os.getenv("KVCACHE_API_KEY"))
    parser.add_argument("--suffix", default="请用三句话概括以上内容。")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()

    prefix = args.prefix_file.read_text(encoding="utf-8").rstrip() + "\n\n"
    build_started = time.perf_counter()
    cache = post(
        args.base_url,
        "/v1/kv-caches",
        {"text": prefix, "chunk_size": args.chunk_size},
        args.api_key,
    )
    build_ms = (time.perf_counter() - build_started) * 1000

    uncached_times = []
    cached_times = []
    cached_server_timings = []
    for _ in range(args.runs):
        uncached_ms, _ = timed_completion(
            args.base_url,
            {
                "prompt": prefix + args.suffix,
                "max_tokens": args.max_tokens,
                "temperature": 0,
            },
            args.api_key,
        )
        cached_ms, cached_response = timed_completion(
            args.base_url,
            {
                "kv_cache_id": cache["cache_id"],
                "prompt": prefix + args.suffix,
                "prompt_mode": "full",
                "max_tokens": args.max_tokens,
                "temperature": 0,
            },
            args.api_key,
        )
        uncached_times.append(uncached_ms)
        cached_times.append(cached_ms)
        cached_server_timings.append(cached_response["timings_ms"])

    uncached_median = statistics.median(uncached_times)
    cached_median = statistics.median(cached_times)
    print(
        json.dumps(
            {
                "cache_id": cache["cache_id"],
                "cached_prefix_tokens": cache["token_count"],
                "cache_build_ms": round(build_ms, 3),
                "runs": args.runs,
                "uncached_median_ms": round(uncached_median, 3),
                "cached_median_ms": round(cached_median, 3),
                "speedup": round(uncached_median / cached_median, 3),
                "cached_server_timings_ms": cached_server_timings,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
