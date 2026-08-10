#!/usr/bin/env python3
"""Bounded-concurrency load test for completion latency and streaming TTFT."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class Sample:
    total_ms: float
    ttft_ms: float
    completion_tokens: int
    error: Optional[str] = None


def percentile(values: List[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


async def one_request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    payload: Dict[str, Any],
) -> Sample:
    async with semaphore:
        started = time.perf_counter()
        first_token_at: Optional[float] = None
        completion_tokens = 0
        try:
            if payload["stream"]:
                async with client.stream("POST", "/v1/completions", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        value = line[5:].strip()
                        if not value or value == "[DONE]":
                            continue
                        event = json.loads(value)
                        if "error" in event:
                            raise RuntimeError(event["error"].get("message", str(event["error"])))
                        choices = event.get("choices") or []
                        if choices and choices[0].get("text") and first_token_at is None:
                            first_token_at = time.perf_counter()
                        usage = event.get("usage") or {}
                        completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
            else:
                response = await client.post("/v1/completions", json=payload)
                response.raise_for_status()
                value = response.json()
                completion_tokens = int(value.get("usage", {}).get("completion_tokens") or 0)
                first_token_at = time.perf_counter()
        except Exception as exc:
            total_ms = (time.perf_counter() - started) * 1000
            return Sample(total_ms, total_ms, completion_tokens, str(exc)[:300])

        finished = time.perf_counter()
        first_token_at = first_token_at or finished
        return Sample(
            total_ms=(finished - started) * 1000,
            ttft_ms=(first_token_at - started) * 1000,
            completion_tokens=completion_tokens,
        )


async def run(args: argparse.Namespace) -> None:
    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    if args.tenant:
        headers["X-Tenant-ID"] = args.tenant
    payload: Dict[str, Any] = {
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": args.stream,
    }
    if args.cache_id:
        payload["kv_cache_id"] = args.cache_id
        payload["prompt_mode"] = args.prompt_mode

    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=args.concurrency,
    )
    timeout = httpx.Timeout(args.timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    wall_started = time.perf_counter()
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"), headers=headers, limits=limits, timeout=timeout
    ) as client:
        samples = await asyncio.gather(
            *(one_request(client, semaphore, payload) for _ in range(args.requests))
        )
    wall_seconds = time.perf_counter() - wall_started

    successful = [sample for sample in samples if sample.error is None]
    failed = [sample for sample in samples if sample.error is not None]
    total = [sample.total_ms for sample in successful]
    ttft = [sample.ttft_ms for sample in successful]
    completion_tokens = sum(sample.completion_tokens for sample in successful)
    report = {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "successful": len(successful),
        "failed": len(failed),
        "wall_seconds": round(wall_seconds, 3),
        "requests_per_second": round(len(successful) / wall_seconds, 3),
        "output_tokens_per_second": round(completion_tokens / wall_seconds, 3),
        "ttft_ms": {
            "median": round(statistics.median(ttft), 3) if ttft else 0,
            "p95": round(percentile(ttft, 0.95), 3),
            "p99": round(percentile(ttft, 0.99), 3),
        },
        "total_ms": {
            "median": round(statistics.median(total), 3) if total else 0,
            "p95": round(percentile(total, 0.95), 3),
            "p99": round(percentile(total, 0.99), 3),
        },
        "errors": [sample.error for sample in failed[:10]],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--prompt", default="Explain KV cache reuse in one sentence.")
    parser.add_argument("--cache-id")
    parser.add_argument("--prompt-mode", choices=("suffix", "full"), default="suffix")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--tenant", default=os.getenv("KVCACHE_TENANT_ID"))
    parser.add_argument("--api-key", default=os.getenv("KVCACHE_API_KEY"))
    parser.add_argument("--no-stream", action="store_false", dest="stream")
    parser.set_defaults(stream=True)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error("--requests and --concurrency must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
