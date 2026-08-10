from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

import httpx

from kvcache_service.config import Settings
from kvcache_service.domain import BuildCacheCommand, CompletionCommand
from kvcache_service.errors import CacheNotFoundError
from kvcache_service.vllm_backend import VLLMBackend


class VLLMBackendTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.requests: List[Dict[str, Any]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET" and request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            if request.method == "GET" and request.url.path == "/healthcheck":
                return httpx.Response(200, json={"status": "healthy"})
            payload = json.loads(request.content)
            self.requests.append(payload)
            if payload.get("stream"):
                body = (
                    'data: {"model":"test/model","choices":[{"text":"o",'
                    '"finish_reason":null}]}\n\n'
                    'data: {"model":"test/model","choices":[{"text":"k",'
                    '"finish_reason":"stop"}],"usage":{"prompt_tokens":12,'
                    '"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":8}}}\n\n'
                    "data: [DONE]\n\n"
                )
                return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
            prompt = payload["prompt"]
            prompt_tokens = len(prompt) if isinstance(prompt, list) else len(prompt)
            return httpx.Response(
                200,
                json={
                    "model": "test/model",
                    "choices": [{"text": "done", "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": 1,
                        "prompt_tokens_details": {"cached_tokens": 8},
                    },
                },
            )

        self.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        settings = Settings(
            backend="vllm",
            model_id="test/model",
            model_revision="immutable",
            store_dir=Path(self.temporary.name),
            vllm_endpoints=("http://replica-1", "http://replica-2"),
            lmcache_controller_url="http://lmcache:8080",
        )
        self.backend = VLLMBackend(settings, client=self.client)

    async def asyncTearDown(self) -> None:
        await self.backend.aclose()
        await self.client.aclose()
        self.temporary.cleanup()

    @staticmethod
    def completion(cache_id: str, tenant_id: str = "alpha") -> CompletionCommand:
        return CompletionCommand(
            cache_id=cache_id,
            prompt="question",
            input_ids=None,
            max_tokens=4,
            temperature=0,
            top_p=1,
            top_k=0,
            seed=1,
            stop=[],
            stop_token_ids=[],
            tenant_id=tenant_id,
        )

    async def test_warm_and_reuse_logical_prefix(self) -> None:
        info = await self.backend.abuild_cache(
            BuildCacheCommand(text="prefix:", input_ids=None, chunk_size=256, tenant_id="alpha")
        )
        result = await self.backend.acomplete(self.completion(info.cache_id))

        self.assertEqual(info.storage.split("@")[0], "vllm/lmcache")
        self.assertEqual(result.text, "done")
        self.assertEqual(result.cached_tokens, 8)
        self.assertEqual(result.completion_tokens, 1)
        self.assertEqual(self.requests[-1]["prompt"], "prefix:question")
        with self.assertRaises(CacheNotFoundError):
            await self.backend.aget_cache(info.cache_id, "beta")

    async def test_streaming_propagates_usage_and_finish_reason(self) -> None:
        info = await self.backend.abuild_cache(
            BuildCacheCommand(text="prefix:", input_ids=None, chunk_size=256, tenant_id="alpha")
        )
        stream = self.backend.astream_complete(self.completion(info.cache_id))
        chunks = [chunk async for chunk in stream]
        self.assertEqual("".join(chunk.text for chunk in chunks), "ok")
        self.assertEqual(chunks[-1].finish_reason, "stop")
        self.assertEqual(chunks[-1].cached_tokens, 8)

    async def test_health_probes_all_replicas(self) -> None:
        health = await self.backend.ahealth()
        self.assertEqual(health["healthy_replicas"], 2)
        self.assertEqual(health["status"], "ok")
        self.assertTrue(health["lmcache"]["healthy"])
