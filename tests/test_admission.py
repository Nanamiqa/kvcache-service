from __future__ import annotations

import asyncio
import unittest

from kvcache_service.admission import AdmissionController
from kvcache_service.config import Settings
from kvcache_service.errors import RequestOverloadedError


class AdmissionControllerTest(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_when_bounded_queue_is_full(self) -> None:
        controller = AdmissionController(
            Settings(
                max_concurrent_requests=1,
                max_concurrent_per_tenant=1,
                admission_queue_timeout_seconds=0.01,
            )
        )
        entered = asyncio.Event()
        release = asyncio.Event()

        async def hold_slot() -> None:
            async with controller.admit("alpha", 1):
                entered.set()
                await release.wait()

        task = asyncio.create_task(hold_slot())
        await entered.wait()
        with self.assertRaises(RequestOverloadedError):
            async with controller.admit("alpha", 1):
                pass
        release.set()
        await task

    async def test_enforces_tenant_token_bucket(self) -> None:
        controller = AdmissionController(
            Settings(rate_limit_tokens_per_minute=10, rate_limit_burst_tokens=10)
        )
        async with controller.admit("alpha", 10):
            pass
        with self.assertRaises(RequestOverloadedError):
            async with controller.admit("alpha", 1):
                pass
