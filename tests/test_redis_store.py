from __future__ import annotations

import unittest
from datetime import datetime, timezone

import fakeredis

from kvcache_service.domain import CacheInfo
from kvcache_service.errors import CacheNotFoundError
from kvcache_service.redis_store import RedisLogicalCacheStore


class RedisLogicalCacheStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = fakeredis.FakeRedis(decode_responses=True)
        self.store = RedisLogicalCacheStore("redis://unused", key_prefix="test", client=self.client)

    @staticmethod
    def info(cache_id: str, tenant_id: str) -> CacheInfo:
        return CacheInfo(
            cache_id=cache_id,
            backend="vllm",
            model="test/model",
            model_fingerprint="f" * 64,
            prefix_sha256="e" * 64,
            token_count=10,
            layer_count=0,
            dtype="logical",
            tensor_bytes=0,
            created_at=datetime.now(timezone.utc).isoformat(),
            chunk_size=256,
            tenant_id=tenant_id,
            storage="vllm/lmcache",
        )

    def test_shared_index_is_tenant_scoped(self) -> None:
        alpha = self.info("a" * 64, "alpha")
        beta = self.info("b" * 64, "beta")
        self.store.save(alpha, "alpha-prefix")
        self.store.save(beta, [1, 2, 3])

        self.assertEqual(self.store.get(alpha.cache_id, "alpha").prefix, "alpha-prefix")
        self.assertEqual(len(self.store.list("alpha")), 1)
        self.assertEqual(len(self.store.list(None)), 2)
        with self.assertRaises(CacheNotFoundError):
            self.store.get(alpha.cache_id, "beta")

    def test_delete_removes_entry_and_indexes(self) -> None:
        info = self.info("c" * 64, "alpha")
        self.store.save(info, "prefix")
        self.assertTrue(self.store.delete(info.cache_id, "alpha"))
        self.assertEqual(self.store.stats("alpha")["cache_count"], 0)
