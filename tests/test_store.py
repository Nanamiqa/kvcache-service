from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from kvcache_service.domain import CacheInfo
from kvcache_service.errors import CacheExpiredError, KVCacheError
from kvcache_service.store import SafeTensorCacheStore


class StoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.torch = torch

    @staticmethod
    def info(cache_id: str) -> CacheInfo:
        return CacheInfo(
            cache_id=cache_id,
            backend="test",
            model="test/model",
            model_fingerprint="f" * 64,
            prefix_sha256="e" * 64,
            token_count=4,
            layer_count=1,
            dtype="float32",
            tensor_bytes=4096,
            created_at=datetime.now(timezone.utc).isoformat(),
            chunk_size=2,
        )

    def tensors(self):
        return {"layer.0.key": self.torch.ones(1024)}

    def test_checksum_detects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SafeTensorCacheStore(Path(directory))
            info = self.info("a" * 64)
            store.save(info, self.tensors())
            tensor_path = Path(directory) / info.cache_id / "tensors.safetensors"
            with tensor_path.open("ab") as handle:
                handle.write(b"corrupt")
            with self.assertRaises(KVCacheError):
                store.load(info.cache_id)

    def test_prune_removes_expired_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SafeTensorCacheStore(Path(directory))
            info = self.info("b" * 64)
            store.save(info, self.tensors())
            metadata_path = Path(directory) / info.cache_id / "metadata.json"
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            payload["expires_at"] = "2000-01-01T00:00:00+00:00"
            metadata_path.write_text(json.dumps(payload), encoding="utf-8")
            result = store.prune()
            self.assertEqual(result["removed_count"], 1)
            self.assertFalse((Path(directory) / info.cache_id).exists())

    def test_ttl_assigns_expiration_on_save(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SafeTensorCacheStore(Path(directory), ttl_seconds=60)
            saved = store.save(self.info("e" * 64), self.tensors())
            self.assertIsNotNone(saved.expires_at)

    def test_expired_read_removes_artifact_and_returns_gone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SafeTensorCacheStore(Path(directory))
            info = self.info("f" * 64)
            store.save(info, self.tensors())
            metadata_path = Path(directory) / info.cache_id / "metadata.json"
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            payload["expires_at"] = "2000-01-01T00:00:00+00:00"
            metadata_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(CacheExpiredError):
                store.get_info(info.cache_id)
            self.assertFalse((Path(directory) / info.cache_id).exists())

    def test_quota_prunes_least_recently_used_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SafeTensorCacheStore(Path(directory))
            first = self.info("c" * 64)
            store.save(first, self.tensors())
            first_size = store.stats()["disk_bytes"]
            store.max_store_bytes = first_size + 100
            second = self.info("d" * 64)
            store.save(second, self.tensors())
            self.assertFalse(store.exists(first.cache_id))
            self.assertTrue(store.exists(second.cache_id))


if __name__ == "__main__":
    unittest.main()
