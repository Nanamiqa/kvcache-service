"""Redis-backed logical prefix index for horizontally scaled gateways."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional

import redis

from .domain import CacheInfo
from .errors import CacheNotFoundError
from .logical_store import LogicalCacheRecord, PrefixValue


class RedisLogicalCacheStore:
    """Shared logical metadata; Redis TTL handles expiry across gateway replicas."""

    def __init__(
        self,
        url: str,
        *,
        key_prefix: str,
        ttl_seconds: int = 0,
        client: Optional[Any] = None,
    ) -> None:
        self.client = client or redis.Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=10,
            health_check_interval=30,
            retry_on_timeout=True,
        )
        self.prefix = key_prefix.rstrip(":")
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def _tenant_hash(tenant_id: str) -> str:
        return hashlib.sha256(tenant_id.encode()).hexdigest()[:24]

    def _key(self, cache_id: str, tenant_id: str) -> str:
        return f"{self.prefix}:entry:{self._tenant_hash(tenant_id)}:{cache_id}"

    def _tenant_index(self, tenant_id: str) -> str:
        return f"{self.prefix}:tenant:{self._tenant_hash(tenant_id)}"

    def _lock_key(self, cache_id: str, tenant_id: str) -> str:
        return f"{self.prefix}:lock:{self._tenant_hash(tenant_id)}:{cache_id}"

    @property
    def _all_index(self) -> str:
        return f"{self.prefix}:all"

    @staticmethod
    def _serialize(info: CacheInfo, prefix: PrefixValue) -> str:
        return json.dumps(
            {"info": info.to_dict(), "prefix": prefix},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _deserialize(payload: str) -> LogicalCacheRecord:
        value = json.loads(payload)
        return LogicalCacheRecord(info=CacheInfo(**value["info"]), prefix=value["prefix"])

    def save(self, info: CacheInfo, prefix: PrefixValue) -> CacheInfo:
        key = self._key(info.cache_id, info.tenant_id)
        score = time.time()
        with self.client.pipeline(transaction=True) as pipeline:
            pipeline.set(key, self._serialize(info, prefix), ex=self.ttl_seconds or None)
            pipeline.zadd(self._tenant_index(info.tenant_id), {key: score})
            pipeline.zadd(self._all_index, {key: score})
            pipeline.execute()
        return info

    def get(self, cache_id: str, tenant_id: str) -> LogicalCacheRecord:
        key = self._key(cache_id, tenant_id)
        payload = self.client.get(key)
        if payload is None:
            raise CacheNotFoundError(f"Cache {cache_id!r} does not exist")
        score = time.time()
        with self.client.pipeline(transaction=False) as pipeline:
            pipeline.zadd(self._tenant_index(tenant_id), {key: score})
            pipeline.zadd(self._all_index, {key: score})
            pipeline.execute()
        return self._deserialize(payload)

    def _list_index(self, index: str, tenant_id: Optional[str]) -> List[CacheInfo]:
        keys = self.client.zrevrange(index, 0, -1)
        if not keys:
            return []
        payloads = self.client.mget(keys)
        stale = [key for key, payload in zip(keys, payloads) if payload is None]
        if stale:
            with self.client.pipeline(transaction=False) as pipeline:
                pipeline.zrem(index, *stale)
                pipeline.zrem(self._all_index, *stale)
                pipeline.execute()
        records = [self._deserialize(payload) for payload in payloads if payload is not None]
        return [
            record.info
            for record in records
            if tenant_id is None or record.info.tenant_id == tenant_id
        ]

    def list(self, tenant_id: Optional[str] = None) -> List[CacheInfo]:
        index = self._all_index if tenant_id is None else self._tenant_index(tenant_id)
        return self._list_index(index, tenant_id)

    def delete(self, cache_id: str, tenant_id: str) -> bool:
        key = self._key(cache_id, tenant_id)
        with self.client.pipeline(transaction=True) as pipeline:
            pipeline.delete(key)
            pipeline.zrem(self._tenant_index(tenant_id), key)
            pipeline.zrem(self._all_index, key)
            results = pipeline.execute()
        return bool(results[0])

    def prune(self) -> Dict[str, int]:
        before = self.client.zcard(self._all_index)
        self._list_index(self._all_index, None)
        after = self.client.zcard(self._all_index)
        return {"removed_count": max(0, before - after), "freed_bytes": 0}

    def stats(self, tenant_id: Optional[str] = None) -> Dict[str, int]:
        count = len(self.list(tenant_id))
        return {
            "cache_count": count,
            "tensor_bytes": 0,
            "disk_bytes": 0,
            "max_store_bytes": 0,
            "ttl_seconds": self.ttl_seconds,
        }

    def checkpoint(self) -> None:
        return None

    def acquire_build_lock(self, cache_id: str, tenant_id: str) -> Optional[Any]:
        lock = self.client.lock(
            self._lock_key(cache_id, tenant_id),
            timeout=600,
            blocking_timeout=30,
        )
        return lock if lock.acquire(blocking=True) else None

    @staticmethod
    def release_build_lock(lock: Any) -> None:
        try:
            lock.release()
        except redis.exceptions.LockError:
            pass

    def close(self) -> None:
        self.client.close()
