"""SQLite-backed logical prefix index for distributed inference backends."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

from .domain import CacheInfo
from .errors import CacheExpiredError, CacheNotFoundError

PrefixValue = Union[str, List[int]]


@dataclass(frozen=True)
class LogicalCacheRecord:
    info: CacheInfo
    prefix: PrefixValue


class LogicalCacheStore:
    """Durable metadata index; physical KV blocks remain owned by vLLM/LMCache."""

    def __init__(self, path: Path, ttl_seconds: int = 0) -> None:
        self.path = path.resolve()
        self.ttl_seconds = ttl_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._lock, self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS logical_caches (
                    cache_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    model TEXT NOT NULL,
                    model_fingerprint TEXT NOT NULL,
                    prefix_sha256 TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    chunk_size INTEGER NOT NULL,
                    affinity_key TEXT,
                    prefix_kind TEXT NOT NULL,
                    prefix_payload TEXT NOT NULL,
                    last_access REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_logical_caches_tenant
                    ON logical_caches (tenant_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_logical_caches_expiry
                    ON logical_caches (expires_at);
                """
            )

    @staticmethod
    def _expired(expires_at: Optional[str]) -> bool:
        if not expires_at:
            return False
        value = datetime.fromisoformat(expires_at)
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) >= value

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> LogicalCacheRecord:
        prefix = json.loads(row["prefix_payload"])
        info = CacheInfo(
            cache_id=row["cache_id"],
            backend=row["backend"],
            model=row["model"],
            model_fingerprint=row["model_fingerprint"],
            prefix_sha256=row["prefix_sha256"],
            token_count=row["token_count"],
            layer_count=0,
            dtype="logical",
            tensor_bytes=0,
            created_at=row["created_at"],
            chunk_size=row["chunk_size"],
            expires_at=row["expires_at"],
            tenant_id=row["tenant_id"],
            affinity_key=row["affinity_key"],
            storage="vllm/lmcache",
        )
        return LogicalCacheRecord(info=info, prefix=prefix)

    def save(self, info: CacheInfo, prefix: PrefixValue) -> CacheInfo:
        kind = "text" if isinstance(prefix, str) else "tokens"
        payload = json.dumps(prefix, ensure_ascii=False, separators=(",", ":"))
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO logical_caches (
                    cache_id, tenant_id, backend, model, model_fingerprint,
                    prefix_sha256, token_count, created_at, expires_at,
                    chunk_size, affinity_key, prefix_kind, prefix_payload, last_access
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_id) DO UPDATE SET
                    last_access=excluded.last_access,
                    token_count=excluded.token_count,
                    affinity_key=excluded.affinity_key
                """,
                (
                    info.cache_id,
                    info.tenant_id,
                    info.backend,
                    info.model,
                    info.model_fingerprint,
                    info.prefix_sha256,
                    info.token_count,
                    info.created_at,
                    info.expires_at,
                    info.chunk_size,
                    info.affinity_key,
                    kind,
                    payload,
                    time.time(),
                ),
            )
        return info

    def get(self, cache_id: str, tenant_id: str) -> LogicalCacheRecord:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM logical_caches WHERE cache_id=? AND tenant_id=?",
                (cache_id, tenant_id),
            ).fetchone()
            if row is None:
                raise CacheNotFoundError(f"Cache {cache_id!r} does not exist")
            if self._expired(row["expires_at"]):
                connection.execute("DELETE FROM logical_caches WHERE cache_id=?", (cache_id,))
                raise CacheExpiredError(f"Cache {cache_id!r} has expired")
            connection.execute(
                "UPDATE logical_caches SET last_access=? WHERE cache_id=?",
                (time.time(), cache_id),
            )
            return self._row_to_record(row)

    def list(self, tenant_id: Optional[str] = None) -> List[CacheInfo]:
        self.prune()
        with self._lock, self._connection() as connection:
            if tenant_id is None:
                rows = connection.execute(
                    "SELECT * FROM logical_caches ORDER BY created_at DESC"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM logical_caches WHERE tenant_id=? ORDER BY created_at DESC",
                    (tenant_id,),
                ).fetchall()
        return [self._row_to_record(row).info for row in rows]

    def delete(self, cache_id: str, tenant_id: str) -> bool:
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM logical_caches WHERE cache_id=? AND tenant_id=?",
                (cache_id, tenant_id),
            )
            return cursor.rowcount > 0

    def prune(self) -> Dict[str, int]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM logical_caches WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
            removed = max(cursor.rowcount, 0)
        return {"removed_count": removed, "freed_bytes": 0}

    def stats(self, tenant_id: Optional[str] = None) -> Dict[str, int]:
        self.prune()
        with self._lock, self._connection() as connection:
            if tenant_id is None:
                count = connection.execute("SELECT COUNT(*) FROM logical_caches").fetchone()[0]
            else:
                count = connection.execute(
                    "SELECT COUNT(*) FROM logical_caches WHERE tenant_id=?", (tenant_id,)
                ).fetchone()[0]
        disk_bytes = self.path.stat().st_size if self.path.exists() else 0
        return {
            "cache_count": int(count),
            "tensor_bytes": 0,
            "disk_bytes": disk_bytes,
            "max_store_bytes": 0,
            "ttl_seconds": self.ttl_seconds,
        }

    def checkpoint(self) -> None:
        with self._lock, self._connection() as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
