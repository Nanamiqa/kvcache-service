"""Atomic, checksum-verified safetensors cache storage."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import MISSING, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

from filelock import FileLock
from safetensors.torch import load_file, save_file

from .domain import CacheInfo
from .errors import CacheExpiredError, CacheNotFoundError, KVCacheError

_CACHE_ID = re.compile(r"^[a-f0-9]{64}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SafeTensorCacheStore:
    """One directory per immutable, content-addressed cache artifact."""

    schema_version = 1

    def __init__(
        self,
        root: Path,
        verify_checksum: bool = True,
        ttl_seconds: int = 0,
        max_store_bytes: int = 0,
    ) -> None:
        self.root = root.resolve()
        self.verify_checksum = verify_checksum
        self.ttl_seconds = ttl_seconds
        self.max_store_bytes = max_store_bytes
        self._thread_lock = threading.RLock()
        self._verified_files: Dict[str, Tuple[int, int, str]] = {}
        self.root.mkdir(parents=True, exist_ok=True)
        self._file_lock = FileLock(str(self.root / ".store.lock"))

    @contextmanager
    def _guard(self) -> Iterator[None]:
        with self._thread_lock:
            with self._file_lock:
                yield

    @staticmethod
    def validate_id(cache_id: str) -> None:
        if not _CACHE_ID.fullmatch(cache_id):
            raise CacheNotFoundError(f"Invalid cache id: {cache_id!r}")

    def _paths(self, cache_id: str) -> Tuple[Path, Path, Path]:
        self.validate_id(cache_id)
        directory = self.root / cache_id
        return directory, directory / "metadata.json", directory / "tensors.safetensors"

    def exists(self, cache_id: str) -> bool:
        with self._guard():
            directory, metadata_path, tensor_path = self._paths(cache_id)
            if not (directory.is_dir() and metadata_path.is_file() and tensor_path.is_file()):
                return False
            try:
                self._metadata_unlocked(cache_id)
            except KVCacheError:
                return False
            return True

    def save(self, info: CacheInfo, tensors: Mapping[str, Any]) -> CacheInfo:
        """Write an immutable artifact and atomically publish its directory."""

        if self.ttl_seconds > 0 and info.expires_at is None:
            expires = datetime.now(timezone.utc) + timedelta(seconds=self.ttl_seconds)
            info = replace(info, expires_at=expires.isoformat())
        destination, _, _ = self._paths(info.cache_id)
        with self._guard():
            if destination.exists():
                try:
                    return self._info_from_payload(self._metadata_unlocked(info.cache_id))
                except KVCacheError:
                    shutil.rmtree(destination)
                    self._verified_files.pop(info.cache_id, None)

            temporary = Path(tempfile.mkdtemp(prefix=f".{info.cache_id}.", dir=str(self.root)))
            try:
                tensor_path = temporary / "tensors.safetensors"
                normalized = {
                    name: tensor.detach().to("cpu").contiguous() for name, tensor in tensors.items()
                }
                save_file(normalized, str(tensor_path))
                payload = info.to_dict()
                payload.update(
                    {
                        "schema_version": self.schema_version,
                        "tensor_sha256": _sha256_file(tensor_path),
                    }
                )
                metadata_path = temporary / "metadata.json"
                metadata_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                try:
                    os.replace(str(temporary), str(destination))
                except OSError:
                    try:
                        self._metadata_unlocked(info.cache_id)
                    except KVCacheError:
                        raise
                published_payload = self._metadata_unlocked(info.cache_id)
                saved = self._info_from_payload(published_payload)
                published_tensor = destination / "tensors.safetensors"
                published_stat = published_tensor.stat()
                self._verified_files[info.cache_id] = (
                    published_stat.st_size,
                    published_stat.st_mtime_ns,
                    published_payload["tensor_sha256"],
                )
                self._prune_unlocked(protected_id=info.cache_id)
                return saved
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)

    def _metadata_unlocked(self, cache_id: str) -> Dict[str, Any]:
        _, metadata_path, _ = self._paths(cache_id)
        if not metadata_path.is_file():
            raise CacheNotFoundError(f"Cache {cache_id!r} does not exist")
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KVCacheError(f"Cache metadata is unreadable for {cache_id!r}") from exc
        if payload.get("schema_version") != self.schema_version:
            raise KVCacheError(
                f"Unsupported cache schema {payload.get('schema_version')!r} for {cache_id!r}"
            )
        expires_at = payload.get("expires_at")
        if not expires_at and self.ttl_seconds > 0 and payload.get("created_at"):
            try:
                created = datetime.fromisoformat(payload["created_at"])
            except ValueError as exc:
                raise KVCacheError(f"Cache creation time is invalid for {cache_id!r}") from exc
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            expires_at = (created + timedelta(seconds=self.ttl_seconds)).isoformat()
            payload["expires_at"] = expires_at
        if expires_at:
            try:
                expires = datetime.fromisoformat(expires_at)
            except ValueError as exc:
                raise KVCacheError(f"Cache expiry is invalid for {cache_id!r}") from exc
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) >= expires:
                raise CacheExpiredError(f"Cache {cache_id!r} has expired")
        return payload

    @staticmethod
    def _info_from_payload(payload: Mapping[str, Any]) -> CacheInfo:
        values: Dict[str, Any] = {}
        for name, definition in CacheInfo.__dataclass_fields__.items():
            if name in payload:
                values[name] = payload[name]
            elif definition.default is not MISSING:
                values[name] = definition.default
            else:
                raise KVCacheError(f"Cache metadata is missing required field {name!r}")
        try:
            return CacheInfo(**values)
        except TypeError as exc:
            raise KVCacheError("Cache metadata contains invalid values") from exc

    def get_info(self, cache_id: str) -> CacheInfo:
        with self._guard():
            try:
                payload = self._metadata_unlocked(cache_id)
            except CacheExpiredError:
                directory, _, _ = self._paths(cache_id)
                if directory.exists():
                    shutil.rmtree(directory)
                self._verified_files.pop(cache_id, None)
                raise
            return self._info_from_payload(payload)

    def load(self, cache_id: str) -> Tuple[CacheInfo, Dict[str, Any]]:
        with self._guard():
            directory, _, tensor_path = self._paths(cache_id)
            try:
                payload = self._metadata_unlocked(cache_id)
            except CacheExpiredError:
                if directory.exists():
                    shutil.rmtree(directory)
                self._verified_files.pop(cache_id, None)
                raise
            if not tensor_path.is_file():
                raise CacheNotFoundError(f"Tensor file for cache {cache_id!r} does not exist")
            if self.verify_checksum:
                expected_checksum = payload.get("tensor_sha256")
                if not isinstance(expected_checksum, str) or not re.fullmatch(
                    r"[a-f0-9]{64}", expected_checksum
                ):
                    raise KVCacheError(f"Tensor checksum is missing for cache {cache_id!r}")
                stat = tensor_path.stat()
                signature = (stat.st_size, stat.st_mtime_ns, expected_checksum)
                if self._verified_files.get(cache_id) != signature:
                    if _sha256_file(tensor_path) != expected_checksum:
                        raise KVCacheError(f"Checksum validation failed for cache {cache_id!r}")
                    self._verified_files[cache_id] = signature
            tensors = load_file(str(tensor_path), device="cpu")
            os.utime(directory, None)
            return self._info_from_payload(payload), tensors

    def list(self) -> List[CacheInfo]:
        self.prune()
        infos: List[CacheInfo] = []
        with self._guard():
            for candidate in sorted(self.root.iterdir()):
                if not candidate.is_dir() or not _CACHE_ID.fullmatch(candidate.name):
                    continue
                try:
                    infos.append(self._info_from_payload(self._metadata_unlocked(candidate.name)))
                except KVCacheError:
                    continue
        return sorted(infos, key=lambda item: item.created_at, reverse=True)

    def delete(self, cache_id: str) -> bool:
        directory, _, _ = self._paths(cache_id)
        with self._guard():
            if not directory.exists():
                return False
            shutil.rmtree(directory)
            self._verified_files.pop(cache_id, None)
            return True

    @staticmethod
    def _directory_bytes(directory: Path) -> int:
        return sum(path.stat().st_size for path in directory.iterdir() if path.is_file())

    def _prune_unlocked(self, protected_id: Optional[str] = None) -> Dict[str, int]:
        removed_count = 0
        freed_bytes = 0
        candidates = []
        for directory in self.root.iterdir():
            if not directory.is_dir() or not _CACHE_ID.fullmatch(directory.name):
                continue
            size = self._directory_bytes(directory)
            try:
                self._metadata_unlocked(directory.name)
            except KVCacheError:
                if directory.name == protected_id:
                    continue
                shutil.rmtree(directory)
                self._verified_files.pop(directory.name, None)
                removed_count += 1
                freed_bytes += size
                continue
            candidates.append((directory.stat().st_mtime_ns, directory, size))

        if self.max_store_bytes > 0:
            total = sum(size for _, _, size in candidates)
            for _, directory, size in sorted(candidates):
                if total <= self.max_store_bytes:
                    break
                if directory.name == protected_id:
                    continue
                shutil.rmtree(directory)
                self._verified_files.pop(directory.name, None)
                total -= size
                removed_count += 1
                freed_bytes += size
        return {"removed_count": removed_count, "freed_bytes": freed_bytes}

    def prune(self) -> Dict[str, int]:
        with self._guard():
            return self._prune_unlocked()

    def _stats_unlocked(self) -> Dict[str, int]:
        cache_count = 0
        tensor_bytes = 0
        disk_bytes = 0
        for directory in self.root.iterdir():
            if not directory.is_dir() or not _CACHE_ID.fullmatch(directory.name):
                continue
            try:
                info = self._info_from_payload(self._metadata_unlocked(directory.name))
            except KVCacheError:
                continue
            cache_count += 1
            tensor_bytes += info.tensor_bytes
            disk_bytes += self._directory_bytes(directory)
        return {
            "cache_count": cache_count,
            "tensor_bytes": tensor_bytes,
            "disk_bytes": disk_bytes,
            "max_store_bytes": self.max_store_bytes,
            "ttl_seconds": self.ttl_seconds,
        }

    def stats(self) -> Dict[str, int]:
        self.prune()
        with self._guard():
            return self._stats_unlocked()
