"""Atomic, checksum-verified safetensors cache storage."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from safetensors.torch import load_file, save_file

from .domain import CacheInfo
from .errors import CacheNotFoundError, KVCacheError

_CACHE_ID = re.compile(r"^[a-f0-9]{64}$")
_INFO_FIELDS = tuple(CacheInfo.__dataclass_fields__)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SafeTensorCacheStore:
    """One directory per immutable, content-addressed cache artifact."""

    schema_version = 1

    def __init__(self, root: Path, verify_checksum: bool = True) -> None:
        self.root = root.resolve()
        self.verify_checksum = verify_checksum
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_id(cache_id: str) -> None:
        if not _CACHE_ID.fullmatch(cache_id):
            raise CacheNotFoundError(f"Invalid cache id: {cache_id!r}")

    def _paths(self, cache_id: str) -> Tuple[Path, Path, Path]:
        self.validate_id(cache_id)
        directory = self.root / cache_id
        return directory, directory / "metadata.json", directory / "tensors.safetensors"

    def exists(self, cache_id: str) -> bool:
        directory, metadata_path, tensor_path = self._paths(cache_id)
        return directory.is_dir() and metadata_path.is_file() and tensor_path.is_file()

    def save(self, info: CacheInfo, tensors: Mapping[str, Any]) -> CacheInfo:
        """Write an immutable artifact and atomically publish its directory."""

        destination, _, _ = self._paths(info.cache_id)
        with self._lock:
            if self.exists(info.cache_id):
                return self.get_info(info.cache_id)

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
                    if not self.exists(info.cache_id):
                        raise
                return self.get_info(info.cache_id)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)

    def _metadata(self, cache_id: str) -> Dict[str, Any]:
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
        return payload

    def get_info(self, cache_id: str) -> CacheInfo:
        payload = self._metadata(cache_id)
        try:
            return CacheInfo(**{field: payload[field] for field in _INFO_FIELDS})
        except (KeyError, TypeError) as exc:
            raise KVCacheError(f"Cache metadata is incomplete for {cache_id!r}") from exc

    def load(self, cache_id: str) -> Tuple[CacheInfo, Dict[str, Any]]:
        _, _, tensor_path = self._paths(cache_id)
        payload = self._metadata(cache_id)
        if not tensor_path.is_file():
            raise CacheNotFoundError(f"Tensor file for cache {cache_id!r} does not exist")
        if self.verify_checksum and _sha256_file(tensor_path) != payload.get("tensor_sha256"):
            raise KVCacheError(f"Checksum validation failed for cache {cache_id!r}")
        return self.get_info(cache_id), load_file(str(tensor_path), device="cpu")

    def list(self) -> List[CacheInfo]:
        infos: List[CacheInfo] = []
        with self._lock:
            for candidate in sorted(self.root.iterdir()):
                if not candidate.is_dir() or not _CACHE_ID.fullmatch(candidate.name):
                    continue
                try:
                    infos.append(self.get_info(candidate.name))
                except KVCacheError:
                    continue
        return sorted(infos, key=lambda item: item.created_at, reverse=True)

    def delete(self, cache_id: str) -> bool:
        directory, _, _ = self._paths(cache_id)
        with self._lock:
            if not directory.exists():
                return False
            shutil.rmtree(directory)
            return True
