"""
Local Storage Adapter

Architectural Intent:
- Implements StoragePort using local filesystem
- Development/testing adapter — production would use GCS or S3
"""

from __future__ import annotations

import os
from pathlib import Path


class LocalStorageAdapter:
    """Implements StoragePort using local filesystem."""

    def __init__(self, base_path: str = "./storage"):
        self._base = Path(base_path)
        self._base.mkdir(parents=True, exist_ok=True)

    async def store(self, key: str, data: bytes) -> str:
        path = self._base / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    async def retrieve(self, key: str) -> bytes | None:
        path = self._base / key
        if path.exists():
            return path.read_bytes()
        return None

    async def delete(self, key: str) -> bool:
        path = self._base / key
        if path.exists():
            path.unlink()
            return True
        return False

    async def exists(self, key: str) -> bool:
        return (self._base / key).exists()
