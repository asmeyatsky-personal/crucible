"""
Storage Port

Architectural Intent:
- Contract for artefact blob storage (trace JSON, report files)
- Implementations: local filesystem, GCS, S3
"""

from __future__ import annotations

from typing import Protocol


class StoragePort(Protocol):
    async def store(self, key: str, data: bytes) -> str: ...
    async def retrieve(self, key: str) -> bytes | None: ...
    async def delete(self, key: str) -> bool: ...
    async def exists(self, key: str) -> bool: ...
