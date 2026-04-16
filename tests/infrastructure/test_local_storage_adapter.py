"""
Tests for LocalStorageAdapter.

Uses pytest tmp_path fixture for filesystem isolation.
"""

from __future__ import annotations

import pytest

from infrastructure.adapters.local_storage_adapter import LocalStorageAdapter


class TestLocalStorageAdapter:
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, tmp_path):
        adapter = LocalStorageAdapter(base_path=str(tmp_path))

        await adapter.store("test.txt", b"hello world")
        result = await adapter.retrieve("test.txt")

        assert result == b"hello world"

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_returns_none(self, tmp_path):
        adapter = LocalStorageAdapter(base_path=str(tmp_path))

        result = await adapter.retrieve("does-not-exist.txt")

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_existing_file(self, tmp_path):
        adapter = LocalStorageAdapter(base_path=str(tmp_path))

        await adapter.store("to-delete.txt", b"data")
        result = await adapter.delete("to-delete.txt")

        assert result is True
        assert await adapter.retrieve("to-delete.txt") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(self, tmp_path):
        adapter = LocalStorageAdapter(base_path=str(tmp_path))

        result = await adapter.delete("nope.txt")

        assert result is False

    @pytest.mark.asyncio
    async def test_exists_true(self, tmp_path):
        adapter = LocalStorageAdapter(base_path=str(tmp_path))

        await adapter.store("exists.txt", b"data")
        result = await adapter.exists("exists.txt")

        assert result is True

    @pytest.mark.asyncio
    async def test_exists_false(self, tmp_path):
        adapter = LocalStorageAdapter(base_path=str(tmp_path))

        result = await adapter.exists("nope.txt")

        assert result is False

    @pytest.mark.asyncio
    async def test_store_creates_subdirectories(self, tmp_path):
        adapter = LocalStorageAdapter(base_path=str(tmp_path))

        path = await adapter.store("sub/dir/file.bin", b"\x00\x01\x02")
        result = await adapter.retrieve("sub/dir/file.bin")

        assert result == b"\x00\x01\x02"
        assert "sub/dir/file.bin" in path

    @pytest.mark.asyncio
    async def test_store_returns_path_string(self, tmp_path):
        adapter = LocalStorageAdapter(base_path=str(tmp_path))

        path = await adapter.store("output.dat", b"binary data")

        assert isinstance(path, str)
        assert "output.dat" in path

    @pytest.mark.asyncio
    async def test_overwrite_existing_file(self, tmp_path):
        adapter = LocalStorageAdapter(base_path=str(tmp_path))

        await adapter.store("file.txt", b"original")
        await adapter.store("file.txt", b"updated")

        result = await adapter.retrieve("file.txt")
        assert result == b"updated"

    @pytest.mark.asyncio
    async def test_store_empty_data(self, tmp_path):
        adapter = LocalStorageAdapter(base_path=str(tmp_path))

        await adapter.store("empty.txt", b"")
        result = await adapter.retrieve("empty.txt")

        assert result == b""

    @pytest.mark.asyncio
    async def test_delete_then_exists(self, tmp_path):
        adapter = LocalStorageAdapter(base_path=str(tmp_path))

        await adapter.store("temp.txt", b"temp data")
        assert await adapter.exists("temp.txt") is True

        await adapter.delete("temp.txt")
        assert await adapter.exists("temp.txt") is False

    @pytest.mark.asyncio
    async def test_creates_base_directory(self, tmp_path):
        new_base = tmp_path / "new_storage_dir"
        adapter = LocalStorageAdapter(base_path=str(new_base))

        assert new_base.exists()
