"""Tests for presentation/api/app.py — create_app and get_container."""

from __future__ import annotations

import pytest
import httpx

from infrastructure.config.settings import Settings
from presentation.api.app import create_app, get_container
import presentation.api.app as app_module


class TestCreateApp:
    async def test_create_app_returns_fastapi_instance(self, tmp_path):
        db_path = str(tmp_path / "app_test.db")
        settings = Settings(
            database_url=f"sqlite+aiosqlite:///{db_path}",
            storage_path=str(tmp_path / "storage"),
            anthropic_api_key="test-key",
        )
        app = create_app(settings)
        assert app.title == "CRUCIBLE\u2122"
        assert app.version == "1.0.0"

    async def test_create_app_health_endpoint_via_lifespan(self, tmp_path):
        """Test the actual create_app with its lifespan (init/shutdown)."""
        db_path = str(tmp_path / "lifespan_test.db")
        settings = Settings(
            database_url=f"sqlite+aiosqlite:///{db_path}",
            storage_path=str(tmp_path / "storage"),
            anthropic_api_key="test-key",
        )
        app = create_app(settings)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"

    async def test_create_app_registers_all_route_prefixes(self, tmp_path):
        db_path = str(tmp_path / "routes_test.db")
        settings = Settings(
            database_url=f"sqlite+aiosqlite:///{db_path}",
            storage_path=str(tmp_path / "storage"),
            anthropic_api_key="test-key",
        )
        app = create_app(settings)
        route_paths = [r.path for r in app.routes]
        # Check all expected prefixes exist via the openapi schema
        assert any("/api/v1/agents" in str(p) for p in route_paths)
        assert any("/api/v1/traces" in str(p) for p in route_paths)
        assert any("/api/v1/rubrics" in str(p) for p in route_paths)
        assert any("/api/v1/evaluations" in str(p) for p in route_paths)
        assert any("/api/v1/reports" in str(p) for p in route_paths)


class TestGetContainer:
    def test_get_container_raises_when_not_initialised(self):
        original = app_module.container
        try:
            app_module.container = None
            with pytest.raises(RuntimeError, match="Container not initialised"):
                get_container()
        finally:
            app_module.container = original

    async def test_get_container_returns_container_after_create_app(self, tmp_path):
        db_path = str(tmp_path / "container_test.db")
        settings = Settings(
            database_url=f"sqlite+aiosqlite:///{db_path}",
            storage_path=str(tmp_path / "storage"),
            anthropic_api_key="test-key",
        )
        create_app(settings)
        container = get_container()
        assert container is not None
        assert container.settings == settings
