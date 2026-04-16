"""
Tests for WebhookNotificationAdapter.

Uses mock httpx.AsyncClient to avoid real HTTP calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from infrastructure.adapters.webhook_notification_adapter import WebhookNotificationAdapter


class TestWebhookNotificationAdapter:
    @pytest.mark.asyncio
    async def test_send_with_http_url_as_recipient(self):
        adapter = WebhookNotificationAdapter()

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("infrastructure.adapters.webhook_notification_adapter.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await adapter.send(
                recipient="https://hooks.example.com/webhook",
                subject="Test Alert",
                body="Something happened",
            )

        assert result is True
        mock_client.post.assert_called_once_with(
            "https://hooks.example.com/webhook",
            json={"subject": "Test Alert", "body": "Something happened"},
            timeout=10.0,
        )

    @pytest.mark.asyncio
    async def test_send_uses_default_url_for_non_http_recipient(self):
        adapter = WebhookNotificationAdapter(default_url="https://default.example.com/hook")

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("infrastructure.adapters.webhook_notification_adapter.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await adapter.send(
                recipient="user@example.com",
                subject="Alert",
                body="Body",
            )

        assert result is True
        mock_client.post.assert_called_once_with(
            "https://default.example.com/hook",
            json={"subject": "Alert", "body": "Body"},
            timeout=10.0,
        )

    @pytest.mark.asyncio
    async def test_send_returns_false_when_no_url(self):
        adapter = WebhookNotificationAdapter()  # no default_url

        result = await adapter.send(
            recipient="user@example.com",
            subject="Alert",
            body="Body",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_send_returns_false_on_4xx_status(self):
        adapter = WebhookNotificationAdapter()

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("infrastructure.adapters.webhook_notification_adapter.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await adapter.send(
                recipient="https://hooks.example.com/webhook",
                subject="Test",
                body="Body",
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_send_returns_true_on_2xx_status(self):
        adapter = WebhookNotificationAdapter()

        mock_response = MagicMock()
        mock_response.status_code = 201

        with patch("infrastructure.adapters.webhook_notification_adapter.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await adapter.send(
                recipient="https://hooks.example.com/webhook",
                subject="Test",
                body="Body",
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_send_returns_false_on_http_error(self):
        adapter = WebhookNotificationAdapter()

        with patch("infrastructure.adapters.webhook_notification_adapter.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.HTTPError("Connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await adapter.send(
                recipient="https://hooks.example.com/webhook",
                subject="Test",
                body="Body",
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_send_returns_true_on_3xx_status(self):
        adapter = WebhookNotificationAdapter()

        mock_response = MagicMock()
        mock_response.status_code = 301

        with patch("infrastructure.adapters.webhook_notification_adapter.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await adapter.send(
                recipient="https://hooks.example.com/webhook",
                subject="Test",
                body="Body",
            )

        assert result is True  # 301 < 400
