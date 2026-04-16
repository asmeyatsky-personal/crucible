"""
Webhook Notification Adapter

Architectural Intent:
- Implements NotificationPort via HTTP webhook
- Used for regression alerts and evaluation result push notifications
"""

from __future__ import annotations

import httpx


class WebhookNotificationAdapter:
    """Implements NotificationPort via HTTP webhook POST."""

    def __init__(self, default_url: str | None = None):
        self._default_url = default_url

    async def send(self, recipient: str, subject: str, body: str) -> bool:
        url = recipient if recipient.startswith("http") else self._default_url
        if not url:
            return False

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    url,
                    json={"subject": subject, "body": body},
                    timeout=10.0,
                )
                return response.status_code < 400
            except httpx.HTTPError:
                return False
