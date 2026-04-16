"""
Notification Port

Architectural Intent:
- Contract for sending notifications (regression alerts, webhook pushes)
- Implementations: webhook, email, Slack, etc.
"""

from __future__ import annotations

from typing import Protocol


class NotificationPort(Protocol):
    async def send(self, recipient: str, subject: str, body: str) -> bool: ...
