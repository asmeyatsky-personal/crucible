"""
Event Bus Port

Architectural Intent:
- Contract for publishing domain events for cross-boundary communication
- Application layer dispatches collected events after use case completion
"""

from __future__ import annotations

from typing import Callable, Protocol

from domain.events.event_base import DomainEvent


class EventBusPort(Protocol):
    async def publish(self, events: list[DomainEvent]) -> None: ...
    async def subscribe(
        self, event_type: type[DomainEvent], handler: Callable,
    ) -> None: ...
