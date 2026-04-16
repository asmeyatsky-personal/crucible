"""
In-Memory Event Bus Adapter

Architectural Intent:
- Simple in-memory implementation of EventBusPort for Phase 1
- Supports synchronous handler dispatch within the same process
- Production would use a message broker (Pub/Sub, RabbitMQ, etc.)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable

from domain.events.event_base import DomainEvent

logger = logging.getLogger(__name__)


class InMemoryEventBusAdapter:
    """Implements EventBusPort with in-memory handler dispatch."""

    def __init__(self) -> None:
        self._handlers: dict[type, list[Callable]] = defaultdict(list)

    async def publish(self, events: list[DomainEvent]) -> None:
        for event in events:
            event_type = type(event)
            handlers = self._handlers.get(event_type, [])
            for handler in handlers:
                try:
                    await handler(event)
                except Exception:
                    logger.exception(
                        "Event handler failed for %s", event_type.__name__,
                    )

    async def subscribe(
        self, event_type: type[DomainEvent], handler: Callable,
    ) -> None:
        self._handlers[event_type].append(handler)
