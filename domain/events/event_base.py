"""
Domain Event Base

Architectural Intent:
- Foundation for all domain events in the CRUCIBLE evaluation harness
- Events are immutable value objects capturing facts about state transitions
- Collected by aggregates, dispatched by application layer — never fired inline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4


@dataclass(frozen=True)
class DomainEvent:
    """Base class for all domain events. Immutable and timestamped."""

    aggregate_id: str
    event_id: str = field(default_factory=lambda: str(uuid4()))
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
