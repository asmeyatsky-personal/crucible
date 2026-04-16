"""
Agent Entity

Architectural Intent:
- Aggregate root representing a registered agent identity
- Agents have rubric assignments and evaluation history
- All state changes produce new instances (immutable)
- Domain events collected for cross-context communication
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from domain.events.event_base import DomainEvent


@dataclass(frozen=True)
class AgentRegisteredEvent(DomainEvent):
    agent_name: str = ""
    agent_type: str = ""


@dataclass(frozen=True)
class RubricAssignedEvent(DomainEvent):
    rubric_id: str = ""


@dataclass(frozen=True)
class Agent:
    """Registered agent identity within CRUCIBLE."""

    id: str
    name: str
    description: str
    agent_type: str  # e.g., "research", "coding", "customer_support", "data_extraction"
    rubric_ids: tuple[str, ...] = field(default=())
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict = field(default_factory=dict)
    domain_events: tuple[DomainEvent, ...] = field(default=())

    @staticmethod
    def register(
        id: str, name: str, description: str, agent_type: str,
        metadata: dict | None = None,
    ) -> Agent:
        """Factory method to register a new agent."""
        now = datetime.now(UTC)
        return Agent(
            id=id,
            name=name,
            description=description,
            agent_type=agent_type,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
            domain_events=(
                AgentRegisteredEvent(
                    aggregate_id=id, agent_name=name, agent_type=agent_type,
                ),
            ),
        )

    def assign_rubric(self, rubric_id: str) -> Agent:
        """Assign a rubric to this agent. Returns new instance."""
        if rubric_id in self.rubric_ids:
            return self
        return replace(
            self,
            rubric_ids=self.rubric_ids + (rubric_id,),
            updated_at=datetime.now(UTC),
            domain_events=self.domain_events + (
                RubricAssignedEvent(aggregate_id=self.id, rubric_id=rubric_id),
            ),
        )

    def clear_events(self) -> Agent:
        return replace(self, domain_events=())
