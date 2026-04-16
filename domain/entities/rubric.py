"""
Rubric Entity (RUBRIC Module)

Architectural Intent:
- Aggregate root representing a versioned set of evaluation criteria
- Rubrics are collections of weighted dimensions with pass thresholds
- Version-controlled and shareable across teams
- YAML-serialisable for version control integration

MCP Integration:
- Exposed as resource rubric://{rubric_id} for read access
- Created/updated via 'create_rubric' / 'update_rubric' tools
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from domain.events.event_base import DomainEvent
from domain.value_objects.rubric_dimension import RubricDimension


@dataclass(frozen=True)
class RubricCreatedEvent(DomainEvent):
    rubric_name: str = ""
    version: int = 1


@dataclass(frozen=True)
class RubricUpdatedEvent(DomainEvent):
    new_version: int = 0


@dataclass(frozen=True)
class Rubric:
    """Versioned evaluation criteria set for CRUCIBLE evaluation."""

    id: str
    name: str
    description: str
    dimensions: tuple[RubricDimension, ...]
    version: int = 1
    agent_type: str | None = None  # Optional archetype binding
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    owner: str = ""
    domain_events: tuple[DomainEvent, ...] = field(default=())

    @staticmethod
    def create(
        id: str, name: str, description: str,
        dimensions: list[RubricDimension],
        agent_type: str | None = None, owner: str = "",
    ) -> Rubric:
        """Factory method to create a new rubric."""
        now = datetime.now(UTC)
        rubric = Rubric(
            id=id,
            name=name,
            description=description,
            dimensions=tuple(dimensions),
            agent_type=agent_type,
            created_at=now,
            updated_at=now,
            owner=owner,
            domain_events=(
                RubricCreatedEvent(aggregate_id=id, rubric_name=name, version=1),
            ),
        )
        rubric._validate_weights()
        return rubric

    def _validate_weights(self) -> None:
        if not self.dimensions:
            raise ValueError("Rubric must have at least one dimension")

    def update_dimensions(self, dimensions: list[RubricDimension]) -> Rubric:
        """Create a new version with updated dimensions."""
        new_version = self.version + 1
        updated = replace(
            self,
            dimensions=tuple(dimensions),
            version=new_version,
            updated_at=datetime.now(UTC),
            domain_events=self.domain_events + (
                RubricUpdatedEvent(aggregate_id=self.id, new_version=new_version),
            ),
        )
        updated._validate_weights()
        return updated

    @property
    def normalised_weights(self) -> dict[str, float]:
        """Return weights normalised to sum to 1.0."""
        total = sum(d.weight for d in self.dimensions)
        if total == 0:
            return {d.name: 0.0 for d in self.dimensions}
        return {d.name: d.weight / total for d in self.dimensions}

    def clear_events(self) -> Rubric:
        return replace(self, domain_events=())
