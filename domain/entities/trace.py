"""
Trace Entity (TRACE Module)

Architectural Intent:
- Aggregate root representing a captured agent execution trajectory
- Immutable artefact — once captured, a trace is never modified
- Contains the full sequence of trajectory steps from system prompt to final output
- Indexed by agent_id, run_id, and timestamp for retrieval

MCP Integration:
- Exposed as resource trace://{trace_id} for read access
- Created via 'capture_trace' tool on the crucible-service MCP server
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from domain.events.event_base import DomainEvent
from domain.value_objects.trajectory_step import TrajectoryStep


@dataclass(frozen=True)
class TraceCapturedEvent(DomainEvent):
    agent_id: str = ""
    run_id: str = ""
    step_count: int = 0


@dataclass(frozen=True)
class Trace:
    """Immutable TRACE artefact — a full agent execution trajectory."""

    id: str
    agent_id: str
    run_id: str
    steps: tuple[TrajectoryStep, ...]
    model_id: str
    temperature: float
    total_tokens: int
    total_latency_ms: float
    captured_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict = field(default_factory=dict)
    domain_events: tuple[DomainEvent, ...] = field(default=())

    @staticmethod
    def capture(
        id: str, agent_id: str, run_id: str, steps: list[TrajectoryStep],
        model_id: str, temperature: float, total_tokens: int,
        total_latency_ms: float, metadata: dict | None = None,
    ) -> Trace:
        """Factory method to capture a new trace."""
        sorted_steps = tuple(sorted(steps, key=lambda s: s.step_index))
        return Trace(
            id=id,
            agent_id=agent_id,
            run_id=run_id,
            steps=sorted_steps,
            model_id=model_id,
            temperature=temperature,
            total_tokens=total_tokens,
            total_latency_ms=total_latency_ms,
            metadata=metadata or {},
            domain_events=(
                TraceCapturedEvent(
                    aggregate_id=id, agent_id=agent_id,
                    run_id=run_id, step_count=len(sorted_steps),
                ),
            ),
        )

    @property
    def tool_calls(self) -> tuple:
        """Extract all tool calls across the trajectory."""
        calls = []
        for step in self.steps:
            calls.extend(step.tool_calls)
        return tuple(calls)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def clear_events(self) -> Trace:
        return replace(self, domain_events=())
