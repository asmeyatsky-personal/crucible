"""
Evaluation Entity (JUDGE + SCORE Modules)

Architectural Intent:
- Aggregate root representing a completed evaluation of a trace against a rubric
- Links a TRACE artefact, RUBRIC version, JUDGE configuration, and resulting scores
- Every evaluation is fully traceable and auditable
- Immutable once created — evaluations are facts, not mutable state

MCP Integration:
- Exposed as resource evaluation://{evaluation_id}
- Created via 'evaluate_run' tool
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from domain.events.event_base import DomainEvent
from domain.value_objects.judge_config import JudgeConfig
from domain.value_objects.score import CompositeScore, DimensionScore


@dataclass(frozen=True)
class EvaluationCompletedEvent(DomainEvent):
    trace_id: str = ""
    rubric_id: str = ""
    composite_score: float = 0.0
    passed: bool = False


@dataclass(frozen=True)
class RegressionDetectedEvent(DomainEvent):
    agent_id: str = ""
    current_score: float = 0.0
    baseline_score: float = 0.0
    delta: float = 0.0


@dataclass(frozen=True)
class Evaluation:
    """A completed evaluation of a trace against a rubric."""

    id: str
    trace_id: str
    rubric_id: str
    rubric_version: int
    agent_id: str
    run_id: str
    judge_config: JudgeConfig
    composite_score: CompositeScore
    dimension_scores: tuple[DimensionScore, ...]
    judge_model_id: str
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict = field(default_factory=dict)
    domain_events: tuple[DomainEvent, ...] = field(default=())

    @staticmethod
    def create(
        id: str, trace_id: str, rubric_id: str, rubric_version: int,
        agent_id: str, run_id: str, judge_config: JudgeConfig,
        composite_score: CompositeScore, dimension_scores: list[DimensionScore],
        judge_model_id: str, metadata: dict | None = None,
    ) -> Evaluation:
        """Factory method to create a completed evaluation."""
        return Evaluation(
            id=id,
            trace_id=trace_id,
            rubric_id=rubric_id,
            rubric_version=rubric_version,
            agent_id=agent_id,
            run_id=run_id,
            judge_config=judge_config,
            composite_score=composite_score,
            dimension_scores=tuple(dimension_scores),
            judge_model_id=judge_model_id,
            metadata=metadata or {},
            domain_events=(
                EvaluationCompletedEvent(
                    aggregate_id=id,
                    trace_id=trace_id,
                    rubric_id=rubric_id,
                    composite_score=composite_score.value,
                    passed=composite_score.passed,
                ),
            ),
        )

    @property
    def passed(self) -> bool:
        return self.composite_score.passed

    @property
    def score_value(self) -> float:
        return self.composite_score.value

    def clear_events(self) -> Evaluation:
        return replace(self, domain_events=())
