"""Shared fixtures for infrastructure tests."""

from __future__ import annotations

from datetime import UTC, datetime

from domain.entities.agent import Agent
from domain.entities.evaluation import Evaluation
from domain.entities.rubric import Rubric
from domain.entities.trace import Trace
from domain.value_objects.judge_config import JudgeConfig, JudgeModel
from domain.value_objects.rubric_dimension import RubricDimension, ScoringMethod
from domain.value_objects.score import CompositeScore, Confidence, DimensionScore
from domain.value_objects.tool_call import ToolCall
from domain.value_objects.trajectory_step import StepType, TrajectoryStep


def make_tool_call(**overrides) -> ToolCall:
    defaults = dict(
        name="web_search",
        input_parameters={"query": "test"},
        output="result",
        latency_ms=120.0,
        success=True,
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        error=None,
    )
    defaults.update(overrides)
    return ToolCall(**defaults)


def make_step(index: int = 0, **overrides) -> TrajectoryStep:
    defaults = dict(
        step_index=index,
        step_type=StepType.TOOL_CALL,
        content="Step content",
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        tool_calls=(make_tool_call(),),
        token_count=100,
        model_id="claude-sonnet-4-6",
        metadata={},
    )
    defaults.update(overrides)
    return TrajectoryStep(**defaults)


def make_trace(**overrides) -> Trace:
    defaults = dict(
        id="trace-1",
        agent_id="agent-1",
        run_id="run-1",
        steps=(make_step(0), make_step(1, step_type=StepType.FINAL_OUTPUT, content="Final output", tool_calls=())),
        model_id="claude-sonnet-4-6",
        temperature=0.0,
        total_tokens=500,
        total_latency_ms=1200.0,
        captured_at=datetime(2025, 1, 1, tzinfo=UTC),
        metadata={},
    )
    defaults.update(overrides)
    return Trace(**defaults)


def make_dimension(**overrides) -> RubricDimension:
    defaults = dict(
        name="accuracy",
        description="How accurate is the output",
        scoring_method=ScoringMethod.LLM_JUDGE,
        weight=1.0,
        pass_threshold=0.7,
        judge_prompt="Evaluate accuracy of the output.",
    )
    defaults.update(overrides)
    return RubricDimension(**defaults)


def make_rubric(**overrides) -> Rubric:
    defaults = dict(
        id="rubric-1",
        name="test-rubric",
        description="A test rubric",
        dimensions=(make_dimension(),),
        version=1,
        agent_type="research",
        owner="test",
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
        updated_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    defaults.update(overrides)
    return Rubric(**defaults)


def make_dimension_score(**overrides) -> DimensionScore:
    defaults = dict(
        dimension_name="accuracy",
        score=0.85,
        passed=True,
        rationale="Good accuracy",
        confidence=Confidence.HIGH,
    )
    defaults.update(overrides)
    return DimensionScore(**defaults)


def make_composite_score(**overrides) -> CompositeScore:
    dim_scores = overrides.pop("dimension_scores", (make_dimension_score(),))
    defaults = dict(
        value=0.85,
        dimension_scores=dim_scores,
        passed=True,
        dimensions_passed=1,
        dimensions_total=1,
    )
    defaults.update(overrides)
    return CompositeScore(**defaults)


def make_evaluation(**overrides) -> Evaluation:
    dim_scores = [make_dimension_score()]
    composite = make_composite_score()
    defaults = dict(
        id="eval-1",
        trace_id="trace-1",
        rubric_id="rubric-1",
        rubric_version=1,
        agent_id="agent-1",
        run_id="run-1",
        judge_config=JudgeConfig(primary_model=JudgeModel.CLAUDE_SONNET),
        composite_score=composite,
        dimension_scores=tuple(dim_scores),
        judge_model_id="claude-sonnet-4-6",
        evaluated_at=datetime(2025, 1, 1, tzinfo=UTC),
        metadata={},
    )
    defaults.update(overrides)
    return Evaluation(**defaults)


def make_agent(**overrides) -> Agent:
    defaults = dict(
        id="agent-1",
        name="test-agent",
        description="A test agent",
        agent_type="research",
        rubric_ids=(),
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
        updated_at=datetime(2025, 1, 1, tzinfo=UTC),
        metadata={},
    )
    defaults.update(overrides)
    return Agent(**defaults)
