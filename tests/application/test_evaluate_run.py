"""
Evaluate Run Use Case Tests — mocked ports, verify orchestration.
"""

import pytest
from unittest.mock import AsyncMock

from application.commands.evaluate_run import EvaluateRunUseCase
from application.dtos.evaluation_dto import EvaluateRunRequest, JudgeConfigDTO
from domain.entities.rubric import Rubric
from domain.entities.trace import Trace
from domain.services.regression_detection_service import RegressionDetectionService
from domain.services.scoring_service import ScoringService
from domain.value_objects.rubric_dimension import RubricDimension, ScoringMethod
from domain.value_objects.score import Confidence, DimensionScore
from domain.value_objects.trajectory_step import StepType, TrajectoryStep
from datetime import UTC, datetime


def _make_trace() -> Trace:
    return Trace.capture(
        id="t-1", agent_id="a-1", run_id="run-1",
        steps=[
            TrajectoryStep(
                step_index=0, step_type=StepType.USER_MESSAGE,
                content="Hello", timestamp=datetime.now(UTC),
            ),
        ],
        model_id="claude-sonnet-4-6", temperature=0.0,
        total_tokens=100, total_latency_ms=500.0,
    ).clear_events()


def _make_rubric() -> Rubric:
    return Rubric.create(
        id="r-1", name="Test Rubric", description="test",
        dimensions=[
            RubricDimension(
                name="goal", description="Goal adherence",
                scoring_method=ScoringMethod.LLM_JUDGE,
                weight=1.0, pass_threshold=0.7,
                judge_prompt="Evaluate goal adherence",
            ),
            RubricDimension(
                name="output", description="Output quality",
                scoring_method=ScoringMethod.LLM_JUDGE,
                weight=1.0, pass_threshold=0.7,
                judge_prompt="Evaluate output quality",
            ),
        ],
    ).clear_events()


def _make_dim_score(name: str, score: float) -> DimensionScore:
    return DimensionScore(
        dimension_name=name, score=score,
        passed=score >= 0.7, rationale="Test rationale",
        confidence=Confidence.HIGH,
    )


class TestEvaluateRunUseCase:
    @pytest.fixture
    def deps(self):
        return {
            "trace_repository": AsyncMock(),
            "rubric_repository": AsyncMock(),
            "evaluation_repository": AsyncMock(),
            "judge": AsyncMock(),
            "scoring_service": ScoringService(),
            "regression_service": RegressionDetectionService(),
            "event_bus": AsyncMock(),
        }

    @pytest.fixture
    def use_case(self, deps):
        return EvaluateRunUseCase(**deps)

    @pytest.mark.asyncio
    async def test_successful_evaluation(self, use_case, deps):
        trace = _make_trace()
        rubric = _make_rubric()

        deps["trace_repository"].get_by_id.return_value = trace
        deps["rubric_repository"].get_by_id.return_value = rubric
        deps["evaluation_repository"].get_latest_by_agent.return_value = None
        deps["judge"].evaluate_dimension.side_effect = [
            _make_dim_score("goal", 0.85),
            _make_dim_score("output", 0.78),
        ]

        request = EvaluateRunRequest(
            trace_id="t-1", rubric_id="r-1",
            judge_config=JudgeConfigDTO(),
        )
        result = await use_case.execute(request)

        assert result.passed is True
        assert len(result.dimension_scores) == 2
        deps["evaluation_repository"].save.assert_awaited_once()
        deps["event_bus"].publish.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_trace_not_found_raises(self, use_case, deps):
        deps["trace_repository"].get_by_id.return_value = None
        deps["rubric_repository"].get_by_id.return_value = _make_rubric()

        request = EvaluateRunRequest(trace_id="bad", rubric_id="r-1")
        with pytest.raises(ValueError, match="Trace not found"):
            await use_case.execute(request)

    @pytest.mark.asyncio
    async def test_rubric_not_found_raises(self, use_case, deps):
        deps["trace_repository"].get_by_id.return_value = _make_trace()
        deps["rubric_repository"].get_by_id.return_value = None

        request = EvaluateRunRequest(trace_id="t-1", rubric_id="bad")
        with pytest.raises(ValueError, match="Rubric not found"):
            await use_case.execute(request)

    @pytest.mark.asyncio
    async def test_dimensions_evaluated_in_parallel(self, use_case, deps):
        """Verify all dimensions are submitted for evaluation (they run via gather)."""
        trace = _make_trace()
        rubric = _make_rubric()

        deps["trace_repository"].get_by_id.return_value = trace
        deps["rubric_repository"].get_by_id.return_value = rubric
        deps["evaluation_repository"].get_latest_by_agent.return_value = None
        deps["judge"].evaluate_dimension.side_effect = [
            _make_dim_score("goal", 0.9),
            _make_dim_score("output", 0.8),
        ]

        request = EvaluateRunRequest(trace_id="t-1", rubric_id="r-1")
        await use_case.execute(request)

        # Judge should be called once per dimension
        assert deps["judge"].evaluate_dimension.await_count == 2
