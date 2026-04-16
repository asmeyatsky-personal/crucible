"""
Application Query Tests — mocked ports, verify query orchestration.
"""

import pytest
from datetime import UTC, datetime
from unittest.mock import AsyncMock

from application.queries.get_run import GetRunQuery, RunDetail
from application.queries.get_scores import GetScoresQuery, ScoreTimeline
from domain.entities.trace import Trace
from domain.services.regression_detection_service import RegressionDetectionService
from domain.value_objects.trajectory_step import StepType, TrajectoryStep
from tests.domain.conftest import make_evaluation


def _make_trace(agent_id: str = "a-1", run_id: str = "run-1") -> Trace:
    return Trace.capture(
        id="t-1", agent_id=agent_id, run_id=run_id,
        steps=[
            TrajectoryStep(
                step_index=0, step_type=StepType.USER_MESSAGE,
                content="Hello", timestamp=datetime.now(UTC),
            ),
        ],
        model_id="model", temperature=0.0,
        total_tokens=100, total_latency_ms=500.0,
    ).clear_events()


class TestGetRunQuery:
    @pytest.mark.asyncio
    async def test_returns_run_detail_with_evaluation(self):
        trace_repo = AsyncMock()
        eval_repo = AsyncMock()
        trace = _make_trace()
        evaluation = make_evaluation(id="e-1", agent_id="a-1")

        trace_repo.get_by_run_id.return_value = trace
        eval_repo.list_by_agent.return_value = [evaluation]

        query = GetRunQuery(trace_repo, eval_repo)
        result = await query.execute("run-1")

        assert result is not None
        assert isinstance(result, RunDetail)
        assert result.trace.id == "t-1"
        assert result.evaluation is not None

    @pytest.mark.asyncio
    async def test_returns_none_when_trace_not_found(self):
        trace_repo = AsyncMock()
        eval_repo = AsyncMock()
        trace_repo.get_by_run_id.return_value = None

        query = GetRunQuery(trace_repo, eval_repo)
        result = await query.execute("missing-run")

        assert result is None
        eval_repo.list_by_agent.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_none_evaluation_when_no_match(self):
        trace_repo = AsyncMock()
        eval_repo = AsyncMock()
        trace = _make_trace(run_id="run-1")

        trace_repo.get_by_run_id.return_value = trace
        eval_repo.list_by_agent.return_value = []

        query = GetRunQuery(trace_repo, eval_repo)
        result = await query.execute("run-1")

        assert result is not None
        assert result.evaluation is None


class TestGetScoresQuery:
    @pytest.mark.asyncio
    async def test_returns_timeline_with_regressions(self):
        eval_repo = AsyncMock()
        regression_service = RegressionDetectionService()

        # Create evaluations with a regression between them
        e1 = make_evaluation(id="e-1", score=0.90)
        e2 = make_evaluation(id="e-2", score=0.50)  # Big drop
        eval_repo.list_by_agent.return_value = [e1, e2]

        query = GetScoresQuery(eval_repo, regression_service)
        result = await query.execute("a-1", limit=50)

        assert isinstance(result, ScoreTimeline)
        assert len(result.evaluations) == 2
        assert len(result.regressions) == 1
        assert result.regressions[0][0] == "e-2"
        assert result.regressions[0][1].regressed is True

    @pytest.mark.asyncio
    async def test_returns_empty_timeline(self):
        eval_repo = AsyncMock()
        regression_service = RegressionDetectionService()
        eval_repo.list_by_agent.return_value = []

        query = GetScoresQuery(eval_repo, regression_service)
        result = await query.execute("a-1")

        assert len(result.evaluations) == 0
        assert len(result.regressions) == 0

    @pytest.mark.asyncio
    async def test_no_regressions_when_scores_stable(self):
        eval_repo = AsyncMock()
        regression_service = RegressionDetectionService()

        e1 = make_evaluation(id="e-1", score=0.85)
        e2 = make_evaluation(id="e-2", score=0.86)
        eval_repo.list_by_agent.return_value = [e1, e2]

        query = GetScoresQuery(eval_repo, regression_service)
        result = await query.execute("a-1")

        assert len(result.regressions) == 0
