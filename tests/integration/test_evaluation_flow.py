"""
Integration Tests — Full Evaluation Flow

Tests the complete pipeline: register agent -> capture trace ->
create rubric -> evaluate (with mock judge) -> check scores.

Uses real SQLite database and local storage, mocks only the LLM judge.
"""

import pytest
from datetime import UTC, datetime
from unittest.mock import AsyncMock

from application.commands.capture_trace import CaptureTraceUseCase
from application.commands.create_rubric import CreateRubricUseCase
from application.commands.evaluate_run import EvaluateRunUseCase
from application.commands.register_agent import RegisterAgentUseCase
from application.dtos.evaluation_dto import EvaluateRunRequest, JudgeConfigDTO
from application.dtos.rubric_dto import CreateRubricRequest, RubricDimensionDTO
from application.dtos.trace_dto import CaptureTraceRequest, TrajectoryStepDTO
from domain.services.regression_detection_service import RegressionDetectionService
from domain.services.scoring_service import ScoringService
from domain.value_objects.score import Confidence, DimensionScore
from infrastructure.adapters.event_bus_adapter import InMemoryEventBusAdapter
from infrastructure.adapters.local_storage_adapter import LocalStorageAdapter
from infrastructure.repositories.sqlite_repository import (
    SQLiteAgentRepository,
    SQLiteDatabase,
    SQLiteEvaluationRepository,
    SQLiteRubricRepository,
    SQLiteTraceRepository,
)


@pytest.fixture
async def database(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "test.db"))
    await db.connect()
    yield db
    await db.close()


@pytest.fixture
def repos(database):
    return {
        "agent_repo": SQLiteAgentRepository(database),
        "trace_repo": SQLiteTraceRepository(database),
        "rubric_repo": SQLiteRubricRepository(database),
        "eval_repo": SQLiteEvaluationRepository(database),
    }


@pytest.fixture
def event_bus():
    return InMemoryEventBusAdapter()


@pytest.fixture
def storage(tmp_path):
    return LocalStorageAdapter(str(tmp_path / "storage"))


class TestFullEvaluationFlow:
    @pytest.mark.asyncio
    async def test_end_to_end_evaluation(self, repos, event_bus, storage):
        # 1. Register agent
        register = RegisterAgentUseCase(repos["agent_repo"], event_bus)
        agent = await register.execute(
            name="test-research-agent",
            description="Integration test agent",
            agent_type="research",
        )
        assert agent.name == "test-research-agent"

        # 2. Capture trace
        capture = CaptureTraceUseCase(
            repos["trace_repo"], repos["agent_repo"], storage, event_bus,
        )
        trace_request = CaptureTraceRequest(
            agent_id=agent.id,
            run_id="integration-run-1",
            steps=[
                TrajectoryStepDTO(
                    step_index=0, step_type="system_prompt",
                    content="You are a research assistant.",
                    timestamp=datetime.now(UTC),
                ),
                TrajectoryStepDTO(
                    step_index=1, step_type="user_message",
                    content="Research the history of Python.",
                    timestamp=datetime.now(UTC),
                ),
                TrajectoryStepDTO(
                    step_index=2, step_type="assistant_response",
                    content="Python was created by Guido van Rossum in 1991...",
                    timestamp=datetime.now(UTC),
                ),
            ],
            model_id="claude-sonnet-4-6",
            total_tokens=350,
            total_latency_ms=2500.0,
        )
        trace = await capture.execute(trace_request)
        assert trace.step_count == 3

        # 3. Create rubric
        create_rubric = CreateRubricUseCase(repos["rubric_repo"], event_bus)
        rubric_request = CreateRubricRequest(
            name="integration-test-rubric",
            description="Test rubric for integration testing",
            dimensions=[
                RubricDimensionDTO(
                    name="goal_adherence",
                    description="Did the agent research Python history?",
                    scoring_method="llm_judge",
                    weight=1.0,
                    pass_threshold=0.7,
                    judge_prompt="Did the agent research Python history?",
                ),
                RubricDimensionDTO(
                    name="output_quality",
                    description="Was the output informative?",
                    scoring_method="llm_judge",
                    weight=1.0,
                    pass_threshold=0.7,
                    judge_prompt="Was the output informative and accurate?",
                ),
            ],
        )
        rubric = await create_rubric.execute(rubric_request)
        assert len(rubric.dimensions) == 2

        # 4. Evaluate (with mock judge)
        mock_judge = AsyncMock()
        mock_judge.evaluate_dimension.side_effect = [
            DimensionScore(
                dimension_name="goal_adherence", score=0.92, passed=True,
                rationale="Agent researched Python history well.",
                confidence=Confidence.HIGH,
            ),
            DimensionScore(
                dimension_name="output_quality", score=0.85, passed=True,
                rationale="Output was informative and accurate.",
                confidence=Confidence.HIGH,
            ),
        ]

        evaluate = EvaluateRunUseCase(
            trace_repository=repos["trace_repo"],
            rubric_repository=repos["rubric_repo"],
            evaluation_repository=repos["eval_repo"],
            judge=mock_judge,
            scoring_service=ScoringService(),
            regression_service=RegressionDetectionService(),
            event_bus=event_bus,
        )

        eval_request = EvaluateRunRequest(
            trace_id=trace.id,
            rubric_id=rubric.id,
            judge_config=JudgeConfigDTO(),
        )
        evaluation = await evaluate.execute(eval_request)

        assert evaluation.passed is True
        assert evaluation.composite_score.value > 0.8
        assert evaluation.composite_score.dimensions_passed == 2

        # 5. Verify persistence
        stored_eval = await repos["eval_repo"].get_by_id(evaluation.id)
        assert stored_eval is not None
        assert stored_eval.composite_score.value == evaluation.composite_score.value

        # 6. Verify trace artefact stored
        artefact = await storage.retrieve(f"traces/{trace.id}.json")
        assert artefact is not None
