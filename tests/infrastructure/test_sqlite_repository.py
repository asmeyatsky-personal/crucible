"""
Tests for SQLite repository implementations.

Uses in-memory SQLite database for isolation.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from domain.entities.agent import Agent
from domain.entities.evaluation import Evaluation
from domain.entities.rubric import Rubric
from domain.entities.trace import Trace
from domain.value_objects.judge_config import ConsensusMode, JudgeConfig, JudgeModel
from domain.value_objects.rubric_dimension import RubricDimension, ScoringMethod
from domain.value_objects.score import CompositeScore, Confidence, DimensionScore
from domain.value_objects.tool_call import ToolCall
from domain.value_objects.trajectory_step import StepType, TrajectoryStep
from infrastructure.repositories.sqlite_repository import (
    SQLiteAgentRepository,
    SQLiteDatabase,
    SQLiteEvaluationRepository,
    SQLiteRubricRepository,
    SQLiteTraceRepository,
)


@pytest.fixture
async def db():
    database = SQLiteDatabase(db_path=":memory:")
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
def agent_repo(db):
    return SQLiteAgentRepository(db)


@pytest.fixture
def trace_repo(db):
    return SQLiteTraceRepository(db)


@pytest.fixture
def rubric_repo(db):
    return SQLiteRubricRepository(db)


@pytest.fixture
def eval_repo(db):
    return SQLiteEvaluationRepository(db)


def _make_agent(id: str = "agent-1", name: str = "test-agent") -> Agent:
    return Agent(
        id=id,
        name=name,
        description="Test agent",
        agent_type="research",
        rubric_ids=("r-1",),
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
        updated_at=datetime(2025, 1, 1, tzinfo=UTC),
        metadata={"key": "value"},
    )


def _make_trace(id: str = "trace-1", agent_id: str = "agent-1", run_id: str = "run-1") -> Trace:
    step = TrajectoryStep(
        step_index=0,
        step_type=StepType.USER_MESSAGE,
        content="Hello agent",
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        tool_calls=(
            ToolCall(
                name="search",
                input_parameters={"q": "test"},
                output="result",
                latency_ms=100.0,
                success=True,
                timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            ),
        ),
        token_count=50,
        model_id="claude-sonnet-4-6",
        metadata={},
    )
    return Trace(
        id=id,
        agent_id=agent_id,
        run_id=run_id,
        steps=(step,),
        model_id="claude-sonnet-4-6",
        temperature=0.0,
        total_tokens=200,
        total_latency_ms=500.0,
        captured_at=datetime(2025, 1, 1, tzinfo=UTC),
        metadata={"env": "test"},
    )


def _make_rubric(id: str = "rubric-1", name: str = "test-rubric") -> Rubric:
    dim = RubricDimension(
        name="accuracy",
        description="How accurate",
        scoring_method=ScoringMethod.LLM_JUDGE,
        weight=1.0,
        pass_threshold=0.7,
        judge_prompt="Evaluate accuracy",
    )
    return Rubric(
        id=id,
        name=name,
        description="Test rubric",
        dimensions=(dim,),
        version=1,
        agent_type="research",
        owner="test-owner",
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
        updated_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


def _make_evaluation(id: str = "eval-1", agent_id: str = "agent-1") -> Evaluation:
    dim_score = DimensionScore(
        dimension_name="accuracy",
        score=0.9,
        passed=True,
        rationale="Accurate output",
        confidence=Confidence.HIGH,
    )
    composite = CompositeScore(
        value=0.9,
        dimension_scores=(dim_score,),
        passed=True,
        dimensions_passed=1,
        dimensions_total=1,
    )
    judge_config = JudgeConfig(primary_model=JudgeModel.CLAUDE_SONNET)
    return Evaluation(
        id=id,
        trace_id="trace-1",
        rubric_id="rubric-1",
        rubric_version=1,
        agent_id=agent_id,
        run_id="run-1",
        judge_config=judge_config,
        composite_score=composite,
        dimension_scores=(dim_score,),
        judge_model_id="claude-sonnet-4-6",
        evaluated_at=datetime(2025, 1, 1, tzinfo=UTC),
        metadata={"test": True},
    )


class TestSQLiteAgentRepository:
    @pytest.mark.asyncio
    async def test_save_and_get_by_id(self, agent_repo):
        agent = _make_agent()
        await agent_repo.save(agent)

        result = await agent_repo.get_by_id("agent-1")

        assert result is not None
        assert result.id == "agent-1"
        assert result.name == "test-agent"
        assert result.agent_type == "research"
        assert result.rubric_ids == ("r-1",)
        assert result.metadata == {"key": "value"}

    @pytest.mark.asyncio
    async def test_get_by_id_nonexistent(self, agent_repo):
        result = await agent_repo.get_by_id("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_name(self, agent_repo):
        agent = _make_agent(name="unique-name")
        await agent_repo.save(agent)

        result = await agent_repo.get_by_name("unique-name")

        assert result is not None
        assert result.name == "unique-name"

    @pytest.mark.asyncio
    async def test_get_by_name_nonexistent(self, agent_repo):
        result = await agent_repo.get_by_name("no-such-agent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_all(self, agent_repo):
        await agent_repo.save(_make_agent(id="a-1", name="agent-one"))
        await agent_repo.save(_make_agent(id="a-2", name="agent-two"))
        await agent_repo.save(_make_agent(id="a-3", name="agent-three"))

        result = await agent_repo.list_all()

        assert len(result) == 3
        ids = {a.id for a in result}
        assert ids == {"a-1", "a-2", "a-3"}

    @pytest.mark.asyncio
    async def test_list_all_empty(self, agent_repo):
        result = await agent_repo.list_all()
        assert result == []

    @pytest.mark.asyncio
    async def test_save_updates_existing(self, agent_repo):
        agent = _make_agent()
        await agent_repo.save(agent)

        updated = Agent(
            id="agent-1",
            name="test-agent",
            description="Updated description",
            agent_type="coding",
            rubric_ids=("r-1", "r-2"),
            created_at=agent.created_at,
            updated_at=datetime(2025, 6, 1, tzinfo=UTC),
            metadata={"updated": True},
        )
        await agent_repo.save(updated)

        result = await agent_repo.get_by_id("agent-1")
        assert result.description == "Updated description"
        assert result.agent_type == "coding"
        assert result.rubric_ids == ("r-1", "r-2")


class TestSQLiteTraceRepository:
    @pytest.mark.asyncio
    async def test_save_and_get_by_id(self, trace_repo):
        trace = _make_trace()
        await trace_repo.save(trace)

        result = await trace_repo.get_by_id("trace-1")

        assert result is not None
        assert result.id == "trace-1"
        assert result.agent_id == "agent-1"
        assert result.model_id == "claude-sonnet-4-6"
        assert len(result.steps) == 1
        assert result.steps[0].content == "Hello agent"
        assert len(result.steps[0].tool_calls) == 1
        assert result.steps[0].tool_calls[0].name == "search"

    @pytest.mark.asyncio
    async def test_get_by_id_nonexistent(self, trace_repo):
        result = await trace_repo.get_by_id("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_run_id(self, trace_repo):
        trace = _make_trace(run_id="unique-run")
        await trace_repo.save(trace)

        result = await trace_repo.get_by_run_id("unique-run")

        assert result is not None
        assert result.run_id == "unique-run"

    @pytest.mark.asyncio
    async def test_get_by_run_id_nonexistent(self, trace_repo):
        result = await trace_repo.get_by_run_id("no-such-run")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_by_agent(self, trace_repo):
        await trace_repo.save(_make_trace(id="t-1", agent_id="a-1", run_id="r-1"))
        await trace_repo.save(_make_trace(id="t-2", agent_id="a-1", run_id="r-2"))
        await trace_repo.save(_make_trace(id="t-3", agent_id="a-2", run_id="r-3"))

        result = await trace_repo.list_by_agent("a-1")

        assert len(result) == 2
        ids = {t.id for t in result}
        assert ids == {"t-1", "t-2"}

    @pytest.mark.asyncio
    async def test_list_by_agent_empty(self, trace_repo):
        result = await trace_repo.list_by_agent("no-agent")
        assert result == []

    @pytest.mark.asyncio
    async def test_list_by_agent_respects_limit(self, trace_repo):
        for i in range(10):
            await trace_repo.save(_make_trace(id=f"t-{i}", agent_id="a-1", run_id=f"r-{i}"))

        result = await trace_repo.list_by_agent("a-1", limit=3)

        assert len(result) == 3


class TestSQLiteRubricRepository:
    @pytest.mark.asyncio
    async def test_save_and_get_by_id(self, rubric_repo):
        rubric = _make_rubric()
        await rubric_repo.save(rubric)

        result = await rubric_repo.get_by_id("rubric-1")

        assert result is not None
        assert result.id == "rubric-1"
        assert result.name == "test-rubric"
        assert len(result.dimensions) == 1
        assert result.dimensions[0].name == "accuracy"
        assert result.dimensions[0].scoring_method == ScoringMethod.LLM_JUDGE

    @pytest.mark.asyncio
    async def test_get_by_id_nonexistent(self, rubric_repo):
        result = await rubric_repo.get_by_id("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_name(self, rubric_repo):
        rubric = _make_rubric(name="named-rubric")
        await rubric_repo.save(rubric)

        result = await rubric_repo.get_by_name("named-rubric")

        assert result is not None
        assert result.name == "named-rubric"

    @pytest.mark.asyncio
    async def test_get_by_name_nonexistent(self, rubric_repo):
        result = await rubric_repo.get_by_name("no-such-rubric")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_all(self, rubric_repo):
        await rubric_repo.save(_make_rubric(id="r-1", name="rubric-one"))
        await rubric_repo.save(_make_rubric(id="r-2", name="rubric-two"))

        result = await rubric_repo.list_all()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_all_empty(self, rubric_repo):
        result = await rubric_repo.list_all()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_by_agent_type(self, rubric_repo):
        r1 = _make_rubric(id="r-1", name="rubric-research")
        r2 = Rubric(
            id="r-2",
            name="rubric-coding",
            description="Coding rubric",
            dimensions=_make_rubric().dimensions,
            version=1,
            agent_type="coding",
            owner="test",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            updated_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        await rubric_repo.save(r1)
        await rubric_repo.save(r2)

        result = await rubric_repo.list_by_agent_type("research")

        assert len(result) == 1
        assert result[0].id == "r-1"

    @pytest.mark.asyncio
    async def test_list_by_agent_type_empty(self, rubric_repo):
        result = await rubric_repo.list_by_agent_type("nonexistent")
        assert result == []


class TestSQLiteEvaluationRepository:
    @pytest.mark.asyncio
    async def test_save_and_get_by_id(self, eval_repo):
        evaluation = _make_evaluation()
        await eval_repo.save(evaluation)

        result = await eval_repo.get_by_id("eval-1")

        assert result is not None
        assert result.id == "eval-1"
        assert result.trace_id == "trace-1"
        assert result.rubric_id == "rubric-1"
        assert result.composite_score.value == 0.9
        assert result.composite_score.passed is True
        assert len(result.dimension_scores) == 1
        assert result.dimension_scores[0].dimension_name == "accuracy"
        assert result.judge_config.primary_model == JudgeModel.CLAUDE_SONNET

    @pytest.mark.asyncio
    async def test_get_by_id_nonexistent(self, eval_repo):
        result = await eval_repo.get_by_id("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_by_agent(self, eval_repo):
        await eval_repo.save(_make_evaluation(id="ev-1", agent_id="a-1"))
        await eval_repo.save(_make_evaluation(id="ev-2", agent_id="a-1"))
        await eval_repo.save(_make_evaluation(id="ev-3", agent_id="a-2"))

        result = await eval_repo.list_by_agent("a-1")

        assert len(result) == 2
        ids = {e.id for e in result}
        assert ids == {"ev-1", "ev-2"}

    @pytest.mark.asyncio
    async def test_list_by_agent_empty(self, eval_repo):
        result = await eval_repo.list_by_agent("no-agent")
        assert result == []

    @pytest.mark.asyncio
    async def test_list_by_rubric(self, eval_repo):
        await eval_repo.save(_make_evaluation(id="ev-1"))
        await eval_repo.save(_make_evaluation(id="ev-2"))

        result = await eval_repo.list_by_rubric("rubric-1")

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_by_rubric_empty(self, eval_repo):
        result = await eval_repo.list_by_rubric("no-rubric")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_latest_by_agent(self, eval_repo):
        eval1 = Evaluation(
            id="ev-old",
            trace_id="t-1",
            rubric_id="r-1",
            rubric_version=1,
            agent_id="a-1",
            run_id="run-1",
            judge_config=JudgeConfig(primary_model=JudgeModel.CLAUDE_SONNET),
            composite_score=CompositeScore(
                value=0.7,
                dimension_scores=(_make_evaluation().dimension_scores[0],),
                passed=True,
                dimensions_passed=1,
                dimensions_total=1,
            ),
            dimension_scores=_make_evaluation().dimension_scores,
            judge_model_id="claude-sonnet-4-6",
            evaluated_at=datetime(2025, 1, 1, tzinfo=UTC),
            metadata={},
        )
        eval2 = Evaluation(
            id="ev-new",
            trace_id="t-2",
            rubric_id="r-1",
            rubric_version=1,
            agent_id="a-1",
            run_id="run-2",
            judge_config=JudgeConfig(primary_model=JudgeModel.CLAUDE_SONNET),
            composite_score=CompositeScore(
                value=0.95,
                dimension_scores=(_make_evaluation().dimension_scores[0],),
                passed=True,
                dimensions_passed=1,
                dimensions_total=1,
            ),
            dimension_scores=_make_evaluation().dimension_scores,
            judge_model_id="claude-sonnet-4-6",
            evaluated_at=datetime(2025, 6, 1, tzinfo=UTC),
            metadata={},
        )
        await eval_repo.save(eval1)
        await eval_repo.save(eval2)

        result = await eval_repo.get_latest_by_agent("a-1")

        assert result is not None
        assert result.id == "ev-new"

    @pytest.mark.asyncio
    async def test_get_latest_by_agent_nonexistent(self, eval_repo):
        result = await eval_repo.get_latest_by_agent("no-agent")
        assert result is None

    @pytest.mark.asyncio
    async def test_roundtrip_preserves_judge_config(self, eval_repo):
        evaluation = _make_evaluation()
        await eval_repo.save(evaluation)

        result = await eval_repo.get_by_id("eval-1")

        assert result.judge_config.primary_model == JudgeModel.CLAUDE_SONNET
        assert result.judge_config.consensus_mode == ConsensusMode.SINGLE
        assert result.judge_config.temperature == 0.0
        assert result.judge_config.max_tokens == 4096

    @pytest.mark.asyncio
    async def test_roundtrip_preserves_dimension_scores(self, eval_repo):
        evaluation = _make_evaluation()
        await eval_repo.save(evaluation)

        result = await eval_repo.get_by_id("eval-1")

        ds = result.dimension_scores[0]
        assert ds.dimension_name == "accuracy"
        assert ds.score == 0.9
        assert ds.passed is True
        assert ds.rationale == "Accurate output"
        assert ds.confidence == Confidence.HIGH
