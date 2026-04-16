"""Shared fixtures for presentation layer tests."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from domain.entities.agent import Agent
from domain.entities.evaluation import Evaluation
from domain.entities.report import ExportFormat, Report
from domain.entities.rubric import Rubric
from domain.entities.trace import Trace
from domain.value_objects.judge_config import JudgeConfig, JudgeModel
from domain.value_objects.rubric_dimension import RubricDimension, ScoringMethod
from domain.value_objects.score import CompositeScore, Confidence, DimensionScore
from domain.value_objects.trajectory_step import StepType, TrajectoryStep
from infrastructure.config.dependency_injection import Container
from infrastructure.config.settings import Settings
from presentation.api.app import create_app


# ---------------------------------------------------------------------------
# Domain object factories
# ---------------------------------------------------------------------------

NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)


def make_agent(
    id: str = "agent-1",
    name: str = "test-agent",
    description: str = "A test agent",
    agent_type: str = "coding",
) -> Agent:
    return Agent(
        id=id,
        name=name,
        description=description,
        agent_type=agent_type,
        rubric_ids=("rubric-1",),
        created_at=NOW,
        updated_at=NOW,
        metadata={"env": "test"},
    )


def make_trace(
    id: str = "trace-1",
    agent_id: str = "agent-1",
    run_id: str = "run-1",
) -> Trace:
    step = TrajectoryStep(
        step_index=0,
        step_type=StepType.USER_MESSAGE,
        content="Hello, world!",
        timestamp=NOW,
    )
    return Trace(
        id=id,
        agent_id=agent_id,
        run_id=run_id,
        steps=(step,),
        model_id="claude-sonnet-4-6",
        temperature=0.0,
        total_tokens=150,
        total_latency_ms=320.5,
        captured_at=NOW,
        metadata={},
    )


def make_rubric(
    id: str = "rubric-1",
    name: str = "test-rubric",
) -> Rubric:
    dim = RubricDimension(
        name="accuracy",
        description="Is the answer accurate?",
        scoring_method=ScoringMethod.LLM_JUDGE,
        weight=1.0,
        pass_threshold=0.7,
        judge_prompt="Rate accuracy 0-1",
    )
    return Rubric(
        id=id,
        name=name,
        description="A test rubric",
        dimensions=(dim,),
        version=1,
        agent_type="coding",
        created_at=NOW,
        updated_at=NOW,
        owner="tester",
    )


def make_dimension_score() -> DimensionScore:
    return DimensionScore(
        dimension_name="accuracy",
        score=0.9,
        passed=True,
        rationale="Very accurate response",
        confidence=Confidence.HIGH,
    )


def make_composite_score() -> CompositeScore:
    ds = make_dimension_score()
    return CompositeScore(
        value=0.9,
        dimension_scores=(ds,),
        passed=True,
        dimensions_passed=1,
        dimensions_total=1,
    )


def make_evaluation(
    id: str = "eval-1",
    trace_id: str = "trace-1",
    rubric_id: str = "rubric-1",
    agent_id: str = "agent-1",
) -> Evaluation:
    ds = make_dimension_score()
    cs = make_composite_score()
    return Evaluation(
        id=id,
        trace_id=trace_id,
        rubric_id=rubric_id,
        rubric_version=1,
        agent_id=agent_id,
        run_id="run-1",
        judge_config=JudgeConfig(primary_model=JudgeModel.CLAUDE_SONNET),
        composite_score=cs,
        dimension_scores=(ds,),
        judge_model_id="claude-sonnet-4-6",
        evaluated_at=NOW,
    )


def make_report(
    id: str = "report-1",
    agent_id: str = "agent-1",
) -> Report:
    return Report(
        id=id,
        title="Test Report",
        agent_id=agent_id,
        evaluation_ids=("eval-1",),
        export_format=ExportFormat.JSON,
        content='{"summary": "all good"}',
        generated_at=NOW,
    )


# ---------------------------------------------------------------------------
# Container fixture with real SQLite DB (tmp_path)
# ---------------------------------------------------------------------------

@pytest.fixture
async def container(tmp_path):
    """Create a Container backed by a temporary SQLite database."""
    db_path = str(tmp_path / "test_crucible.db")
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        storage_path=str(tmp_path / "storage"),
        anthropic_api_key="test-key",
    )
    c = Container(settings)
    await c.init()
    yield c
    await c.shutdown()


@pytest.fixture
async def app(container):
    """Create a FastAPI app with the container already initialized.

    We bypass the lifespan since the container is already init'd by the fixture.
    """
    from presentation.api.app import create_app as _create_app
    import presentation.api.app as app_module

    # Build app without lifespan (container already initialized)
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from presentation.api.routes.agent_routes import create_agent_router
    from presentation.api.routes.evaluation_routes import create_evaluation_router
    from presentation.api.routes.report_routes import create_report_router
    from presentation.api.routes.rubric_routes import create_rubric_router
    from presentation.api.routes.trace_routes import create_trace_router

    @asynccontextmanager
    async def noop_lifespan(app: FastAPI):
        yield

    application = FastAPI(
        title="CRUCIBLE-TEST",
        lifespan=noop_lifespan,
    )
    application.include_router(
        create_agent_router(container), prefix="/api/v1/agents", tags=["Agents"],
    )
    application.include_router(
        create_trace_router(container), prefix="/api/v1/traces", tags=["Traces"],
    )
    application.include_router(
        create_rubric_router(container), prefix="/api/v1/rubrics", tags=["Rubrics"],
    )
    application.include_router(
        create_evaluation_router(container), prefix="/api/v1/evaluations", tags=["Evaluations"],
    )
    application.include_router(
        create_report_router(container), prefix="/api/v1/reports", tags=["Reports"],
    )

    @application.get("/health")
    async def health():
        return {"status": "healthy", "service": "crucible", "version": "1.0.0"}

    return application


@pytest.fixture
async def client(app):
    """Async HTTP client for testing FastAPI routes."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
