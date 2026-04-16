"""Tests for the CRUCIBLE CLI (presentation/cli/main.py).

Uses click.testing.CliRunner and mocks the Container to avoid
real database/network dependencies.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
from click.testing import CliRunner

from domain.entities.agent import Agent
from domain.entities.evaluation import Evaluation
from domain.entities.rubric import Rubric
from domain.value_objects.judge_config import JudgeConfig, JudgeModel
from domain.value_objects.rubric_dimension import RubricDimension, ScoringMethod
from domain.value_objects.score import (
    CompositeScore,
    Confidence,
    DimensionScore,
    RegressionResult,
)
from presentation.cli.main import cli

NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def runner():
    return CliRunner()


def _make_mock_container():
    """Create a mock Container with all required methods and repos."""
    container = MagicMock()
    container.init = AsyncMock()
    container.shutdown = AsyncMock()
    return container


def _make_agent(name="test-agent", agent_type="coding", id="agent-1"):
    return Agent(
        id=id,
        name=name,
        description="A test agent",
        agent_type=agent_type,
        created_at=NOW,
        updated_at=NOW,
    )


def _make_rubric():
    dim = RubricDimension(
        name="accuracy",
        description="Is the answer accurate?",
        scoring_method=ScoringMethod.LLM_JUDGE,
        weight=1.0,
        pass_threshold=0.7,
        judge_prompt="Rate accuracy 0-1",
    )
    return Rubric(
        id="rubric-1",
        name="test-rubric",
        description="Test",
        dimensions=(dim,),
        version=1,
        created_at=NOW,
        updated_at=NOW,
    )


def _make_dim_score(score=0.85, passed=True):
    return DimensionScore(
        dimension_name="accuracy",
        score=score,
        passed=passed,
        rationale="test",
        confidence=Confidence.HIGH,
    )


def _make_composite(value=0.85, passed=True):
    ds = _make_dim_score(value, passed)
    return CompositeScore(
        value=value,
        dimension_scores=(ds,),
        passed=passed,
        dimensions_passed=1 if passed else 0,
        dimensions_total=1,
    )


def _make_evaluation(id="eval-1", score=0.85, passed=True):
    cs = _make_composite(score, passed)
    ds = _make_dim_score(score, passed)
    return Evaluation(
        id=id,
        trace_id="t-1",
        rubric_id="r-1",
        rubric_version=1,
        agent_id="agent-1",
        run_id="run-1",
        judge_config=JudgeConfig(primary_model=JudgeModel.CLAUDE_SONNET),
        composite_score=cs,
        dimension_scores=(ds,),
        judge_model_id="claude-sonnet-4-6",
        evaluated_at=NOW,
    )


# ===================================================================
# CLI group
# ===================================================================


class TestCLIGroup:
    def test_cli_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "CRUCIBLE" in result.output

    def test_cli_no_command_shows_help(self, runner):
        result = runner.invoke(cli, [])
        # click exits with code 0 when showing help via the group
        # but some versions exit 0, some 2 — just check output
        assert "CRUCIBLE" in result.output or result.exit_code == 0


# ===================================================================
# serve command
# ===================================================================


class TestServeCommand:
    def test_serve_help(self, runner):
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output

    def test_serve_invokes_uvicorn(self, runner):
        mock_uvicorn = MagicMock()
        mock_app = MagicMock()

        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}), \
             patch("presentation.api.app.create_app", return_value=mock_app), \
             patch("presentation.cli.main.Settings") as mock_settings_cls:
            mock_settings_cls.from_env.return_value = MagicMock()
            result = runner.invoke(cli, ["serve", "--host", "127.0.0.1", "--port", "9000"])
            assert result.exit_code == 0
            mock_uvicorn.run.assert_called_once_with(mock_app, host="127.0.0.1", port=9000)


# ===================================================================
# register command
# ===================================================================


class TestRegisterCommand:
    def test_register_help(self, runner):
        result = runner.invoke(cli, ["register", "--help"])
        assert result.exit_code == 0
        assert "--name" in result.output
        assert "--description" in result.output
        assert "--type" in result.output

    @patch("presentation.cli.main.Container")
    def test_register_success(self, mock_container_cls, runner):
        mock_container = _make_mock_container()
        mock_container_cls.return_value = mock_container

        agent = _make_agent()
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = agent
        mock_container.register_agent.return_value = mock_use_case

        result = runner.invoke(
            cli,
            ["register", "--name", "my-agent", "--description", "desc", "--type", "coding"],
        )
        assert result.exit_code == 0
        assert "Agent registered" in result.output
        assert "test-agent" in result.output
        mock_use_case.execute.assert_called_once_with(
            name="my-agent", description="desc", agent_type="coding",
        )

    @patch("presentation.cli.main.Container")
    def test_register_missing_required_options(self, mock_container_cls, runner):
        result = runner.invoke(cli, ["register", "--name", "only-name"])
        assert result.exit_code != 0


# ===================================================================
# agents command
# ===================================================================


class TestAgentsCommand:
    @patch("presentation.cli.main.Container")
    def test_agents_list_empty(self, mock_container_cls, runner):
        mock_container = _make_mock_container()
        mock_container_cls.return_value = mock_container

        mock_query = AsyncMock()
        mock_query.execute.return_value = []
        mock_container.list_agents_query.return_value = mock_query

        result = runner.invoke(cli, ["agents"])
        assert result.exit_code == 0
        assert "No agents registered" in result.output

    @patch("presentation.cli.main.Container")
    def test_agents_list_populated(self, mock_container_cls, runner):
        mock_container = _make_mock_container()
        mock_container_cls.return_value = mock_container

        agents = [
            _make_agent("agent-alpha", "coding", "a-1"),
            _make_agent("agent-beta", "research", "a-2"),
        ]
        mock_query = AsyncMock()
        mock_query.execute.return_value = agents
        mock_container.list_agents_query.return_value = mock_query

        result = runner.invoke(cli, ["agents"])
        assert result.exit_code == 0
        assert "agent-alpha" in result.output
        assert "coding" in result.output
        assert "agent-beta" in result.output
        assert "research" in result.output


# ===================================================================
# scores command
# ===================================================================


class TestScoresCommand:
    def test_scores_help(self, runner):
        result = runner.invoke(cli, ["scores", "--help"])
        assert result.exit_code == 0
        assert "--limit" in result.output

    @patch("presentation.cli.main.Container")
    def test_scores_no_evaluations(self, mock_container_cls, runner):
        mock_container = _make_mock_container()
        mock_container_cls.return_value = mock_container

        from application.queries.get_scores import ScoreTimeline

        mock_query = AsyncMock()
        mock_query.execute.return_value = ScoreTimeline(evaluations=[], regressions=[])
        mock_container.get_scores_query.return_value = mock_query

        result = runner.invoke(cli, ["scores", "agent-1"])
        assert result.exit_code == 0
        assert "No evaluations found" in result.output

    @patch("presentation.cli.main.Container")
    def test_scores_with_evaluations(self, mock_container_cls, runner):
        mock_container = _make_mock_container()
        mock_container_cls.return_value = mock_container

        from application.queries.get_scores import ScoreTimeline

        evaluations = [
            _make_evaluation("e-1", 0.9, True),
            _make_evaluation("e-2", 0.7, True),
        ]
        mock_query = AsyncMock()
        mock_query.execute.return_value = ScoreTimeline(
            evaluations=evaluations, regressions=[],
        )
        mock_container.get_scores_query.return_value = mock_query

        result = runner.invoke(cli, ["scores", "agent-1", "--limit", "5"])
        assert result.exit_code == 0
        assert "PASS" in result.output
        assert "0.900" in result.output
        assert "0.700" in result.output

    @patch("presentation.cli.main.Container")
    def test_scores_with_regressions(self, mock_container_cls, runner):
        mock_container = _make_mock_container()
        mock_container_cls.return_value = mock_container

        from application.queries.get_scores import ScoreTimeline

        evaluations = [_make_evaluation("e-1", 0.9, True)]
        regression = RegressionResult(
            current_score=0.5,
            baseline_score=0.9,
            delta=-0.4,
            regressed=True,
            severity="critical",
        )
        mock_query = AsyncMock()
        mock_query.execute.return_value = ScoreTimeline(
            evaluations=evaluations,
            regressions=[("e-1", regression)],
        )
        mock_container.get_scores_query.return_value = mock_query

        result = runner.invoke(cli, ["scores", "agent-1"])
        assert result.exit_code == 0
        assert "Regressions detected" in result.output
        assert "CRITICAL" in result.output
        assert "-0.400" in result.output

    @patch("presentation.cli.main.Container")
    def test_scores_with_fail(self, mock_container_cls, runner):
        mock_container = _make_mock_container()
        mock_container_cls.return_value = mock_container

        from application.queries.get_scores import ScoreTimeline

        evaluations = [_make_evaluation("e-fail", 0.3, False)]
        mock_query = AsyncMock()
        mock_query.execute.return_value = ScoreTimeline(
            evaluations=evaluations, regressions=[],
        )
        mock_container.get_scores_query.return_value = mock_query

        result = runner.invoke(cli, ["scores", "agent-1"])
        assert result.exit_code == 0
        assert "FAIL" in result.output


# ===================================================================
# load-rubric command
# ===================================================================


class TestLoadRubricCommand:
    def test_load_rubric_help(self, runner):
        result = runner.invoke(cli, ["load-rubric", "--help"])
        assert result.exit_code == 0

    @patch("presentation.cli.main.Container")
    def test_load_rubric_success(self, mock_container_cls, runner, tmp_path):
        # Write a YAML rubric file
        rubric_file = tmp_path / "rubric.yaml"
        rubric_file.write_text(
            "name: yaml-rubric\n"
            "description: Loaded from YAML\n"
            "agent_type: coding\n"
            "owner: tester\n"
            "dimensions:\n"
            "  - name: accuracy\n"
            "    description: Is it accurate?\n"
            "    scoring_method: llm_judge\n"
            "    weight: 1.0\n"
            "    pass_threshold: 0.7\n"
            "    judge_prompt: Rate accuracy\n"
        )

        mock_container = _make_mock_container()
        mock_container_cls.return_value = mock_container

        rubric = _make_rubric()
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = rubric
        mock_container.create_rubric.return_value = mock_use_case

        result = runner.invoke(cli, ["load-rubric", str(rubric_file)])
        assert result.exit_code == 0
        assert "Rubric created" in result.output
        assert "test-rubric" in result.output

    def test_load_rubric_file_not_found(self, runner):
        result = runner.invoke(cli, ["load-rubric", "/nonexistent/rubric.yaml"])
        assert result.exit_code != 0
