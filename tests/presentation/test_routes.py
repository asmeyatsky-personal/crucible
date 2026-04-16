"""Tests for all FastAPI presentation routes.

Uses a real SQLite database (via tmp_path) for integration-style testing.
The LLM judge is mocked where needed (evaluate_run).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from domain.value_objects.score import Confidence, DimensionScore
from tests.presentation.conftest import (
    NOW,
    make_agent,
    make_composite_score,
    make_dimension_score,
    make_evaluation,
    make_report,
    make_rubric,
    make_trace,
)


# ===================================================================
# Health endpoint
# ===================================================================


class TestHealthEndpoint:
    async def test_health_returns_200(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "crucible"
        assert data["version"] == "1.0.0"


# ===================================================================
# Agent routes
# ===================================================================


class TestAgentRoutes:
    async def test_register_agent_success(self, client):
        resp = await client.post(
            "/api/v1/agents",
            json={
                "name": "my-agent",
                "description": "A testing agent",
                "agent_type": "coding",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "my-agent"
        assert data["description"] == "A testing agent"
        assert data["agent_type"] == "coding"
        assert "id" in data
        assert "created_at" in data

    async def test_register_agent_with_metadata(self, client):
        resp = await client.post(
            "/api/v1/agents",
            json={
                "name": "meta-agent",
                "description": "Has metadata",
                "agent_type": "research",
                "metadata": {"version": "1.0"},
            },
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "meta-agent"

    async def test_register_agent_duplicate_name_returns_400(self, client):
        payload = {
            "name": "unique-agent",
            "description": "First one",
            "agent_type": "coding",
        }
        resp1 = await client.post("/api/v1/agents", json=payload)
        assert resp1.status_code == 200

        resp2 = await client.post("/api/v1/agents", json=payload)
        assert resp2.status_code == 400
        assert "already exists" in resp2.json()["detail"]

    async def test_list_agents_empty(self, client):
        resp = await client.get("/api/v1/agents")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_agents_after_registration(self, client):
        await client.post(
            "/api/v1/agents",
            json={
                "name": "listed-agent",
                "description": "To be listed",
                "agent_type": "coding",
            },
        )
        resp = await client.get("/api/v1/agents")
        assert resp.status_code == 200
        agents = resp.json()
        assert len(agents) == 1
        assert agents[0]["name"] == "listed-agent"
        assert "rubric_ids" in agents[0]

    async def test_get_agent_success(self, client):
        # Register first
        create_resp = await client.post(
            "/api/v1/agents",
            json={
                "name": "get-agent",
                "description": "Get me",
                "agent_type": "research",
            },
        )
        agent_id = create_resp.json()["id"]

        resp = await client.get(f"/api/v1/agents/{agent_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == agent_id
        assert data["name"] == "get-agent"
        assert "metadata" in data
        assert "updated_at" in data

    async def test_get_agent_not_found_returns_404(self, client):
        resp = await client.get("/api/v1/agents/nonexistent-id")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


# ===================================================================
# Trace routes
# ===================================================================


class TestTraceRoutes:
    async def _register_agent(self, client, name="trace-agent"):
        resp = await client.post(
            "/api/v1/agents",
            json={"name": name, "description": "For traces", "agent_type": "coding"},
        )
        return resp.json()["id"]

    async def test_capture_trace_success(self, client):
        agent_id = await self._register_agent(client)
        resp = await client.post(
            "/api/v1/traces",
            json={
                "agent_id": agent_id,
                "run_id": "run-001",
                "model_id": "claude-sonnet-4-6",
                "steps": [
                    {
                        "step_index": 0,
                        "step_type": "user_message",
                        "content": "Hello",
                        "timestamp": NOW.isoformat(),
                    },
                    {
                        "step_index": 1,
                        "step_type": "assistant_response",
                        "content": "Hi there!",
                        "timestamp": NOW.isoformat(),
                    },
                ],
                "total_tokens": 100,
                "total_latency_ms": 250.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == agent_id
        assert data["run_id"] == "run-001"
        assert data["step_count"] == 2
        assert data["total_tokens"] == 100
        assert "id" in data

    async def test_capture_trace_unknown_agent_returns_400(self, client):
        resp = await client.post(
            "/api/v1/traces",
            json={
                "agent_id": "no-such-agent",
                "run_id": "run-002",
                "model_id": "claude-sonnet-4-6",
                "steps": [
                    {
                        "step_index": 0,
                        "step_type": "user_message",
                        "content": "Hello",
                        "timestamp": NOW.isoformat(),
                    },
                ],
            },
        )
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"].lower()

    async def test_get_trace_success(self, client):
        agent_id = await self._register_agent(client, "get-trace-agent")
        create_resp = await client.post(
            "/api/v1/traces",
            json={
                "agent_id": agent_id,
                "run_id": "run-003",
                "model_id": "claude-sonnet-4-6",
                "steps": [
                    {
                        "step_index": 0,
                        "step_type": "user_message",
                        "content": "Test content",
                        "timestamp": NOW.isoformat(),
                    },
                ],
                "total_tokens": 50,
                "total_latency_ms": 100.0,
            },
        )
        trace_id = create_resp.json()["id"]

        resp = await client.get(f"/api/v1/traces/{trace_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == trace_id
        assert data["run_id"] == "run-003"
        assert len(data["steps"]) == 1
        assert data["steps"][0]["step_type"] == "user_message"

    async def test_get_trace_not_found_returns_404(self, client):
        resp = await client.get("/api/v1/traces/nonexistent-trace")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    async def test_list_traces_by_agent(self, client):
        agent_id = await self._register_agent(client, "list-trace-agent")
        # Capture two traces
        for run_id in ("run-A", "run-B"):
            await client.post(
                "/api/v1/traces",
                json={
                    "agent_id": agent_id,
                    "run_id": run_id,
                    "model_id": "claude-sonnet-4-6",
                    "steps": [
                        {
                            "step_index": 0,
                            "step_type": "user_message",
                            "content": "msg",
                            "timestamp": NOW.isoformat(),
                        },
                    ],
                },
            )

        resp = await client.get(f"/api/v1/traces/agent/{agent_id}")
        assert resp.status_code == 200
        traces = resp.json()
        assert len(traces) == 2

    async def test_list_traces_by_agent_empty(self, client):
        resp = await client.get("/api/v1/traces/agent/no-agent")
        assert resp.status_code == 200
        assert resp.json() == []


# ===================================================================
# Rubric routes
# ===================================================================


class TestRubricRoutes:
    def _rubric_payload(self, name="test-rubric"):
        return {
            "name": name,
            "description": "A test rubric",
            "dimensions": [
                {
                    "name": "accuracy",
                    "description": "Is the answer correct?",
                    "scoring_method": "llm_judge",
                    "weight": 1.0,
                    "pass_threshold": 0.7,
                    "judge_prompt": "Rate accuracy from 0 to 1.",
                },
            ],
            "agent_type": "coding",
            "owner": "tester",
        }

    async def test_create_rubric_success(self, client):
        resp = await client.post("/api/v1/rubrics", json=self._rubric_payload())
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test-rubric"
        assert data["version"] == 1
        assert data["dimension_count"] == 1
        assert data["agent_type"] == "coding"
        assert "id" in data

    async def test_create_rubric_duplicate_returns_400(self, client):
        payload = self._rubric_payload("dup-rubric")
        resp1 = await client.post("/api/v1/rubrics", json=payload)
        assert resp1.status_code == 200

        resp2 = await client.post("/api/v1/rubrics", json=payload)
        assert resp2.status_code == 400
        assert "already exists" in resp2.json()["detail"]

    async def test_create_rubric_no_dimensions_returns_400(self, client):
        payload = self._rubric_payload("empty-rubric")
        payload["dimensions"] = []
        resp = await client.post("/api/v1/rubrics", json=payload)
        assert resp.status_code == 400

    async def test_list_rubrics_empty(self, client):
        resp = await client.get("/api/v1/rubrics")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_rubrics_after_creation(self, client):
        await client.post("/api/v1/rubrics", json=self._rubric_payload("listed-rubric"))
        resp = await client.get("/api/v1/rubrics")
        assert resp.status_code == 200
        rubrics = resp.json()
        assert len(rubrics) == 1
        assert rubrics[0]["name"] == "listed-rubric"

    async def test_get_rubric_success(self, client):
        create_resp = await client.post(
            "/api/v1/rubrics", json=self._rubric_payload("get-rubric"),
        )
        rubric_id = create_resp.json()["id"]

        resp = await client.get(f"/api/v1/rubrics/{rubric_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == rubric_id
        assert data["name"] == "get-rubric"
        assert len(data["dimensions"]) == 1
        assert data["dimensions"][0]["name"] == "accuracy"
        assert data["owner"] == "tester"

    async def test_get_rubric_not_found_returns_404(self, client):
        resp = await client.get("/api/v1/rubrics/nonexistent")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


# ===================================================================
# Evaluation routes
# ===================================================================


class TestEvaluationRoutes:
    async def _setup_trace_and_rubric(self, client):
        """Create an agent, trace, and rubric, return their IDs."""
        agent_resp = await client.post(
            "/api/v1/agents",
            json={"name": "eval-agent", "description": "For evals", "agent_type": "coding"},
        )
        agent_id = agent_resp.json()["id"]

        trace_resp = await client.post(
            "/api/v1/traces",
            json={
                "agent_id": agent_id,
                "run_id": "eval-run-1",
                "model_id": "claude-sonnet-4-6",
                "steps": [
                    {
                        "step_index": 0,
                        "step_type": "user_message",
                        "content": "Solve this problem",
                        "timestamp": NOW.isoformat(),
                    },
                    {
                        "step_index": 1,
                        "step_type": "assistant_response",
                        "content": "Here is the solution",
                        "timestamp": NOW.isoformat(),
                    },
                ],
                "total_tokens": 200,
                "total_latency_ms": 500.0,
            },
        )
        trace_id = trace_resp.json()["id"]

        rubric_resp = await client.post(
            "/api/v1/rubrics",
            json={
                "name": "eval-rubric",
                "description": "Eval rubric",
                "dimensions": [
                    {
                        "name": "correctness",
                        "description": "Is it correct?",
                        "scoring_method": "llm_judge",
                        "weight": 1.0,
                        "pass_threshold": 0.7,
                        "judge_prompt": "Rate correctness",
                    },
                ],
            },
        )
        rubric_id = rubric_resp.json()["id"]

        return agent_id, trace_id, rubric_id

    async def test_evaluate_run_success(self, client, container):
        agent_id, trace_id, rubric_id = await self._setup_trace_and_rubric(client)

        # Mock the judge to return a deterministic score
        mock_judge = AsyncMock()
        mock_judge.evaluate_dimension.return_value = DimensionScore(
            dimension_name="correctness",
            score=0.85,
            passed=True,
            rationale="Good answer",
            confidence=Confidence.HIGH,
        )
        container.judge = mock_judge

        resp = await client.post(
            "/api/v1/evaluations",
            json={
                "trace_id": trace_id,
                "rubric_id": rubric_id,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["trace_id"] == trace_id
        assert data["rubric_id"] == rubric_id
        assert data["agent_id"] == agent_id
        assert data["composite_score"]["value"] == 0.85
        assert data["composite_score"]["passed"] is True
        assert len(data["dimension_scores"]) == 1
        assert data["dimension_scores"][0]["dimension_name"] == "correctness"
        assert "id" in data
        assert "evaluated_at" in data

    async def test_evaluate_run_trace_not_found_returns_400(self, client):
        # Create a rubric but no trace
        await client.post(
            "/api/v1/rubrics",
            json={
                "name": "orphan-rubric",
                "description": "No trace for this",
                "dimensions": [
                    {
                        "name": "dim1",
                        "description": "d",
                        "scoring_method": "llm_judge",
                        "weight": 1.0,
                        "pass_threshold": 0.7,
                        "judge_prompt": "judge",
                    },
                ],
            },
        )
        rubric_id = (await client.get("/api/v1/rubrics")).json()[0]["id"]

        resp = await client.post(
            "/api/v1/evaluations",
            json={"trace_id": "no-such-trace", "rubric_id": rubric_id},
        )
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"].lower()

    async def test_evaluate_run_rubric_not_found_returns_400(self, client, container):
        # Register an agent and capture a trace
        agent_resp = await client.post(
            "/api/v1/agents",
            json={
                "name": "rubric-miss-agent",
                "description": "test",
                "agent_type": "coding",
            },
        )
        agent_id = agent_resp.json()["id"]
        trace_resp = await client.post(
            "/api/v1/traces",
            json={
                "agent_id": agent_id,
                "run_id": "run-x",
                "model_id": "claude-sonnet-4-6",
                "steps": [
                    {
                        "step_index": 0,
                        "step_type": "user_message",
                        "content": "hi",
                        "timestamp": NOW.isoformat(),
                    },
                ],
            },
        )
        trace_id = trace_resp.json()["id"]

        resp = await client.post(
            "/api/v1/evaluations",
            json={"trace_id": trace_id, "rubric_id": "no-such-rubric"},
        )
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"].lower()

    async def test_get_evaluation_success(self, client, container):
        agent_id, trace_id, rubric_id = await self._setup_trace_and_rubric(client)

        mock_judge = AsyncMock()
        mock_judge.evaluate_dimension.return_value = DimensionScore(
            dimension_name="correctness",
            score=0.9,
            passed=True,
            rationale="Excellent",
            confidence=Confidence.HIGH,
        )
        container.judge = mock_judge

        create_resp = await client.post(
            "/api/v1/evaluations",
            json={"trace_id": trace_id, "rubric_id": rubric_id},
        )
        eval_id = create_resp.json()["id"]

        resp = await client.get(f"/api/v1/evaluations/{eval_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == eval_id
        assert data["composite_score"]["value"] == 0.9

    async def test_get_evaluation_not_found_returns_404(self, client):
        resp = await client.get("/api/v1/evaluations/nonexistent")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    async def test_get_agent_scores_empty(self, client):
        resp = await client.get("/api/v1/evaluations/agent/agent-1/scores")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == "agent-1"
        assert data["evaluation_count"] == 0
        assert data["scores"] == []
        assert data["regressions"] == []

    async def test_get_agent_scores_with_evaluations(self, client, container):
        # Set up agent, trace, rubric and run an evaluation
        agent_resp = await client.post(
            "/api/v1/agents",
            json={"name": "scores-agent", "description": "s", "agent_type": "coding"},
        )
        agent_id = agent_resp.json()["id"]

        trace_resp = await client.post(
            "/api/v1/traces",
            json={
                "agent_id": agent_id,
                "run_id": "scores-run",
                "model_id": "claude-sonnet-4-6",
                "steps": [
                    {
                        "step_index": 0,
                        "step_type": "user_message",
                        "content": "hi",
                        "timestamp": NOW.isoformat(),
                    },
                ],
            },
        )
        trace_id = trace_resp.json()["id"]

        rubric_resp = await client.post(
            "/api/v1/rubrics",
            json={
                "name": "scores-rubric",
                "description": "r",
                "dimensions": [
                    {
                        "name": "quality",
                        "description": "q",
                        "scoring_method": "llm_judge",
                        "weight": 1.0,
                        "pass_threshold": 0.5,
                        "judge_prompt": "rate quality",
                    },
                ],
            },
        )
        rubric_id = rubric_resp.json()["id"]

        mock_judge = AsyncMock()
        mock_judge.evaluate_dimension.return_value = DimensionScore(
            dimension_name="quality",
            score=0.8,
            passed=True,
            rationale="Good",
            confidence=Confidence.MEDIUM,
        )
        container.judge = mock_judge

        await client.post(
            "/api/v1/evaluations",
            json={"trace_id": trace_id, "rubric_id": rubric_id},
        )

        resp = await client.get(f"/api/v1/evaluations/agent/{agent_id}/scores")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == agent_id
        assert data["evaluation_count"] == 1
        assert len(data["scores"]) == 1
        assert data["scores"][0]["composite_score"] == 0.8
        assert data["scores"][0]["passed"] is True


# ===================================================================
# Report routes
# ===================================================================


class TestReportRoutes:
    async def _create_evaluation(self, client, container):
        """Create a full agent -> trace -> rubric -> evaluation pipeline."""
        agent_resp = await client.post(
            "/api/v1/agents",
            json={"name": "report-agent", "description": "r", "agent_type": "coding"},
        )
        agent_id = agent_resp.json()["id"]

        trace_resp = await client.post(
            "/api/v1/traces",
            json={
                "agent_id": agent_id,
                "run_id": "report-run",
                "model_id": "claude-sonnet-4-6",
                "steps": [
                    {
                        "step_index": 0,
                        "step_type": "user_message",
                        "content": "test",
                        "timestamp": NOW.isoformat(),
                    },
                ],
            },
        )
        trace_id = trace_resp.json()["id"]

        rubric_resp = await client.post(
            "/api/v1/rubrics",
            json={
                "name": "report-rubric",
                "description": "r",
                "dimensions": [
                    {
                        "name": "dim",
                        "description": "d",
                        "scoring_method": "llm_judge",
                        "weight": 1.0,
                        "pass_threshold": 0.5,
                        "judge_prompt": "rate",
                    },
                ],
            },
        )
        rubric_id = rubric_resp.json()["id"]

        mock_judge = AsyncMock()
        mock_judge.evaluate_dimension.return_value = DimensionScore(
            dimension_name="dim",
            score=0.75,
            passed=True,
            rationale="OK",
            confidence=Confidence.MEDIUM,
        )
        container.judge = mock_judge

        eval_resp = await client.post(
            "/api/v1/evaluations",
            json={"trace_id": trace_id, "rubric_id": rubric_id},
        )
        eval_id = eval_resp.json()["id"]
        return agent_id, eval_id

    async def test_generate_report_success(self, client, container):
        agent_id, eval_id = await self._create_evaluation(client, container)

        resp = await client.post(
            "/api/v1/reports",
            json={
                "agent_id": agent_id,
                "title": "My Report",
                "evaluation_ids": [eval_id],
                "export_format": "json",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "My Report"
        assert data["agent_id"] == agent_id
        assert data["export_format"] == "json"
        assert data["evaluation_count"] == 1
        assert "content" in data
        assert "id" in data

    async def test_generate_report_no_evaluations_returns_400(self, client):
        resp = await client.post(
            "/api/v1/reports",
            json={
                "agent_id": "no-evals-agent",
                "title": "Empty Report",
                "export_format": "json",
            },
        )
        assert resp.status_code == 400
        assert "no evaluations" in resp.json()["detail"].lower()

    async def test_generate_report_missing_evaluation_id_returns_400(self, client):
        resp = await client.post(
            "/api/v1/reports",
            json={
                "agent_id": "some-agent",
                "title": "Bad Report",
                "evaluation_ids": ["nonexistent-eval"],
                "export_format": "json",
            },
        )
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"].lower()
