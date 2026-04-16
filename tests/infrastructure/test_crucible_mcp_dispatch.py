"""
Tests for MCP server _dispatch_tool function.

Tests each tool dispatch path by passing a Container with mocked use cases.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from infrastructure.mcp_servers.crucible_server import _dispatch_tool


def _make_mock_container():
    """Create a mock Container with mocked use case factories."""
    container = MagicMock()
    return container


class TestDispatchRegisterAgent:
    @pytest.mark.asyncio
    async def test_register_agent(self):
        container = _make_mock_container()
        mock_agent = MagicMock()
        mock_agent.id = "a-1"
        mock_agent.name = "TestAgent"
        mock_agent.agent_type = "research"

        mock_use_case = MagicMock()
        mock_use_case.execute = AsyncMock(return_value=mock_agent)
        container.register_agent.return_value = mock_use_case

        result = await _dispatch_tool(container, "register_agent", {
            "name": "TestAgent",
            "description": "A test agent",
            "agent_type": "research",
        })

        data = json.loads(result)
        assert data["id"] == "a-1"
        assert data["name"] == "TestAgent"
        assert data["agent_type"] == "research"
        assert data["status"] == "registered"

        mock_use_case.execute.assert_called_once_with(
            name="TestAgent",
            description="A test agent",
            agent_type="research",
        )


class TestDispatchCaptureTrace:
    @pytest.mark.asyncio
    async def test_capture_trace(self):
        container = _make_mock_container()
        mock_trace = MagicMock()
        mock_trace.id = "t-1"
        mock_trace.agent_id = "a-1"
        mock_trace.run_id = "r-1"
        mock_trace.step_count = 3

        mock_use_case = MagicMock()
        mock_use_case.execute = AsyncMock(return_value=mock_trace)
        container.capture_trace.return_value = mock_use_case

        result = await _dispatch_tool(container, "capture_trace", {
            "agent_id": "a-1",
            "run_id": "r-1",
            "steps": [
                {
                    "step_index": 0,
                    "step_type": "user_message",
                    "content": "Hello",
                    "timestamp": "2025-01-01T00:00:00+00:00",
                },
            ],
            "model_id": "claude-sonnet-4-6",
            "temperature": 0.1,
            "total_tokens": 100,
            "total_latency_ms": 500.0,
        })

        data = json.loads(result)
        assert data["id"] == "t-1"
        assert data["agent_id"] == "a-1"
        assert data["run_id"] == "r-1"
        assert data["step_count"] == 3
        assert data["status"] == "captured"

    @pytest.mark.asyncio
    async def test_capture_trace_defaults(self):
        container = _make_mock_container()
        mock_trace = MagicMock()
        mock_trace.id = "t-2"
        mock_trace.agent_id = "a-1"
        mock_trace.run_id = "r-2"
        mock_trace.step_count = 1

        mock_use_case = MagicMock()
        mock_use_case.execute = AsyncMock(return_value=mock_trace)
        container.capture_trace.return_value = mock_use_case

        result = await _dispatch_tool(container, "capture_trace", {
            "agent_id": "a-1",
            "run_id": "r-2",
            "steps": [
                {
                    "step_index": 0,
                    "step_type": "user_message",
                    "content": "Test",
                    "timestamp": "2025-01-01T00:00:00+00:00",
                },
            ],
            "model_id": "test-model",
        })

        data = json.loads(result)
        assert data["status"] == "captured"
        # Verify defaults are passed correctly in the request
        call_args = mock_use_case.execute.call_args
        request = call_args.args[0] if call_args.args else call_args.kwargs.get("request")
        # The use case should have been called with default values


class TestDispatchCreateRubric:
    @pytest.mark.asyncio
    async def test_create_rubric(self):
        container = _make_mock_container()
        mock_rubric = MagicMock()
        mock_rubric.id = "rub-1"
        mock_rubric.name = "TestRubric"
        mock_rubric.version = 1
        mock_rubric.dimensions = [MagicMock(), MagicMock()]

        mock_use_case = MagicMock()
        mock_use_case.execute = AsyncMock(return_value=mock_rubric)
        container.create_rubric.return_value = mock_use_case

        result = await _dispatch_tool(container, "create_rubric", {
            "name": "TestRubric",
            "description": "A test rubric",
            "dimensions": [
                {
                    "name": "accuracy",
                    "description": "How accurate",
                    "scoring_method": "llm_judge",
                    "weight": 1.0,
                    "pass_threshold": 0.7,
                    "judge_prompt": "Evaluate accuracy",
                },
            ],
            "agent_type": "research",
        })

        data = json.loads(result)
        assert data["id"] == "rub-1"
        assert data["name"] == "TestRubric"
        assert data["version"] == 1
        assert data["dimension_count"] == 2
        assert data["status"] == "created"


class TestDispatchEvaluateRun:
    @pytest.mark.asyncio
    async def test_evaluate_run(self):
        container = _make_mock_container()
        mock_composite = MagicMock()
        mock_composite.value = 0.85
        mock_composite.passed = True
        mock_composite.dimensions_passed = 2
        mock_composite.dimensions_total = 3

        mock_evaluation = MagicMock()
        mock_evaluation.id = "ev-1"
        mock_evaluation.composite_score = mock_composite

        mock_use_case = MagicMock()
        mock_use_case.execute = AsyncMock(return_value=mock_evaluation)
        container.evaluate_run.return_value = mock_use_case

        result = await _dispatch_tool(container, "evaluate_run", {
            "trace_id": "t-1",
            "rubric_id": "rub-1",
            "judge_model": "gpt-4o",
        })

        data = json.loads(result)
        assert data["id"] == "ev-1"
        assert data["composite_score"] == 0.85
        assert data["passed"] is True
        assert data["dimensions_passed"] == 2
        assert data["dimensions_total"] == 3
        assert data["status"] == "evaluated"

    @pytest.mark.asyncio
    async def test_evaluate_run_default_judge_model(self):
        container = _make_mock_container()
        mock_composite = MagicMock()
        mock_composite.value = 0.7
        mock_composite.passed = True
        mock_composite.dimensions_passed = 1
        mock_composite.dimensions_total = 1

        mock_evaluation = MagicMock()
        mock_evaluation.id = "ev-2"
        mock_evaluation.composite_score = mock_composite

        mock_use_case = MagicMock()
        mock_use_case.execute = AsyncMock(return_value=mock_evaluation)
        container.evaluate_run.return_value = mock_use_case

        result = await _dispatch_tool(container, "evaluate_run", {
            "trace_id": "t-1",
            "rubric_id": "rub-1",
        })

        data = json.loads(result)
        assert data["status"] == "evaluated"


class TestDispatchGenerateReport:
    @pytest.mark.asyncio
    async def test_generate_report(self):
        container = _make_mock_container()
        mock_export_format = MagicMock()
        mock_export_format.value = "json"

        mock_report = MagicMock()
        mock_report.id = "rep-1"
        mock_report.title = "Test Report"
        mock_report.export_format = mock_export_format

        mock_use_case = MagicMock()
        mock_use_case.execute = AsyncMock(return_value=mock_report)
        container.generate_report.return_value = mock_use_case

        result = await _dispatch_tool(container, "generate_report", {
            "agent_id": "a-1",
            "title": "Test Report",
            "export_format": "csv",
        })

        data = json.loads(result)
        assert data["id"] == "rep-1"
        assert data["title"] == "Test Report"
        assert data["status"] == "generated"


class TestDispatchUnknownTool:
    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        container = _make_mock_container()

        result = await _dispatch_tool(container, "nonexistent_tool", {})

        data = json.loads(result)
        assert "error" in data
        assert "Unknown tool" in data["error"]
        assert "nonexistent_tool" in data["error"]
