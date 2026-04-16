"""
Tests for ClaudeJudgeAdapter.

Tests prompt building, trajectory formatting, and response parsing
without making real API calls.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from domain.value_objects.judge_config import JudgeConfig, JudgeModel
from domain.value_objects.score import Confidence
from infrastructure.adapters.claude_judge_adapter import ClaudeJudgeAdapter
from tests.infrastructure.conftest import make_dimension, make_trace, make_step, make_tool_call
from domain.value_objects.trajectory_step import StepType


class TestDefaultSystemPrompt:
    def test_returns_nonempty_string(self):
        adapter = ClaudeJudgeAdapter(api_key="test-key")
        prompt = adapter._default_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_contains_json_instructions(self):
        adapter = ClaudeJudgeAdapter(api_key="test-key")
        prompt = adapter._default_system_prompt()
        assert "JSON" in prompt
        assert "score" in prompt
        assert "rationale" in prompt
        assert "confidence" in prompt

    def test_mentions_crucible(self):
        adapter = ClaudeJudgeAdapter(api_key="test-key")
        prompt = adapter._default_system_prompt()
        assert "CRUCIBLE" in prompt


class TestBuildEvaluationPrompt:
    def test_contains_dimension_info(self):
        adapter = ClaudeJudgeAdapter(api_key="test-key")
        trace = make_trace()
        dimension = make_dimension(name="test_dim", description="Test description", pass_threshold=0.8)

        prompt = adapter._build_evaluation_prompt(trace, dimension)

        assert "test_dim" in prompt
        assert "Test description" in prompt
        assert "0.8" in prompt

    def test_contains_judge_prompt(self):
        adapter = ClaudeJudgeAdapter(api_key="test-key")
        trace = make_trace()
        dimension = make_dimension(judge_prompt="Evaluate the code quality carefully.")

        prompt = adapter._build_evaluation_prompt(trace, dimension)

        assert "Evaluate the code quality carefully." in prompt

    def test_contains_trajectory_text(self):
        adapter = ClaudeJudgeAdapter(api_key="test-key")
        trace = make_trace(agent_id="agent-42", model_id="gpt-4o")
        dimension = make_dimension()

        prompt = adapter._build_evaluation_prompt(trace, dimension)

        assert "agent-42" in prompt
        assert "gpt-4o" in prompt

    def test_contains_json_instruction(self):
        adapter = ClaudeJudgeAdapter(api_key="test-key")
        prompt = adapter._build_evaluation_prompt(make_trace(), make_dimension())
        assert "JSON" in prompt


class TestFormatTrajectory:
    def test_includes_agent_and_model_info(self):
        adapter = ClaudeJudgeAdapter(api_key="test-key")
        trace = make_trace(agent_id="agent-99", model_id="test-model", total_tokens=1234)
        result = adapter._format_trajectory(trace)

        assert "agent-99" in result
        assert "test-model" in result
        assert "1234" in result

    def test_includes_step_content(self):
        adapter = ClaudeJudgeAdapter(api_key="test-key")
        step = make_step(0, content="Hello world step content")
        trace = make_trace(steps=(step,))
        result = adapter._format_trajectory(trace)

        assert "Hello world step content" in result

    def test_includes_tool_call_info(self):
        adapter = ClaudeJudgeAdapter(api_key="test-key")
        tc = make_tool_call(name="web_search", success=True, latency_ms=150.0)
        step = make_step(0, tool_calls=(tc,))
        trace = make_trace(steps=(step,))
        result = adapter._format_trajectory(trace)

        assert "web_search" in result
        assert "OK" in result
        assert "150" in result

    def test_includes_failed_tool_call(self):
        adapter = ClaudeJudgeAdapter(api_key="test-key")
        tc = make_tool_call(name="code_exec", success=False, error="Timeout exceeded")
        step = make_step(0, tool_calls=(tc,))
        trace = make_trace(steps=(step,))
        result = adapter._format_trajectory(trace)

        assert "FAILED" in result
        assert "Timeout exceeded" in result

    def test_step_type_included(self):
        adapter = ClaudeJudgeAdapter(api_key="test-key")
        step = make_step(0, step_type=StepType.REASONING, tool_calls=())
        trace = make_trace(steps=(step,))
        result = adapter._format_trajectory(trace)

        assert "reasoning" in result

    def test_truncates_long_content(self):
        adapter = ClaudeJudgeAdapter(api_key="test-key")
        long_content = "x" * 5000
        step = make_step(0, content=long_content, tool_calls=())
        trace = make_trace(steps=(step,))
        result = adapter._format_trajectory(trace)

        # Content is truncated to 2000 chars
        assert "x" * 2000 in result
        assert "x" * 2001 not in result


class TestParseResponse:
    def _make_response(self, text: str):
        """Create a mock Anthropic response."""
        content_block = MagicMock()
        content_block.text = text
        response = MagicMock()
        response.content = [content_block]
        return response

    def test_parses_valid_json(self):
        adapter = ClaudeJudgeAdapter(api_key="test-key")
        dimension = make_dimension(name="accuracy", pass_threshold=0.7)
        response = self._make_response(json.dumps({
            "score": 0.9,
            "passed": True,
            "rationale": "Very accurate output",
            "confidence": "high",
        }))

        result = adapter._parse_response(response, dimension)

        assert result.dimension_name == "accuracy"
        assert result.score == 0.9
        assert result.passed is True
        assert result.rationale == "Very accurate output"
        assert result.confidence == Confidence.HIGH

    def test_parses_json_in_markdown_code_block(self):
        adapter = ClaudeJudgeAdapter(api_key="test-key")
        dimension = make_dimension(name="quality", pass_threshold=0.5)
        text = '```json\n{"score": 0.75, "passed": true, "rationale": "Good", "confidence": "medium"}\n```'
        response = self._make_response(text)

        result = adapter._parse_response(response, dimension)

        assert result.score == 0.75
        assert result.confidence == Confidence.MEDIUM

    def test_parses_json_in_generic_code_block(self):
        adapter = ClaudeJudgeAdapter(api_key="test-key")
        dimension = make_dimension(name="test", pass_threshold=0.5)
        text = '```\n{"score": 0.6, "rationale": "OK", "confidence": "low"}\n```'
        response = self._make_response(text)

        result = adapter._parse_response(response, dimension)

        assert result.score == 0.6
        assert result.confidence == Confidence.LOW

    def test_fallback_on_invalid_json(self):
        adapter = ClaudeJudgeAdapter(api_key="test-key")
        dimension = make_dimension(name="test", pass_threshold=0.7)
        response = self._make_response("This is not JSON at all")

        result = adapter._parse_response(response, dimension)

        assert result.score == 0.5
        assert result.passed is False
        assert result.confidence == Confidence.LOW
        assert "could not be parsed" in result.rationale

    def test_clamps_score_above_1(self):
        adapter = ClaudeJudgeAdapter(api_key="test-key")
        dimension = make_dimension(name="test", pass_threshold=0.7)
        response = self._make_response(json.dumps({
            "score": 1.5,
            "rationale": "Overscored",
            "confidence": "high",
        }))

        result = adapter._parse_response(response, dimension)
        assert result.score == 1.0

    def test_clamps_score_below_0(self):
        adapter = ClaudeJudgeAdapter(api_key="test-key")
        dimension = make_dimension(name="test", pass_threshold=0.7)
        response = self._make_response(json.dumps({
            "score": -0.5,
            "rationale": "Underscored",
            "confidence": "low",
        }))

        result = adapter._parse_response(response, dimension)
        assert result.score == 0.0

    def test_uses_pass_threshold_for_passed(self):
        adapter = ClaudeJudgeAdapter(api_key="test-key")
        dimension = make_dimension(name="test", pass_threshold=0.8)
        response = self._make_response(json.dumps({
            "score": 0.75,
            "rationale": "Close but not enough",
            "confidence": "high",
        }))

        result = adapter._parse_response(response, dimension)
        assert result.passed is False  # 0.75 < 0.8

    def test_default_values_on_missing_fields(self):
        adapter = ClaudeJudgeAdapter(api_key="test-key")
        dimension = make_dimension(name="test", pass_threshold=0.3)
        response = self._make_response(json.dumps({}))

        result = adapter._parse_response(response, dimension)
        assert result.score == 0.5
        assert result.rationale == "No rationale provided"
        assert result.confidence == Confidence.MEDIUM


class TestEvaluateDimension:
    @pytest.mark.asyncio
    async def test_calls_api_with_correct_params(self):
        adapter = ClaudeJudgeAdapter(api_key="test-key")

        mock_response = MagicMock()
        content_block = MagicMock()
        content_block.text = json.dumps({
            "score": 0.9,
            "rationale": "Good",
            "confidence": "high",
        })
        mock_response.content = [content_block]

        adapter._client = MagicMock()
        adapter._client.messages.create = AsyncMock(return_value=mock_response)

        trace = make_trace()
        dimension = make_dimension()
        judge_config = JudgeConfig(primary_model=JudgeModel.CLAUDE_SONNET, temperature=0.0, max_tokens=4096)

        result = await adapter.evaluate_dimension(trace, dimension, judge_config)

        adapter._client.messages.create.assert_called_once()
        call_kwargs = adapter._client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-6"
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_tokens"] == 4096
        assert result.score == 0.9

    @pytest.mark.asyncio
    async def test_uses_custom_system_prompt(self):
        adapter = ClaudeJudgeAdapter(api_key="test-key")

        mock_response = MagicMock()
        content_block = MagicMock()
        content_block.text = json.dumps({"score": 0.5, "rationale": "OK", "confidence": "medium"})
        mock_response.content = [content_block]

        adapter._client = MagicMock()
        adapter._client.messages.create = AsyncMock(return_value=mock_response)

        trace = make_trace()
        dimension = make_dimension()
        judge_config = JudgeConfig(
            primary_model=JudgeModel.CLAUDE_SONNET,
            custom_system_prompt="Custom prompt here",
        )

        await adapter.evaluate_dimension(trace, dimension, judge_config)

        call_kwargs = adapter._client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "Custom prompt here"
