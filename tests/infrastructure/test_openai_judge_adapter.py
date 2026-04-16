"""
Tests for OpenAIJudgeAdapter.

Mirrors claude adapter tests — mock the OpenAI client,
test prompt building and response parsing.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from domain.value_objects.judge_config import JudgeConfig, JudgeModel
from domain.value_objects.score import Confidence
from domain.value_objects.trajectory_step import StepType
from infrastructure.adapters.openai_judge_adapter import OpenAIJudgeAdapter
from tests.infrastructure.conftest import make_dimension, make_trace, make_step, make_tool_call


class TestDefaultSystemPrompt:
    def test_returns_nonempty_string(self):
        adapter = OpenAIJudgeAdapter(api_key="test-key")
        prompt = adapter._default_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_contains_json_instructions(self):
        adapter = OpenAIJudgeAdapter(api_key="test-key")
        prompt = adapter._default_system_prompt()
        assert "JSON" in prompt
        assert "score" in prompt
        assert "rationale" in prompt
        assert "confidence" in prompt

    def test_mentions_crucible(self):
        adapter = OpenAIJudgeAdapter(api_key="test-key")
        prompt = adapter._default_system_prompt()
        assert "CRUCIBLE" in prompt


class TestBuildEvaluationPrompt:
    def test_contains_dimension_info(self):
        adapter = OpenAIJudgeAdapter(api_key="test-key")
        trace = make_trace()
        dimension = make_dimension(name="relevance", description="Is the output relevant", pass_threshold=0.6)

        prompt = adapter._build_evaluation_prompt(trace, dimension)

        assert "relevance" in prompt
        assert "Is the output relevant" in prompt
        assert "0.6" in prompt

    def test_contains_judge_prompt(self):
        adapter = OpenAIJudgeAdapter(api_key="test-key")
        trace = make_trace()
        dimension = make_dimension(judge_prompt="Check for factual errors.")

        prompt = adapter._build_evaluation_prompt(trace, dimension)

        assert "Check for factual errors." in prompt

    def test_contains_trajectory(self):
        adapter = OpenAIJudgeAdapter(api_key="test-key")
        trace = make_trace(agent_id="agent-77")
        dimension = make_dimension()

        prompt = adapter._build_evaluation_prompt(trace, dimension)

        assert "agent-77" in prompt


class TestFormatTrajectory:
    def test_includes_agent_and_model_info(self):
        adapter = OpenAIJudgeAdapter(api_key="test-key")
        trace = make_trace(agent_id="openai-agent", model_id="gpt-4o", total_tokens=999)
        result = adapter._format_trajectory(trace)

        assert "openai-agent" in result
        assert "gpt-4o" in result
        assert "999" in result

    def test_includes_step_content(self):
        adapter = OpenAIJudgeAdapter(api_key="test-key")
        step = make_step(0, content="OpenAI step content here")
        trace = make_trace(steps=(step,))
        result = adapter._format_trajectory(trace)

        assert "OpenAI step content here" in result

    def test_includes_tool_calls(self):
        adapter = OpenAIJudgeAdapter(api_key="test-key")
        tc = make_tool_call(name="file_read", success=True, latency_ms=50.0)
        step = make_step(0, tool_calls=(tc,))
        trace = make_trace(steps=(step,))
        result = adapter._format_trajectory(trace)

        assert "file_read" in result
        assert "OK" in result

    def test_includes_failed_tool_calls(self):
        adapter = OpenAIJudgeAdapter(api_key="test-key")
        tc = make_tool_call(name="api_call", success=False, error="Rate limited")
        step = make_step(0, tool_calls=(tc,))
        trace = make_trace(steps=(step,))
        result = adapter._format_trajectory(trace)

        assert "FAILED" in result
        assert "Rate limited" in result


class TestParseResponse:
    def _make_response(self, text: str | None):
        """Create a mock OpenAI ChatCompletion response."""
        message = MagicMock()
        message.content = text
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        return response

    def test_parses_valid_json(self):
        adapter = OpenAIJudgeAdapter(api_key="test-key")
        dimension = make_dimension(name="accuracy", pass_threshold=0.7)
        response = self._make_response(json.dumps({
            "score": 0.88,
            "passed": True,
            "rationale": "Highly accurate",
            "confidence": "high",
        }))

        result = adapter._parse_response(response, dimension)

        assert result.dimension_name == "accuracy"
        assert result.score == 0.88
        assert result.passed is True
        assert result.rationale == "Highly accurate"
        assert result.confidence == Confidence.HIGH

    def test_fallback_on_invalid_json(self):
        adapter = OpenAIJudgeAdapter(api_key="test-key")
        dimension = make_dimension(name="test", pass_threshold=0.7)
        response = self._make_response("not json")

        result = adapter._parse_response(response, dimension)

        assert result.score == 0.5
        assert result.passed is False
        assert result.confidence == Confidence.LOW
        assert "could not be parsed" in result.rationale

    def test_handles_none_content(self):
        adapter = OpenAIJudgeAdapter(api_key="test-key")
        dimension = make_dimension(name="test", pass_threshold=0.3)
        response = self._make_response(None)

        result = adapter._parse_response(response, dimension)
        # None becomes "{}" which parses to empty dict, uses defaults
        assert result.score == 0.5

    def test_clamps_score_above_1(self):
        adapter = OpenAIJudgeAdapter(api_key="test-key")
        dimension = make_dimension(name="test", pass_threshold=0.5)
        response = self._make_response(json.dumps({"score": 2.0, "rationale": "Over", "confidence": "high"}))

        result = adapter._parse_response(response, dimension)
        assert result.score == 1.0

    def test_clamps_score_below_0(self):
        adapter = OpenAIJudgeAdapter(api_key="test-key")
        dimension = make_dimension(name="test", pass_threshold=0.5)
        response = self._make_response(json.dumps({"score": -1.0, "rationale": "Under", "confidence": "low"}))

        result = adapter._parse_response(response, dimension)
        assert result.score == 0.0

    def test_uses_pass_threshold_for_passed_flag(self):
        adapter = OpenAIJudgeAdapter(api_key="test-key")
        dimension = make_dimension(name="test", pass_threshold=0.9)
        response = self._make_response(json.dumps({"score": 0.85, "rationale": "Close", "confidence": "high"}))

        result = adapter._parse_response(response, dimension)
        assert result.passed is False  # 0.85 < 0.9

    def test_default_values_on_missing_fields(self):
        adapter = OpenAIJudgeAdapter(api_key="test-key")
        dimension = make_dimension(name="test", pass_threshold=0.3)
        response = self._make_response(json.dumps({}))

        result = adapter._parse_response(response, dimension)
        assert result.score == 0.5
        assert result.rationale == "No rationale provided"
        assert result.confidence == Confidence.MEDIUM


class TestEvaluateDimension:
    @pytest.mark.asyncio
    async def test_calls_openai_api_with_correct_params(self):
        adapter = OpenAIJudgeAdapter(api_key="test-key")

        mock_message = MagicMock()
        mock_message.content = json.dumps({
            "score": 0.8,
            "rationale": "Good result",
            "confidence": "high",
        })
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        adapter._client = MagicMock()
        adapter._client.chat.completions.create = AsyncMock(return_value=mock_response)

        trace = make_trace()
        dimension = make_dimension()
        judge_config = JudgeConfig(primary_model=JudgeModel.GPT_4O, temperature=0.1, max_tokens=2048)

        result = await adapter.evaluate_dimension(trace, dimension, judge_config)

        adapter._client.chat.completions.create.assert_called_once()
        call_kwargs = adapter._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["temperature"] == 0.1
        assert call_kwargs["max_tokens"] == 2048
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["role"] == "user"
        assert result.score == 0.8

    @pytest.mark.asyncio
    async def test_uses_custom_system_prompt(self):
        adapter = OpenAIJudgeAdapter(api_key="test-key")

        mock_message = MagicMock()
        mock_message.content = json.dumps({"score": 0.5, "rationale": "OK", "confidence": "medium"})
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        adapter._client = MagicMock()
        adapter._client.chat.completions.create = AsyncMock(return_value=mock_response)

        trace = make_trace()
        dimension = make_dimension()
        judge_config = JudgeConfig(
            primary_model=JudgeModel.GPT_4O,
            custom_system_prompt="My custom prompt",
        )

        await adapter.evaluate_dimension(trace, dimension, judge_config)

        call_kwargs = adapter._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"][0]["content"] == "My custom prompt"
