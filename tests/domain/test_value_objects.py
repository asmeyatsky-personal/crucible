"""
Value Object Tests — pure domain logic, no mocks needed.

Covers validation edge cases for:
- JudgeConfig (MULTI consensus mode without secondary_models)
- RubricDimension (negative weight, bad threshold, missing judge_prompt)
- DimensionScore / CompositeScore (out-of-range values)
- ToolCall (empty name, negative latency)
- TrajectoryStep (negative step_index)
"""

import pytest
from datetime import UTC, datetime

from domain.value_objects.judge_config import ConsensusMode, JudgeConfig, JudgeModel
from domain.value_objects.rubric_dimension import RubricDimension, ScoringMethod
from domain.value_objects.score import CompositeScore, Confidence, DimensionScore
from domain.value_objects.tool_call import ToolCall
from domain.value_objects.trajectory_step import StepType, TrajectoryStep


class TestJudgeConfig:
    def test_single_mode_no_secondary_models(self):
        config = JudgeConfig(
            primary_model=JudgeModel.CLAUDE_SONNET,
            consensus_mode=ConsensusMode.SINGLE,
        )
        assert config.consensus_mode == ConsensusMode.SINGLE
        assert config.secondary_models == ()

    def test_multi_mode_without_secondary_models_raises(self):
        with pytest.raises(ValueError, match="secondary_models required"):
            JudgeConfig(
                primary_model=JudgeModel.CLAUDE_SONNET,
                consensus_mode=ConsensusMode.MULTI,
            )

    def test_multi_mode_with_secondary_models_ok(self):
        config = JudgeConfig(
            primary_model=JudgeModel.CLAUDE_SONNET,
            consensus_mode=ConsensusMode.MULTI,
            secondary_models=(JudgeModel.GPT_4O,),
        )
        assert config.consensus_mode == ConsensusMode.MULTI
        assert len(config.secondary_models) == 1


class TestRubricDimension:
    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="Weight must be non-negative"):
            RubricDimension(
                name="bad", description="bad weight",
                scoring_method=ScoringMethod.EXACT_MATCH,
                weight=-0.5, pass_threshold=0.5,
            )

    def test_pass_threshold_below_zero_raises(self):
        with pytest.raises(ValueError, match="Pass threshold must be between"):
            RubricDimension(
                name="bad", description="bad threshold",
                scoring_method=ScoringMethod.EXACT_MATCH,
                weight=1.0, pass_threshold=-0.1,
            )

    def test_pass_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="Pass threshold must be between"):
            RubricDimension(
                name="bad", description="bad threshold",
                scoring_method=ScoringMethod.EXACT_MATCH,
                weight=1.0, pass_threshold=1.5,
            )

    def test_llm_judge_without_prompt_raises(self):
        with pytest.raises(ValueError, match="judge_prompt is required"):
            RubricDimension(
                name="bad", description="missing prompt",
                scoring_method=ScoringMethod.LLM_JUDGE,
                weight=1.0, pass_threshold=0.5,
                judge_prompt=None,
            )

    def test_llm_judge_with_empty_prompt_raises(self):
        with pytest.raises(ValueError, match="judge_prompt is required"):
            RubricDimension(
                name="bad", description="empty prompt",
                scoring_method=ScoringMethod.LLM_JUDGE,
                weight=1.0, pass_threshold=0.5,
                judge_prompt="",
            )

    def test_valid_dimension(self):
        dim = RubricDimension(
            name="good", description="all good",
            scoring_method=ScoringMethod.EXACT_MATCH,
            weight=0.5, pass_threshold=0.7,
        )
        assert dim.name == "good"
        assert dim.weight == 0.5


class TestDimensionScore:
    def test_score_below_zero_raises(self):
        with pytest.raises(ValueError, match="Score must be between 0.0 and 1.0"):
            DimensionScore(
                dimension_name="bad", score=-0.1, passed=False,
                rationale="test", confidence=Confidence.HIGH,
            )

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError, match="Score must be between 0.0 and 1.0"):
            DimensionScore(
                dimension_name="bad", score=1.5, passed=True,
                rationale="test", confidence=Confidence.HIGH,
            )

    def test_valid_score(self):
        ds = DimensionScore(
            dimension_name="ok", score=0.75, passed=True,
            rationale="good", confidence=Confidence.MEDIUM,
        )
        assert ds.score == 0.75


class TestCompositeScore:
    def test_value_below_zero_raises(self):
        with pytest.raises(ValueError, match="Composite score must be between 0.0 and 1.0"):
            CompositeScore(
                value=-0.5,
                dimension_scores=(),
                passed=False,
                dimensions_passed=0,
                dimensions_total=0,
            )

    def test_value_above_one_raises(self):
        with pytest.raises(ValueError, match="Composite score must be between 0.0 and 1.0"):
            CompositeScore(
                value=1.1,
                dimension_scores=(),
                passed=True,
                dimensions_passed=0,
                dimensions_total=0,
            )

    def test_valid_composite(self):
        cs = CompositeScore(
            value=0.80,
            dimension_scores=(),
            passed=True,
            dimensions_passed=0,
            dimensions_total=0,
        )
        assert cs.value == 0.80


class TestToolCall:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="Tool call name must not be empty"):
            ToolCall(
                name="", input_parameters={}, output=None,
                latency_ms=10.0, success=True, timestamp=datetime.now(UTC),
            )

    def test_negative_latency_raises(self):
        with pytest.raises(ValueError, match="latency_ms must be non-negative"):
            ToolCall(
                name="search", input_parameters={}, output=None,
                latency_ms=-5.0, success=True, timestamp=datetime.now(UTC),
            )

    def test_valid_tool_call(self):
        tc = ToolCall(
            name="search", input_parameters={"q": "test"},
            output="found", latency_ms=100.0, success=True,
            timestamp=datetime.now(UTC),
        )
        assert tc.name == "search"
        assert tc.latency_ms == 100.0


class TestTrajectoryStep:
    def test_negative_step_index_raises(self):
        with pytest.raises(ValueError, match="step_index must be non-negative"):
            TrajectoryStep(
                step_index=-1,
                step_type=StepType.REASONING,
                content="test",
                timestamp=datetime.now(UTC),
            )

    def test_valid_step(self):
        step = TrajectoryStep(
            step_index=0,
            step_type=StepType.REASONING,
            content="thinking",
            timestamp=datetime.now(UTC),
        )
        assert step.step_index == 0
        assert step.step_type == StepType.REASONING
