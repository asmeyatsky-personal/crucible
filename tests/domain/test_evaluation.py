"""
Evaluation Entity Tests — pure domain logic, no mocks needed.
"""

from domain.entities.evaluation import Evaluation, EvaluationCompletedEvent
from domain.value_objects.judge_config import ConsensusMode, JudgeConfig, JudgeModel
from domain.value_objects.score import CompositeScore, Confidence, DimensionScore


def _dim_score(name: str, score: float, passed: bool) -> DimensionScore:
    return DimensionScore(
        dimension_name=name,
        score=score,
        passed=passed,
        rationale=f"Assessment of {name}",
        confidence=Confidence.HIGH,
    )


def _composite(value: float, passed: bool, dims: list[DimensionScore]) -> CompositeScore:
    return CompositeScore(
        value=value,
        dimension_scores=tuple(dims),
        passed=passed,
        dimensions_passed=sum(1 for d in dims if d.passed),
        dimensions_total=len(dims),
    )


class TestEvaluation:
    def test_create_evaluation(self):
        dims = [_dim_score("goal", 0.85, True), _dim_score("output", 0.72, True)]
        composite = _composite(0.79, True, dims)
        judge_config = JudgeConfig(primary_model=JudgeModel.CLAUDE_SONNET)

        evaluation = Evaluation.create(
            id="e-1", trace_id="t-1", rubric_id="r-1", rubric_version=1,
            agent_id="a-1", run_id="run-1", judge_config=judge_config,
            composite_score=composite, dimension_scores=dims,
            judge_model_id="claude-sonnet-4-6",
        )

        assert evaluation.id == "e-1"
        assert evaluation.passed is True
        assert evaluation.score_value == 0.79
        assert len(evaluation.dimension_scores) == 2

    def test_create_produces_domain_event(self):
        dims = [_dim_score("goal", 0.85, True)]
        composite = _composite(0.85, True, dims)
        judge_config = JudgeConfig(primary_model=JudgeModel.CLAUDE_SONNET)

        evaluation = Evaluation.create(
            id="e-1", trace_id="t-1", rubric_id="r-1", rubric_version=1,
            agent_id="a-1", run_id="run-1", judge_config=judge_config,
            composite_score=composite, dimension_scores=dims,
            judge_model_id="claude-sonnet-4-6",
        )

        assert len(evaluation.domain_events) == 1
        event = evaluation.domain_events[0]
        assert isinstance(event, EvaluationCompletedEvent)
        assert event.composite_score == 0.85
        assert event.passed is True

    def test_failed_evaluation(self):
        dims = [_dim_score("goal", 0.3, False), _dim_score("output", 0.4, False)]
        composite = _composite(0.35, False, dims)
        judge_config = JudgeConfig(primary_model=JudgeModel.CLAUDE_SONNET)

        evaluation = Evaluation.create(
            id="e-1", trace_id="t-1", rubric_id="r-1", rubric_version=1,
            agent_id="a-1", run_id="run-1", judge_config=judge_config,
            composite_score=composite, dimension_scores=dims,
            judge_model_id="claude-sonnet-4-6",
        )

        assert evaluation.passed is False

    def test_evaluation_is_immutable(self):
        dims = [_dim_score("goal", 0.85, True)]
        composite = _composite(0.85, True, dims)
        judge_config = JudgeConfig(primary_model=JudgeModel.CLAUDE_SONNET)

        evaluation = Evaluation.create(
            id="e-1", trace_id="t-1", rubric_id="r-1", rubric_version=1,
            agent_id="a-1", run_id="run-1", judge_config=judge_config,
            composite_score=composite, dimension_scores=dims,
            judge_model_id="claude-sonnet-4-6",
        )

        try:
            evaluation.id = "changed"
            assert False, "Should not be able to modify frozen dataclass"
        except AttributeError:
            pass
