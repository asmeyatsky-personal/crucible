"""Test fixtures for domain tests."""

from domain.entities.evaluation import Evaluation
from domain.value_objects.judge_config import JudgeConfig, JudgeModel
from domain.value_objects.score import CompositeScore, Confidence, DimensionScore


def make_evaluation(
    id: str = "e-1",
    score: float = 0.85,
    passed: bool = True,
    agent_id: str = "a-1",
) -> Evaluation:
    dim_scores = [
        DimensionScore(
            dimension_name="test_dim",
            score=score,
            passed=passed,
            rationale="test",
            confidence=Confidence.HIGH,
        )
    ]
    composite = CompositeScore(
        value=score,
        dimension_scores=tuple(dim_scores),
        passed=passed,
        dimensions_passed=1 if passed else 0,
        dimensions_total=1,
    )
    return Evaluation.create(
        id=id,
        trace_id="t-1",
        rubric_id="r-1",
        rubric_version=1,
        agent_id=agent_id,
        run_id="run-1",
        judge_config=JudgeConfig(primary_model=JudgeModel.CLAUDE_SONNET),
        composite_score=composite,
        dimension_scores=dim_scores,
        judge_model_id="claude-sonnet-4-6",
    ).clear_events()


def make_evaluation_with_dimensions(
    id: str = "e-1",
    agent_id: str = "a-1",
    dimension_scores: list[DimensionScore] | None = None,
    composite_value: float | None = None,
    passed: bool = True,
) -> Evaluation:
    """Create an evaluation with explicit dimension scores for regression tests."""
    if dimension_scores is None:
        dimension_scores = [
            DimensionScore(
                dimension_name="test_dim",
                score=0.85,
                passed=True,
                rationale="test",
                confidence=Confidence.HIGH,
            )
        ]
    if composite_value is None:
        composite_value = sum(ds.score for ds in dimension_scores) / len(dimension_scores)
    composite = CompositeScore(
        value=composite_value,
        dimension_scores=tuple(dimension_scores),
        passed=passed,
        dimensions_passed=sum(1 for d in dimension_scores if d.passed),
        dimensions_total=len(dimension_scores),
    )
    return Evaluation.create(
        id=id,
        trace_id="t-1",
        rubric_id="r-1",
        rubric_version=1,
        agent_id=agent_id,
        run_id="run-1",
        judge_config=JudgeConfig(primary_model=JudgeModel.CLAUDE_SONNET),
        composite_score=composite,
        dimension_scores=dimension_scores,
        judge_model_id="claude-sonnet-4-6",
    ).clear_events()
