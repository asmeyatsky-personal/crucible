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
