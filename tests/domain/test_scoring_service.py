"""
Scoring Service Tests — pure domain logic, no mocks needed.
"""

import pytest

from domain.entities.rubric import Rubric
from domain.services.scoring_service import ScoringService
from domain.value_objects.rubric_dimension import RubricDimension, ScoringMethod
from domain.value_objects.score import Confidence, DimensionScore


def _dim(name: str, weight: float) -> RubricDimension:
    return RubricDimension(
        name=name, description=f"dim {name}",
        scoring_method=ScoringMethod.LLM_JUDGE,
        weight=weight, pass_threshold=0.7,
        judge_prompt=f"Evaluate {name}",
    )


def _score(name: str, value: float, passed: bool) -> DimensionScore:
    return DimensionScore(
        dimension_name=name, score=value, passed=passed,
        rationale="test", confidence=Confidence.HIGH,
    )


class TestScoringService:
    def setup_method(self):
        self.service = ScoringService()

    def test_weighted_composite_score(self):
        rubric = Rubric.create(
            id="r-1", name="test", description="test",
            dimensions=[_dim("goal", 2.0), _dim("output", 1.0)],
        )
        scores = [_score("goal", 0.9, True), _score("output", 0.6, False)]

        result = self.service.compute_composite_score(scores, rubric)

        # goal weight = 2/3, output weight = 1/3
        # composite = 0.9 * (2/3) + 0.6 * (1/3) = 0.6 + 0.2 = 0.8
        assert abs(result.value - 0.8) < 0.001
        assert result.passed is False  # output dimension failed
        assert result.dimensions_passed == 1
        assert result.dimensions_total == 2

    def test_all_dimensions_pass(self):
        rubric = Rubric.create(
            id="r-1", name="test", description="test",
            dimensions=[_dim("goal", 1.0), _dim("output", 1.0)],
        )
        scores = [_score("goal", 0.85, True), _score("output", 0.80, True)]

        result = self.service.compute_composite_score(scores, rubric)

        assert result.passed is True
        assert result.dimensions_passed == 2

    def test_equal_weights(self):
        rubric = Rubric.create(
            id="r-1", name="test", description="test",
            dimensions=[_dim("d1", 1.0), _dim("d2", 1.0), _dim("d3", 1.0)],
        )
        scores = [
            _score("d1", 0.9, True),
            _score("d2", 0.6, False),
            _score("d3", 0.3, False),
        ]

        result = self.service.compute_composite_score(scores, rubric)
        assert abs(result.value - 0.6) < 0.001

    def test_missing_dimension_score_raises(self):
        rubric = Rubric.create(
            id="r-1", name="test", description="test",
            dimensions=[_dim("goal", 1.0), _dim("output", 1.0)],
        )
        scores = [_score("goal", 0.9, True)]  # Missing "output"

        with pytest.raises(ValueError, match="Missing score for dimension"):
            self.service.compute_composite_score(scores, rubric)

    def test_empty_scores_raises(self):
        rubric = Rubric.create(
            id="r-1", name="test", description="test",
            dimensions=[_dim("goal", 1.0)],
        )
        with pytest.raises(ValueError, match="no dimension scores"):
            self.service.compute_composite_score([], rubric)
