"""
Regression Detection Service Tests — pure domain logic, no mocks needed.
"""

from domain.value_objects.score import RegressionResult
from domain.services.regression_detection_service import RegressionDetectionService


class TestRegressionResult:
    def test_no_regression(self):
        result = RegressionResult.compute(current=0.85, baseline=0.80)
        assert result.regressed is False
        assert result.severity == "none"
        assert result.delta == 0.05

    def test_minor_regression(self):
        result = RegressionResult.compute(current=0.74, baseline=0.80)
        assert result.regressed is True
        assert result.severity == "minor"

    def test_major_regression(self):
        result = RegressionResult.compute(current=0.68, baseline=0.80)
        assert result.regressed is True
        assert result.severity == "major"

    def test_critical_regression(self):
        result = RegressionResult.compute(current=0.55, baseline=0.80)
        assert result.regressed is True
        assert result.severity == "critical"

    def test_improvement_is_not_regression(self):
        result = RegressionResult.compute(current=0.90, baseline=0.80)
        assert result.regressed is False
        assert result.severity == "none"
        assert result.delta == 0.10

    def test_equal_scores(self):
        result = RegressionResult.compute(current=0.80, baseline=0.80)
        assert result.regressed is False
        assert result.severity == "none"
        assert result.delta == 0.0

    def test_custom_thresholds(self):
        result = RegressionResult.compute(
            current=0.77, baseline=0.80,
            minor_threshold=0.02, major_threshold=0.05, critical_threshold=0.10,
        )
        assert result.regressed is True
        assert result.severity == "minor"


class TestRegressionDetectionService:
    def setup_method(self):
        self.service = RegressionDetectionService()

    def test_no_baseline_returns_no_regression(self):
        from tests.domain.conftest import make_evaluation
        current = make_evaluation(score=0.85)
        result = self.service.check_regression(current, None)
        assert result.regressed is False
        assert result.severity == "none"

    def test_check_regression_with_baseline(self):
        from tests.domain.conftest import make_evaluation
        current = make_evaluation(id="e-cur", score=0.55)
        baseline = make_evaluation(id="e-base", score=0.80)
        result = self.service.check_regression(current, baseline)
        assert result.regressed is True
        assert result.severity == "critical"

    def test_check_dimension_regressions_no_baseline_dimension(self):
        """When a dimension exists only in the current eval, it gets severity 'none'."""
        from tests.domain.conftest import make_evaluation_with_dimensions
        from domain.value_objects.score import Confidence, DimensionScore

        current_dims = [
            DimensionScore(
                dimension_name="new_dim", score=0.80, passed=True,
                rationale="test", confidence=Confidence.HIGH,
            ),
        ]
        baseline_dims = [
            DimensionScore(
                dimension_name="old_dim", score=0.90, passed=True,
                rationale="test", confidence=Confidence.HIGH,
            ),
        ]
        current = make_evaluation_with_dimensions(
            id="e-cur", dimension_scores=current_dims,
        )
        baseline = make_evaluation_with_dimensions(
            id="e-base", dimension_scores=baseline_dims,
        )
        results = self.service.check_dimension_regressions(current, baseline)
        assert "new_dim" in results
        r = results["new_dim"]
        assert r.regressed is False
        assert r.severity == "none"
        assert r.current_score == 0.80
        assert r.baseline_score == 0.80
        assert r.delta == 0.0

    def test_check_dimension_regressions_with_regression(self):
        """When a dimension drops significantly, it detects the regression."""
        from tests.domain.conftest import make_evaluation_with_dimensions
        from domain.value_objects.score import Confidence, DimensionScore

        dim_name = "accuracy"
        current_dims = [
            DimensionScore(
                dimension_name=dim_name, score=0.50, passed=False,
                rationale="dropped", confidence=Confidence.HIGH,
            ),
        ]
        baseline_dims = [
            DimensionScore(
                dimension_name=dim_name, score=0.90, passed=True,
                rationale="was fine", confidence=Confidence.HIGH,
            ),
        ]
        current = make_evaluation_with_dimensions(
            id="e-cur", dimension_scores=current_dims,
        )
        baseline = make_evaluation_with_dimensions(
            id="e-base", dimension_scores=baseline_dims,
        )
        results = self.service.check_dimension_regressions(current, baseline)
        assert dim_name in results
        r = results[dim_name]
        assert r.regressed is True
        assert r.severity == "critical"

    def test_check_dimension_regressions_no_regression(self):
        """When dimension scores are stable, no regression is detected."""
        from tests.domain.conftest import make_evaluation_with_dimensions
        from domain.value_objects.score import Confidence, DimensionScore

        dim_name = "quality"
        current_dims = [
            DimensionScore(
                dimension_name=dim_name, score=0.85, passed=True,
                rationale="good", confidence=Confidence.HIGH,
            ),
        ]
        baseline_dims = [
            DimensionScore(
                dimension_name=dim_name, score=0.84, passed=True,
                rationale="good", confidence=Confidence.HIGH,
            ),
        ]
        current = make_evaluation_with_dimensions(
            id="e-cur", dimension_scores=current_dims,
        )
        baseline = make_evaluation_with_dimensions(
            id="e-base", dimension_scores=baseline_dims,
        )
        results = self.service.check_dimension_regressions(current, baseline)
        assert dim_name in results
        r = results[dim_name]
        assert r.regressed is False
        assert r.severity == "none"

    def test_check_dimension_regressions_multiple_dimensions(self):
        """Test with multiple dimensions: one regressed, one new, one stable."""
        from tests.domain.conftest import make_evaluation_with_dimensions
        from domain.value_objects.score import Confidence, DimensionScore

        current_dims = [
            DimensionScore(
                dimension_name="stable", score=0.85, passed=True,
                rationale="ok", confidence=Confidence.HIGH,
            ),
            DimensionScore(
                dimension_name="regressed", score=0.50, passed=False,
                rationale="dropped", confidence=Confidence.HIGH,
            ),
            DimensionScore(
                dimension_name="new_only", score=0.70, passed=True,
                rationale="new", confidence=Confidence.HIGH,
            ),
        ]
        baseline_dims = [
            DimensionScore(
                dimension_name="stable", score=0.84, passed=True,
                rationale="ok", confidence=Confidence.HIGH,
            ),
            DimensionScore(
                dimension_name="regressed", score=0.90, passed=True,
                rationale="was fine", confidence=Confidence.HIGH,
            ),
        ]
        current = make_evaluation_with_dimensions(
            id="e-cur", dimension_scores=current_dims, composite_value=0.68,
        )
        baseline = make_evaluation_with_dimensions(
            id="e-base", dimension_scores=baseline_dims, composite_value=0.87,
        )
        results = self.service.check_dimension_regressions(current, baseline)

        assert len(results) == 3
        assert results["stable"].regressed is False
        assert results["regressed"].regressed is True
        assert results["regressed"].severity == "critical"
        assert results["new_only"].regressed is False
        assert results["new_only"].severity == "none"
