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
