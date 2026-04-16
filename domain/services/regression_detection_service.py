"""
Regression Detection Domain Service

Architectural Intent:
- Pure domain logic for detecting quality regressions across evaluation runs
- Compares current scores against prior baselines
- Produces RegressionResult value objects with severity classification
- Stateless — all inputs passed as arguments
"""

from __future__ import annotations

from domain.entities.evaluation import Evaluation
from domain.value_objects.score import RegressionResult


class RegressionDetectionService:
    """Detects quality regressions by comparing evaluation scores against baselines."""

    def __init__(
        self,
        minor_threshold: float = 0.05,
        major_threshold: float = 0.10,
        critical_threshold: float = 0.20,
    ):
        self.minor_threshold = minor_threshold
        self.major_threshold = major_threshold
        self.critical_threshold = critical_threshold

    def check_regression(
        self,
        current: Evaluation,
        baseline: Evaluation | None,
    ) -> RegressionResult:
        """
        Compare current evaluation score against a baseline.
        If no baseline exists, returns a 'none' severity result.
        """
        if baseline is None:
            return RegressionResult(
                current_score=current.score_value,
                baseline_score=current.score_value,
                delta=0.0,
                regressed=False,
                severity="none",
            )

        return RegressionResult.compute(
            current=current.score_value,
            baseline=baseline.score_value,
            minor_threshold=self.minor_threshold,
            major_threshold=self.major_threshold,
            critical_threshold=self.critical_threshold,
        )

    def check_dimension_regressions(
        self,
        current: Evaluation,
        baseline: Evaluation,
    ) -> dict[str, RegressionResult]:
        """Check for regressions at the individual dimension level."""
        baseline_map = {ds.dimension_name: ds for ds in baseline.dimension_scores}
        results: dict[str, RegressionResult] = {}

        for ds in current.dimension_scores:
            baseline_ds = baseline_map.get(ds.dimension_name)
            if baseline_ds is None:
                results[ds.dimension_name] = RegressionResult(
                    current_score=ds.score,
                    baseline_score=ds.score,
                    delta=0.0,
                    regressed=False,
                    severity="none",
                )
            else:
                results[ds.dimension_name] = RegressionResult.compute(
                    current=ds.score,
                    baseline=baseline_ds.score,
                    minor_threshold=self.minor_threshold,
                    major_threshold=self.major_threshold,
                    critical_threshold=self.critical_threshold,
                )

        return results
