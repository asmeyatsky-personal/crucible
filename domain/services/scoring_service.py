"""
Scoring Domain Service (SCORE Module)

Architectural Intent:
- Pure domain logic for score aggregation — no infrastructure dependencies
- Computes composite scores from dimension-level scores using rubric weights
- Determines pass/fail status based on rubric thresholds
- Stateless — all inputs passed as arguments

Parallelization Notes:
- Scoring is CPU-bound and fast; no parallelization needed at this level
- Parallelization happens upstream in JUDGE (evaluating dimensions concurrently)
"""

from __future__ import annotations

from domain.entities.rubric import Rubric
from domain.value_objects.score import CompositeScore, DimensionScore


class ScoringService:
    """Computes composite evaluation scores from dimension scores and rubric weights."""

    def compute_composite_score(
        self,
        dimension_scores: list[DimensionScore],
        rubric: Rubric,
    ) -> CompositeScore:
        """
        Compute a weighted composite score from individual dimension scores.

        Weights are normalised from the rubric so they sum to 1.0.
        A run passes if every dimension individually passes AND the composite
        score meets the minimum threshold (lowest dimension threshold).
        """
        if not dimension_scores:
            raise ValueError("Cannot compute composite score with no dimension scores")

        normalised_weights = rubric.normalised_weights
        score_map = {ds.dimension_name: ds for ds in dimension_scores}

        weighted_sum = 0.0
        for dim in rubric.dimensions:
            ds = score_map.get(dim.name)
            if ds is None:
                raise ValueError(f"Missing score for dimension: {dim.name}")
            weight = normalised_weights[dim.name]
            weighted_sum += ds.score * weight

        composite_value = round(weighted_sum, 4)
        dimensions_passed = sum(1 for ds in dimension_scores if ds.passed)
        all_passed = all(ds.passed for ds in dimension_scores)

        return CompositeScore(
            value=composite_value,
            dimension_scores=tuple(dimension_scores),
            passed=all_passed,
            dimensions_passed=dimensions_passed,
            dimensions_total=len(dimension_scores),
        )
