"""
Score Value Objects

Architectural Intent:
- Immutable value objects representing evaluation scores
- Encapsulate scoring semantics: numeric range, pass/fail, confidence
- No identity — equality is by value
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class DimensionScore:
    """Score for a single rubric dimension."""

    dimension_name: str
    score: float  # 0.0 – 1.0
    passed: bool
    rationale: str
    confidence: Confidence

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {self.score}")


@dataclass(frozen=True)
class CompositeScore:
    """Weighted aggregate score across all rubric dimensions."""

    value: float  # 0.0 – 1.0
    dimension_scores: tuple[DimensionScore, ...]
    passed: bool
    dimensions_passed: int
    dimensions_total: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Composite score must be between 0.0 and 1.0, got {self.value}")


@dataclass(frozen=True)
class RegressionResult:
    """Result of comparing a score against a prior baseline."""

    current_score: float
    baseline_score: float
    delta: float
    regressed: bool
    severity: Literal["none", "minor", "major", "critical"]

    @staticmethod
    def compute(
        current: float, baseline: float, minor_threshold: float = 0.05,
        major_threshold: float = 0.10, critical_threshold: float = 0.20,
    ) -> RegressionResult:
        delta = current - baseline
        regressed = delta < -minor_threshold
        if delta < -critical_threshold:
            severity: Literal["none", "minor", "major", "critical"] = "critical"
        elif delta < -major_threshold:
            severity = "major"
        elif delta < -minor_threshold:
            severity = "minor"
        else:
            severity = "none"
        return RegressionResult(
            current_score=current,
            baseline_score=baseline,
            delta=round(delta, 4),
            regressed=regressed,
            severity=severity,
        )
