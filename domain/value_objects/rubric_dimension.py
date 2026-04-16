"""
Rubric Dimension Value Object

Architectural Intent:
- Immutable specification of a single evaluation dimension within a rubric
- Defines scoring method, weight, pass threshold, and description
- Part of the RUBRIC module evaluation criteria engine
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ScoringMethod(Enum):
    LLM_JUDGE = "llm_judge"
    EXACT_MATCH = "exact_match"
    REGEX_MATCH = "regex_match"
    CONTAINS = "contains"
    CUSTOM = "custom"


@dataclass(frozen=True)
class RubricDimension:
    """A single evaluation dimension within a rubric."""

    name: str
    description: str
    scoring_method: ScoringMethod
    weight: float  # 0.0 – 1.0, weights are normalised at rubric level
    pass_threshold: float  # 0.0 – 1.0
    judge_prompt: str | None = None  # Required when scoring_method is LLM_JUDGE
    expected_value: str | None = None  # Used for EXACT_MATCH, REGEX_MATCH, CONTAINS

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError(f"Weight must be non-negative, got {self.weight}")
        if not 0.0 <= self.pass_threshold <= 1.0:
            raise ValueError(
                f"Pass threshold must be between 0.0 and 1.0, got {self.pass_threshold}"
            )
        if self.scoring_method == ScoringMethod.LLM_JUDGE and not self.judge_prompt:
            raise ValueError("judge_prompt is required when scoring_method is LLM_JUDGE")
