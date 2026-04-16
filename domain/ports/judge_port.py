"""
Judge Port

Architectural Intent:
- Contract for LLM-as-judge evaluation — the core of the JUDGE module
- Implementations wrap specific AI providers (Claude, OpenAI, custom)
- Returns structured DimensionScore results
"""

from __future__ import annotations

from typing import Protocol

from domain.entities.trace import Trace
from domain.value_objects.judge_config import JudgeConfig
from domain.value_objects.rubric_dimension import RubricDimension
from domain.value_objects.score import DimensionScore


class JudgePort(Protocol):
    async def evaluate_dimension(
        self,
        trace: Trace,
        dimension: RubricDimension,
        judge_config: JudgeConfig,
    ) -> DimensionScore: ...
