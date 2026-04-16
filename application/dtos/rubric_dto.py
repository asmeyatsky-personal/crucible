"""
Rubric DTOs — structured schemas for API boundaries.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class RubricDimensionDTO(BaseModel):
    name: str
    description: str
    scoring_method: str = "llm_judge"
    weight: float = 1.0
    pass_threshold: float = 0.7
    judge_prompt: str | None = None
    expected_value: str | None = None


class CreateRubricRequest(BaseModel):
    name: str
    description: str
    dimensions: list[RubricDimensionDTO]
    agent_type: str | None = None
    owner: str = ""


class RubricResponse(BaseModel):
    id: str
    name: str
    description: str
    dimensions: list[RubricDimensionDTO]
    version: int
    agent_type: str | None = None
    owner: str
    created_at: datetime
    updated_at: datetime
