"""
Evaluation DTOs — structured schemas for API boundaries.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class JudgeConfigDTO(BaseModel):
    primary_model: str = "claude-sonnet-4-6"
    consensus_mode: str = "single"
    secondary_models: list[str] = Field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int = 4096
    custom_system_prompt: str | None = None


class EvaluateRunRequest(BaseModel):
    trace_id: str
    rubric_id: str
    judge_config: JudgeConfigDTO = Field(default_factory=JudgeConfigDTO)


class DimensionScoreDTO(BaseModel):
    dimension_name: str
    score: float
    passed: bool
    rationale: str
    confidence: str


class CompositeScoreDTO(BaseModel):
    value: float
    passed: bool
    dimensions_passed: int
    dimensions_total: int


class EvaluationResponse(BaseModel):
    id: str
    trace_id: str
    rubric_id: str
    rubric_version: int
    agent_id: str
    run_id: str
    composite_score: CompositeScoreDTO
    dimension_scores: list[DimensionScoreDTO]
    judge_model_id: str
    evaluated_at: datetime
    metadata: dict = Field(default_factory=dict)


class RegressionResultDTO(BaseModel):
    current_score: float
    baseline_score: float
    delta: float
    regressed: bool
    severity: str
