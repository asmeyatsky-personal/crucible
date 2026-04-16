"""
Trace DTOs — structured schemas for API boundaries.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ToolCallDTO(BaseModel):
    name: str
    input_parameters: dict
    output: str | dict | None = None
    latency_ms: float
    success: bool
    timestamp: datetime
    error: str | None = None


class TrajectoryStepDTO(BaseModel):
    step_index: int
    step_type: str
    content: str
    timestamp: datetime
    tool_calls: list[ToolCallDTO] = Field(default_factory=list)
    token_count: int | None = None
    model_id: str | None = None
    metadata: dict = Field(default_factory=dict)


class CaptureTraceRequest(BaseModel):
    agent_id: str
    run_id: str
    steps: list[TrajectoryStepDTO]
    model_id: str
    temperature: float = 0.0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    metadata: dict = Field(default_factory=dict)


class TraceResponse(BaseModel):
    id: str
    agent_id: str
    run_id: str
    model_id: str
    step_count: int
    total_tokens: int
    total_latency_ms: float
    captured_at: datetime
    metadata: dict = Field(default_factory=dict)
