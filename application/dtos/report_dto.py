"""
Report DTOs — structured schemas for API boundaries.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class GenerateReportRequest(BaseModel):
    agent_id: str
    title: str
    evaluation_ids: list[str] = Field(default_factory=list)
    export_format: str = "json"


class ReportResponse(BaseModel):
    id: str
    title: str
    agent_id: str
    evaluation_ids: list[str]
    export_format: str
    generated_at: datetime
    content: str
