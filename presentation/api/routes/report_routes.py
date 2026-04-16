"""Report API Routes — presentation layer for evidence report generation."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from application.dtos.report_dto import GenerateReportRequest
from infrastructure.config.dependency_injection import Container


def create_report_router(container: Container) -> APIRouter:
    router = APIRouter()

    @router.post("")
    async def generate_report(request: GenerateReportRequest):
        try:
            use_case = container.generate_report()
            report = await use_case.execute(request)
            return {
                "id": report.id,
                "title": report.title,
                "agent_id": report.agent_id,
                "export_format": report.export_format.value,
                "evaluation_count": len(report.evaluation_ids),
                "generated_at": report.generated_at.isoformat(),
                "content": report.content,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return router
