"""
Generate Report Use Case (REPORT Module)

Architectural Intent:
- Generates evidence reports from evaluation data
- Delegates serialisation to ReportExportPort (infrastructure)
- Supports JSON, PDF, CSV, webhook formats
"""

from __future__ import annotations

from uuid import uuid4

from application.dtos.report_dto import GenerateReportRequest
from domain.entities.report import ExportFormat, Report
from domain.ports.evaluation_repository_port import EvaluationRepositoryPort
from domain.ports.event_bus_port import EventBusPort
from domain.ports.report_export_port import ReportExportPort


class GenerateReportUseCase:
    def __init__(
        self,
        evaluation_repository: EvaluationRepositoryPort,
        report_export: ReportExportPort,
        event_bus: EventBusPort,
    ):
        self._eval_repo = evaluation_repository
        self._export = report_export
        self._event_bus = event_bus

    async def execute(self, request: GenerateReportRequest) -> Report:
        export_format = ExportFormat(request.export_format)

        # Fetch evaluations — either specific IDs or all for agent
        if request.evaluation_ids:
            evaluations = []
            for eval_id in request.evaluation_ids:
                ev = await self._eval_repo.get_by_id(eval_id)
                if ev is None:
                    raise ValueError(f"Evaluation not found: {eval_id}")
                evaluations.append(ev)
        else:
            evaluations = await self._eval_repo.list_by_agent(request.agent_id)

        if not evaluations:
            raise ValueError(f"No evaluations found for agent: {request.agent_id}")

        content = await self._export.export(evaluations, export_format, request.title)

        report = Report.generate(
            id=str(uuid4()),
            title=request.title,
            agent_id=request.agent_id,
            evaluation_ids=[e.id for e in evaluations],
            export_format=export_format,
            content=content,
        )

        await self._event_bus.publish(list(report.domain_events))
        return report.clear_events()
