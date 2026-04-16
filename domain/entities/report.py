"""
Report Entity (REPORT Module)

Architectural Intent:
- Aggregate root representing a generated evidence report
- Reports are immutable artefacts produced from evaluation data
- Support multiple export formats: JSON, PDF, CSV, Webhook

MCP Integration:
- Exposed as resource report://{report_id}
- Created via 'generate_report' tool
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum

from domain.events.event_base import DomainEvent


class ExportFormat(Enum):
    JSON = "json"
    PDF = "pdf"
    CSV = "csv"
    WEBHOOK = "webhook"


@dataclass(frozen=True)
class ReportGeneratedEvent(DomainEvent):
    export_format: str = ""
    evaluation_count: int = 0


@dataclass(frozen=True)
class Report:
    """Generated evidence report from CRUCIBLE evaluations."""

    id: str
    title: str
    agent_id: str
    evaluation_ids: tuple[str, ...]
    export_format: ExportFormat
    content: str  # Serialised report content
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict = field(default_factory=dict)
    domain_events: tuple[DomainEvent, ...] = field(default=())

    @staticmethod
    def generate(
        id: str, title: str, agent_id: str,
        evaluation_ids: list[str], export_format: ExportFormat,
        content: str, metadata: dict | None = None,
    ) -> Report:
        return Report(
            id=id,
            title=title,
            agent_id=agent_id,
            evaluation_ids=tuple(evaluation_ids),
            export_format=export_format,
            content=content,
            metadata=metadata or {},
            domain_events=(
                ReportGeneratedEvent(
                    aggregate_id=id,
                    export_format=export_format.value,
                    evaluation_count=len(evaluation_ids),
                ),
            ),
        )

    def clear_events(self) -> Report:
        return replace(self, domain_events=())
