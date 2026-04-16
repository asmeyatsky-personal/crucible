"""
Report Export Port

Architectural Intent:
- Contract for exporting evaluation reports in various formats
- Implementations handle JSON, PDF, CSV serialisation
"""

from __future__ import annotations

from typing import Protocol

from domain.entities.evaluation import Evaluation
from domain.entities.report import ExportFormat


class ReportExportPort(Protocol):
    async def export(
        self,
        evaluations: list[Evaluation],
        export_format: ExportFormat,
        title: str,
    ) -> str: ...
