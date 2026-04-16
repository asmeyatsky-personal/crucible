"""
Report Export Adapter

Architectural Intent:
- Implements ReportExportPort with JSON and CSV export formats
- Serialises evaluation data into structured evidence reports
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime

from domain.entities.evaluation import Evaluation
from domain.entities.report import ExportFormat


class ReportExportAdapter:
    """Implements ReportExportPort with JSON and CSV export."""

    async def export(
        self,
        evaluations: list[Evaluation],
        export_format: ExportFormat,
        title: str,
    ) -> str:
        if export_format == ExportFormat.JSON:
            return self._export_json(evaluations, title)
        elif export_format == ExportFormat.CSV:
            return self._export_csv(evaluations, title)
        elif export_format == ExportFormat.PDF:
            # PDF generation would require a dedicated library (reportlab, weasyprint)
            # Return JSON as fallback with a note
            return self._export_json(evaluations, title)
        elif export_format == ExportFormat.WEBHOOK:
            return self._export_json(evaluations, title)
        else:
            return self._export_json(evaluations, title)

    def _export_json(self, evaluations: list[Evaluation], title: str) -> str:
        report_data = {
            "title": title,
            "generated_at": datetime.utcnow().isoformat(),
            "evaluation_count": len(evaluations),
            "evaluations": [
                {
                    "id": e.id,
                    "agent_id": e.agent_id,
                    "run_id": e.run_id,
                    "trace_id": e.trace_id,
                    "rubric_id": e.rubric_id,
                    "rubric_version": e.rubric_version,
                    "composite_score": {
                        "value": e.composite_score.value,
                        "passed": e.composite_score.passed,
                        "dimensions_passed": e.composite_score.dimensions_passed,
                        "dimensions_total": e.composite_score.dimensions_total,
                    },
                    "dimension_scores": [
                        {
                            "dimension_name": ds.dimension_name,
                            "score": ds.score,
                            "passed": ds.passed,
                            "rationale": ds.rationale,
                            "confidence": ds.confidence.value,
                        }
                        for ds in e.dimension_scores
                    ],
                    "judge_model_id": e.judge_model_id,
                    "evaluated_at": e.evaluated_at.isoformat(),
                }
                for e in evaluations
            ],
        }
        return json.dumps(report_data, indent=2)

    def _export_csv(self, evaluations: list[Evaluation], title: str) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "evaluation_id", "agent_id", "run_id", "trace_id",
            "rubric_id", "composite_score", "passed",
            "dimensions_passed", "dimensions_total",
            "judge_model", "evaluated_at",
        ])
        for e in evaluations:
            writer.writerow([
                e.id, e.agent_id, e.run_id, e.trace_id,
                e.rubric_id, e.composite_score.value, e.composite_score.passed,
                e.composite_score.dimensions_passed, e.composite_score.dimensions_total,
                e.judge_model_id, e.evaluated_at.isoformat(),
            ])
        return output.getvalue()
