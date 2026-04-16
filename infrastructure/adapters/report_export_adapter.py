"""
Report Export Adapter

Architectural Intent:
- Implements ReportExportPort with JSON, CSV, and PDF export formats
- Serialises evaluation data into structured evidence reports
- PDF uses a lightweight text-based format (no heavy dependencies)
"""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime

from domain.entities.evaluation import Evaluation
from domain.entities.report import ExportFormat


class ReportExportAdapter:
    """Implements ReportExportPort with JSON, CSV, and PDF export."""

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
            return self._export_pdf(evaluations, title)
        elif export_format == ExportFormat.WEBHOOK:
            return self._export_json(evaluations, title)
        else:
            return self._export_json(evaluations, title)

    def _export_json(self, evaluations: list[Evaluation], title: str) -> str:
        report_data = {
            "title": title,
            "generated_at": datetime.now(UTC).isoformat(),
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

    def _export_pdf(self, evaluations: list[Evaluation], title: str) -> str:
        """
        Generate a structured text-based PDF evaluation certificate.

        Uses plain text layout suitable for conversion to PDF via any renderer.
        The content is a formatted evidence report matching the PRD's
        'human-readable evaluation certificate for audit submission' requirement.
        """
        lines: list[str] = []
        sep = "=" * 72
        lines.append(sep)
        lines.append(f"  CRUCIBLE™ EVALUATION CERTIFICATE")
        lines.append(sep)
        lines.append(f"  Title:     {title}")
        lines.append(f"  Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"  Evaluations: {len(evaluations)}")
        lines.append(sep)
        lines.append("")

        for i, e in enumerate(evaluations, 1):
            status = "PASS" if e.composite_score.passed else "FAIL"
            lines.append(f"  [{i}] Evaluation: {e.id}")
            lines.append(f"      Agent:    {e.agent_id}")
            lines.append(f"      Run:      {e.run_id}")
            lines.append(f"      Trace:    {e.trace_id}")
            lines.append(f"      Rubric:   {e.rubric_id} (v{e.rubric_version})")
            lines.append(f"      Judge:    {e.judge_model_id}")
            lines.append(f"      Date:     {e.evaluated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            lines.append(f"      Result:   {status}")
            lines.append(
                f"      Score:    {e.composite_score.value:.3f} "
                f"({e.composite_score.dimensions_passed}/{e.composite_score.dimensions_total} dimensions passed)"
            )
            lines.append("")
            lines.append("      Dimension Breakdown:")
            lines.append(f"      {'Dimension':<30} {'Score':>6} {'Status':>8} {'Confidence':>12}")
            lines.append(f"      {'-' * 60}")
            for ds in e.dimension_scores:
                ds_status = "PASS" if ds.passed else "FAIL"
                lines.append(
                    f"      {ds.dimension_name:<30} {ds.score:>6.3f} {ds_status:>8} {ds.confidence.value:>12}"
                )
            lines.append("")
            for ds in e.dimension_scores:
                lines.append(f"      [{ds.dimension_name}] {ds.rationale}")
            lines.append("")
            lines.append(f"  {'-' * 68}")
            lines.append("")

        lines.append(sep)
        lines.append("  CRUCIBLE™ — Smeyatsky Labs Ltd — Machine-Generated Evidence Report")
        lines.append(sep)

        return "\n".join(lines)
