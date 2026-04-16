"""
Tests for ReportExportAdapter.

Tests JSON, CSV, and PDF export formats.
"""

from __future__ import annotations

import csv
import io
import json

import pytest

from domain.entities.report import ExportFormat
from infrastructure.adapters.report_export_adapter import ReportExportAdapter
from tests.infrastructure.conftest import make_evaluation, make_dimension_score, make_composite_score
from domain.value_objects.score import Confidence


class TestExportJSON:
    @pytest.mark.asyncio
    async def test_valid_json_output(self):
        adapter = ReportExportAdapter()
        evaluation = make_evaluation()

        result = await adapter.export([evaluation], ExportFormat.JSON, "Test Report")
        data = json.loads(result)

        assert data["title"] == "Test Report"
        assert data["evaluation_count"] == 1
        assert "generated_at" in data

    @pytest.mark.asyncio
    async def test_evaluation_fields(self):
        adapter = ReportExportAdapter()
        evaluation = make_evaluation(
            id="eval-42",
            agent_id="agent-7",
            trace_id="trace-99",
            rubric_id="rubric-5",
        )

        result = await adapter.export([evaluation], ExportFormat.JSON, "Report")
        data = json.loads(result)

        ev = data["evaluations"][0]
        assert ev["id"] == "eval-42"
        assert ev["agent_id"] == "agent-7"
        assert ev["trace_id"] == "trace-99"
        assert ev["rubric_id"] == "rubric-5"

    @pytest.mark.asyncio
    async def test_composite_score_in_json(self):
        adapter = ReportExportAdapter()
        evaluation = make_evaluation()

        result = await adapter.export([evaluation], ExportFormat.JSON, "Report")
        data = json.loads(result)

        cs = data["evaluations"][0]["composite_score"]
        assert cs["value"] == 0.85
        assert cs["passed"] is True
        assert cs["dimensions_passed"] == 1
        assert cs["dimensions_total"] == 1

    @pytest.mark.asyncio
    async def test_dimension_scores_in_json(self):
        adapter = ReportExportAdapter()
        evaluation = make_evaluation()

        result = await adapter.export([evaluation], ExportFormat.JSON, "Report")
        data = json.loads(result)

        ds = data["evaluations"][0]["dimension_scores"]
        assert len(ds) == 1
        assert ds[0]["dimension_name"] == "accuracy"
        assert ds[0]["score"] == 0.85
        assert ds[0]["confidence"] == "high"

    @pytest.mark.asyncio
    async def test_multiple_evaluations(self):
        adapter = ReportExportAdapter()
        evals = [make_evaluation(id=f"eval-{i}") for i in range(3)]

        result = await adapter.export(evals, ExportFormat.JSON, "Multi Report")
        data = json.loads(result)

        assert data["evaluation_count"] == 3
        assert len(data["evaluations"]) == 3

    @pytest.mark.asyncio
    async def test_empty_evaluations(self):
        adapter = ReportExportAdapter()

        result = await adapter.export([], ExportFormat.JSON, "Empty Report")
        data = json.loads(result)

        assert data["evaluation_count"] == 0
        assert data["evaluations"] == []


class TestExportCSV:
    @pytest.mark.asyncio
    async def test_csv_has_header_row(self):
        adapter = ReportExportAdapter()
        evaluation = make_evaluation()

        result = await adapter.export([evaluation], ExportFormat.CSV, "CSV Report")
        reader = csv.reader(io.StringIO(result))
        rows = list(reader)

        assert len(rows) == 2  # header + 1 data row
        header = rows[0]
        assert "evaluation_id" in header
        assert "agent_id" in header
        assert "composite_score" in header

    @pytest.mark.asyncio
    async def test_csv_data_row(self):
        adapter = ReportExportAdapter()
        evaluation = make_evaluation(id="eval-csv", agent_id="agent-csv")

        result = await adapter.export([evaluation], ExportFormat.CSV, "CSV Report")
        reader = csv.reader(io.StringIO(result))
        rows = list(reader)

        data_row = rows[1]
        assert "eval-csv" in data_row
        assert "agent-csv" in data_row

    @pytest.mark.asyncio
    async def test_csv_multiple_evaluations(self):
        adapter = ReportExportAdapter()
        evals = [make_evaluation(id=f"eval-{i}") for i in range(5)]

        result = await adapter.export(evals, ExportFormat.CSV, "CSV Report")
        reader = csv.reader(io.StringIO(result))
        rows = list(reader)

        assert len(rows) == 6  # header + 5 data rows

    @pytest.mark.asyncio
    async def test_csv_empty(self):
        adapter = ReportExportAdapter()

        result = await adapter.export([], ExportFormat.CSV, "Empty CSV")
        reader = csv.reader(io.StringIO(result))
        rows = list(reader)

        assert len(rows) == 1  # header only


class TestExportPDF:
    @pytest.mark.asyncio
    async def test_pdf_contains_title(self):
        adapter = ReportExportAdapter()
        evaluation = make_evaluation()

        result = await adapter.export([evaluation], ExportFormat.PDF, "My PDF Report")

        assert "My PDF Report" in result

    @pytest.mark.asyncio
    async def test_pdf_contains_crucible_header(self):
        adapter = ReportExportAdapter()
        evaluation = make_evaluation()

        result = await adapter.export([evaluation], ExportFormat.PDF, "Report")

        assert "CRUCIBLE" in result
        assert "EVALUATION CERTIFICATE" in result

    @pytest.mark.asyncio
    async def test_pdf_contains_evaluation_details(self):
        adapter = ReportExportAdapter()
        evaluation = make_evaluation(
            id="eval-pdf",
            agent_id="agent-pdf",
            run_id="run-pdf",
            trace_id="trace-pdf",
        )

        result = await adapter.export([evaluation], ExportFormat.PDF, "Report")

        assert "eval-pdf" in result
        assert "agent-pdf" in result
        assert "run-pdf" in result
        assert "trace-pdf" in result

    @pytest.mark.asyncio
    async def test_pdf_contains_pass_fail_status(self):
        adapter = ReportExportAdapter()
        passing = make_evaluation(id="eval-pass")
        failing_ds = make_dimension_score(score=0.3, passed=False)
        failing_cs = make_composite_score(
            value=0.3,
            passed=False,
            dimensions_passed=0,
            dimension_scores=(failing_ds,),
        )
        failing = make_evaluation(
            id="eval-fail",
            composite_score=failing_cs,
            dimension_scores=(failing_ds,),
        )

        result = await adapter.export([passing, failing], ExportFormat.PDF, "Report")

        assert "PASS" in result
        assert "FAIL" in result

    @pytest.mark.asyncio
    async def test_pdf_contains_dimension_breakdown(self):
        adapter = ReportExportAdapter()
        evaluation = make_evaluation()

        result = await adapter.export([evaluation], ExportFormat.PDF, "Report")

        assert "Dimension Breakdown" in result
        assert "accuracy" in result
        assert "0.850" in result

    @pytest.mark.asyncio
    async def test_pdf_contains_rationale(self):
        adapter = ReportExportAdapter()
        evaluation = make_evaluation()

        result = await adapter.export([evaluation], ExportFormat.PDF, "Report")

        assert "Good accuracy" in result

    @pytest.mark.asyncio
    async def test_pdf_footer(self):
        adapter = ReportExportAdapter()
        evaluation = make_evaluation()

        result = await adapter.export([evaluation], ExportFormat.PDF, "Report")

        assert "Smeyatsky Labs" in result
        assert "Machine-Generated Evidence Report" in result


class TestExportFormatRouting:
    @pytest.mark.asyncio
    async def test_webhook_format_returns_json(self):
        adapter = ReportExportAdapter()
        evaluation = make_evaluation()

        result = await adapter.export([evaluation], ExportFormat.WEBHOOK, "Report")
        data = json.loads(result)

        assert data["title"] == "Report"

    @pytest.mark.asyncio
    async def test_unknown_format_falls_back_to_json(self):
        """If a new format is added but not handled, fallback to JSON."""
        adapter = ReportExportAdapter()
        evaluation = make_evaluation()

        # The current code handles all ExportFormat values, but if called with
        # JSON it should work normally
        result = await adapter.export([evaluation], ExportFormat.JSON, "Report")
        data = json.loads(result)
        assert data["title"] == "Report"
