"""
Report Entity Tests — pure domain logic, no mocks needed.
"""

from domain.entities.report import ExportFormat, Report, ReportGeneratedEvent


class TestReport:
    def test_generate_creates_report_with_event(self):
        report = Report.generate(
            id="rpt-1", title="Weekly Report", agent_id="a-1",
            evaluation_ids=["e-1", "e-2", "e-3"],
            export_format=ExportFormat.JSON,
            content='{"summary": "all good"}',
        )
        assert report.id == "rpt-1"
        assert report.title == "Weekly Report"
        assert report.agent_id == "a-1"
        assert report.evaluation_ids == ("e-1", "e-2", "e-3")
        assert report.export_format == ExportFormat.JSON
        assert report.content == '{"summary": "all good"}'
        assert report.metadata == {}

        assert len(report.domain_events) == 1
        event = report.domain_events[0]
        assert isinstance(event, ReportGeneratedEvent)
        assert event.aggregate_id == "rpt-1"
        assert event.export_format == "json"
        assert event.evaluation_count == 3

    def test_generate_with_metadata(self):
        report = Report.generate(
            id="rpt-2", title="Report", agent_id="a-1",
            evaluation_ids=["e-1"],
            export_format=ExportFormat.PDF,
            content="pdf bytes here",
            metadata={"author": "test"},
        )
        assert report.metadata == {"author": "test"}

    def test_clear_events(self):
        report = Report.generate(
            id="rpt-1", title="Report", agent_id="a-1",
            evaluation_ids=["e-1"],
            export_format=ExportFormat.CSV,
            content="col1,col2",
        )
        assert len(report.domain_events) == 1
        cleared = report.clear_events()
        assert len(cleared.domain_events) == 0
        assert len(report.domain_events) == 1  # original unchanged

    def test_report_is_immutable(self):
        report = Report.generate(
            id="rpt-1", title="Report", agent_id="a-1",
            evaluation_ids=["e-1"],
            export_format=ExportFormat.WEBHOOK,
            content="payload",
        )
        try:
            report.id = "changed"
            assert False, "Should not be able to modify frozen dataclass"
        except AttributeError:
            pass
