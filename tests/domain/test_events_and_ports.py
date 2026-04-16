"""
Tests for domain event re-export modules and ports — pure imports, no mocks needed.
"""


class TestEvaluationEventsReExport:
    def test_evaluation_completed_event_importable(self):
        from domain.events.evaluation_events import EvaluationCompletedEvent
        assert EvaluationCompletedEvent is not None

    def test_regression_detected_event_importable(self):
        from domain.events.evaluation_events import RegressionDetectedEvent
        assert RegressionDetectedEvent is not None

    def test_all_exports(self):
        from domain.events import evaluation_events
        assert "EvaluationCompletedEvent" in evaluation_events.__all__
        assert "RegressionDetectedEvent" in evaluation_events.__all__


class TestReportEventsReExport:
    def test_report_generated_event_importable(self):
        from domain.events.report_events import ReportGeneratedEvent
        assert ReportGeneratedEvent is not None

    def test_all_exports(self):
        from domain.events import report_events
        assert "ReportGeneratedEvent" in report_events.__all__


class TestTraceEventsReExport:
    def test_trace_captured_event_importable(self):
        from domain.events.trace_events import TraceCapturedEvent
        assert TraceCapturedEvent is not None

    def test_all_exports(self):
        from domain.events import trace_events
        assert "TraceCapturedEvent" in trace_events.__all__


class TestNotificationPort:
    def test_notification_port_is_protocol(self):
        from domain.ports.notification_port import NotificationPort
        from typing import Protocol
        assert NotificationPort is not None
        assert issubclass(NotificationPort, Protocol)

    def test_notification_port_has_send_method(self):
        from domain.ports.notification_port import NotificationPort
        assert hasattr(NotificationPort, "send")


class TestAgentRepositoryPort:
    def test_importable_and_is_protocol(self):
        from domain.ports.agent_repository_port import AgentRepositoryPort
        from typing import Protocol
        assert issubclass(AgentRepositoryPort, Protocol)

    def test_has_expected_methods(self):
        from domain.ports.agent_repository_port import AgentRepositoryPort
        assert hasattr(AgentRepositoryPort, "save")
        assert hasattr(AgentRepositoryPort, "get_by_id")
        assert hasattr(AgentRepositoryPort, "get_by_name")
        assert hasattr(AgentRepositoryPort, "list_all")


class TestEvaluationRepositoryPort:
    def test_importable_and_is_protocol(self):
        from domain.ports.evaluation_repository_port import EvaluationRepositoryPort
        from typing import Protocol
        assert issubclass(EvaluationRepositoryPort, Protocol)

    def test_has_expected_methods(self):
        from domain.ports.evaluation_repository_port import EvaluationRepositoryPort
        assert hasattr(EvaluationRepositoryPort, "save")
        assert hasattr(EvaluationRepositoryPort, "get_by_id")
        assert hasattr(EvaluationRepositoryPort, "get_latest_by_agent")


class TestEventBusPort:
    def test_importable_and_is_protocol(self):
        from domain.ports.event_bus_port import EventBusPort
        from typing import Protocol
        assert issubclass(EventBusPort, Protocol)

    def test_has_expected_methods(self):
        from domain.ports.event_bus_port import EventBusPort
        assert hasattr(EventBusPort, "publish")
        assert hasattr(EventBusPort, "subscribe")


class TestJudgePort:
    def test_importable_and_is_protocol(self):
        from domain.ports.judge_port import JudgePort
        from typing import Protocol
        assert issubclass(JudgePort, Protocol)

    def test_has_expected_methods(self):
        from domain.ports.judge_port import JudgePort
        assert hasattr(JudgePort, "evaluate_dimension")


class TestReportExportPort:
    def test_importable_and_is_protocol(self):
        from domain.ports.report_export_port import ReportExportPort
        from typing import Protocol
        assert issubclass(ReportExportPort, Protocol)

    def test_has_expected_methods(self):
        from domain.ports.report_export_port import ReportExportPort
        assert hasattr(ReportExportPort, "export")


class TestRubricRepositoryPort:
    def test_importable_and_is_protocol(self):
        from domain.ports.rubric_repository_port import RubricRepositoryPort
        from typing import Protocol
        assert issubclass(RubricRepositoryPort, Protocol)

    def test_has_expected_methods(self):
        from domain.ports.rubric_repository_port import RubricRepositoryPort
        assert hasattr(RubricRepositoryPort, "save")
        assert hasattr(RubricRepositoryPort, "get_by_id")
        assert hasattr(RubricRepositoryPort, "list_by_agent_type")


class TestStoragePort:
    def test_importable_and_is_protocol(self):
        from domain.ports.storage_port import StoragePort
        from typing import Protocol
        assert issubclass(StoragePort, Protocol)

    def test_has_expected_methods(self):
        from domain.ports.storage_port import StoragePort
        assert hasattr(StoragePort, "store")
        assert hasattr(StoragePort, "retrieve")
        assert hasattr(StoragePort, "delete")
        assert hasattr(StoragePort, "exists")


class TestTraceRepositoryPort:
    def test_importable_and_is_protocol(self):
        from domain.ports.trace_repository_port import TraceRepositoryPort
        from typing import Protocol
        assert issubclass(TraceRepositoryPort, Protocol)

    def test_has_expected_methods(self):
        from domain.ports.trace_repository_port import TraceRepositoryPort
        assert hasattr(TraceRepositoryPort, "save")
        assert hasattr(TraceRepositoryPort, "get_by_id")
        assert hasattr(TraceRepositoryPort, "list_by_agent")
