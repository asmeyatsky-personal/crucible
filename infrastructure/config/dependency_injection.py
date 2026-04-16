"""
Dependency Injection Container (Composition Root)

Architectural Intent:
- Wires all implementations to their port interfaces at application startup
- Single place where infrastructure choices are made
- All use cases and queries are resolved with their dependencies injected
"""

from __future__ import annotations

from application.commands.capture_trace import CaptureTraceUseCase
from application.commands.create_rubric import CreateRubricUseCase
from application.commands.evaluate_run import EvaluateRunUseCase
from application.commands.generate_report import GenerateReportUseCase
from application.commands.register_agent import RegisterAgentUseCase
from application.queries.get_report import GetEvaluationQuery
from application.queries.get_run import GetRunQuery
from application.queries.get_scores import GetScoresQuery
from application.queries.list_agents import ListAgentsQuery
from domain.services.regression_detection_service import RegressionDetectionService
from domain.services.scoring_service import ScoringService
from infrastructure.adapters.claude_judge_adapter import ClaudeJudgeAdapter
from infrastructure.adapters.event_bus_adapter import InMemoryEventBusAdapter
from infrastructure.adapters.local_storage_adapter import LocalStorageAdapter
from infrastructure.adapters.report_export_adapter import ReportExportAdapter
from infrastructure.config.settings import Settings
from infrastructure.repositories.sqlite_repository import (
    SQLiteAgentRepository,
    SQLiteDatabase,
    SQLiteEvaluationRepository,
    SQLiteRubricRepository,
    SQLiteTraceRepository,
)


class Container:
    """Dependency injection container — the composition root of CRUCIBLE."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()

        # Infrastructure
        self.database = SQLiteDatabase(
            db_path=self.settings.database_url.replace("sqlite+aiosqlite:///", "")
            if "sqlite" in self.settings.database_url
            else "crucible.db"
        )
        self.event_bus = InMemoryEventBusAdapter()
        self.storage = LocalStorageAdapter(self.settings.storage_path)

        # Repositories
        self.agent_repository = SQLiteAgentRepository(self.database)
        self.trace_repository = SQLiteTraceRepository(self.database)
        self.rubric_repository = SQLiteRubricRepository(self.database)
        self.evaluation_repository = SQLiteEvaluationRepository(self.database)

        # Adapters
        self.judge = ClaudeJudgeAdapter(self.settings.anthropic_api_key)
        self.report_export = ReportExportAdapter()

        # Domain Services
        self.scoring_service = ScoringService()
        self.regression_service = RegressionDetectionService()

    async def init(self) -> None:
        """Initialise async resources (database connections, etc.)."""
        await self.database.connect()

    async def shutdown(self) -> None:
        """Clean up async resources."""
        await self.database.close()

    # --- Use Case Factories ---

    def register_agent(self) -> RegisterAgentUseCase:
        return RegisterAgentUseCase(
            agent_repository=self.agent_repository,
            event_bus=self.event_bus,
        )

    def capture_trace(self) -> CaptureTraceUseCase:
        return CaptureTraceUseCase(
            trace_repository=self.trace_repository,
            agent_repository=self.agent_repository,
            storage=self.storage,
            event_bus=self.event_bus,
        )

    def create_rubric(self) -> CreateRubricUseCase:
        return CreateRubricUseCase(
            rubric_repository=self.rubric_repository,
            event_bus=self.event_bus,
        )

    def evaluate_run(self) -> EvaluateRunUseCase:
        return EvaluateRunUseCase(
            trace_repository=self.trace_repository,
            rubric_repository=self.rubric_repository,
            evaluation_repository=self.evaluation_repository,
            judge=self.judge,
            scoring_service=self.scoring_service,
            regression_service=self.regression_service,
            event_bus=self.event_bus,
        )

    def generate_report(self) -> GenerateReportUseCase:
        return GenerateReportUseCase(
            evaluation_repository=self.evaluation_repository,
            report_export=self.report_export,
            event_bus=self.event_bus,
        )

    # --- Query Factories ---

    def get_run_query(self) -> GetRunQuery:
        return GetRunQuery(
            trace_repository=self.trace_repository,
            evaluation_repository=self.evaluation_repository,
        )

    def get_scores_query(self) -> GetScoresQuery:
        return GetScoresQuery(
            evaluation_repository=self.evaluation_repository,
            regression_service=self.regression_service,
        )

    def list_agents_query(self) -> ListAgentsQuery:
        return ListAgentsQuery(agent_repository=self.agent_repository)

    def get_evaluation_query(self) -> GetEvaluationQuery:
        return GetEvaluationQuery(evaluation_repository=self.evaluation_repository)
