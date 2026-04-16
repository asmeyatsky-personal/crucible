"""
CRUCIBLE REST API Application

Architectural Intent:
- FastAPI application serving the CRUCIBLE evaluation harness API
- Presentation layer only — delegates all logic to application layer use cases
- Manages application lifecycle (startup/shutdown) via DI container
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from infrastructure.config.dependency_injection import Container
from infrastructure.config.settings import Settings
from presentation.api.routes.agent_routes import create_agent_router
from presentation.api.routes.evaluation_routes import create_evaluation_router
from presentation.api.routes.report_routes import create_report_router
from presentation.api.routes.rubric_routes import create_rubric_router
from presentation.api.routes.trace_routes import create_trace_router

container: Container | None = None


def get_container() -> Container:
    if container is None:
        raise RuntimeError("Container not initialised")
    return container


def create_app(settings: Settings | None = None) -> FastAPI:
    global container
    container = Container(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await container.init()
        yield
        await container.shutdown()

    app = FastAPI(
        title="CRUCIBLE™",
        description="Agentic Evaluation Harness — Heat-test your agents. Ship what survives.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.include_router(
        create_agent_router(container), prefix="/api/v1/agents", tags=["Agents"],
    )
    app.include_router(
        create_trace_router(container), prefix="/api/v1/traces", tags=["Traces"],
    )
    app.include_router(
        create_rubric_router(container), prefix="/api/v1/rubrics", tags=["Rubrics"],
    )
    app.include_router(
        create_evaluation_router(container), prefix="/api/v1/evaluations", tags=["Evaluations"],
    )
    app.include_router(
        create_report_router(container), prefix="/api/v1/reports", tags=["Reports"],
    )

    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "crucible", "version": "1.0.0"}

    return app
