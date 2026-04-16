"""
CRUCIBLE CLI

Architectural Intent:
- Command-line interface for CRUCIBLE evaluation harness
- Provides commands for server management, rubric loading, and quick evaluations
- Presentation layer — delegates to application layer use cases
"""

from __future__ import annotations

import asyncio
import json
import sys

import click
import yaml

from application.dtos.rubric_dto import CreateRubricRequest, RubricDimensionDTO
from infrastructure.config.dependency_injection import Container
from infrastructure.config.settings import Settings


def _run_async(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


@click.group()
@click.pass_context
def cli(ctx):
    """CRUCIBLE™ — Agentic Evaluation Harness"""
    ctx.ensure_object(dict)


@cli.command()
@click.option("--host", default="0.0.0.0", help="API host")
@click.option("--port", default=8100, type=int, help="API port")
def serve(host: str, port: int):
    """Start the CRUCIBLE API server."""
    import uvicorn
    from presentation.api.app import create_app

    settings = Settings.from_env()
    app = create_app(settings)
    uvicorn.run(app, host=host, port=port)


@cli.command()
@click.argument("rubric_file", type=click.Path(exists=True))
def load_rubric(rubric_file: str):
    """Load a rubric from a YAML file."""

    async def _load():
        with open(rubric_file) as f:
            data = yaml.safe_load(f)

        container = Container()
        await container.init()

        try:
            request = CreateRubricRequest(
                name=data["name"],
                description=data["description"],
                dimensions=[RubricDimensionDTO(**d) for d in data["dimensions"]],
                agent_type=data.get("agent_type"),
                owner=data.get("owner", ""),
            )
            use_case = container.create_rubric()
            rubric = await use_case.execute(request)
            click.echo(f"Rubric created: {rubric.name} (id={rubric.id}, v{rubric.version})")
        finally:
            await container.shutdown()

    _run_async(_load())


@cli.command()
@click.option("--name", required=True, help="Agent name")
@click.option("--description", required=True, help="Agent description")
@click.option("--type", "agent_type", required=True, help="Agent type")
def register(name: str, description: str, agent_type: str):
    """Register a new agent."""

    async def _register():
        container = Container()
        await container.init()
        try:
            use_case = container.register_agent()
            agent = await use_case.execute(
                name=name, description=description, agent_type=agent_type,
            )
            click.echo(f"Agent registered: {agent.name} (id={agent.id})")
        finally:
            await container.shutdown()

    _run_async(_register())


@cli.command()
def agents():
    """List all registered agents."""

    async def _list():
        container = Container()
        await container.init()
        try:
            query = container.list_agents_query()
            result = await query.execute()
            if not result:
                click.echo("No agents registered.")
                return
            for a in result:
                click.echo(f"  {a.name} ({a.agent_type}) — {a.id}")
        finally:
            await container.shutdown()

    _run_async(_list())


@cli.command()
@click.argument("agent_id")
@click.option("--limit", default=10, help="Number of evaluations to show")
def scores(agent_id: str, limit: int):
    """Show score timeline for an agent."""

    async def _scores():
        container = Container()
        await container.init()
        try:
            query = container.get_scores_query()
            timeline = await query.execute(agent_id, limit)
            if not timeline.evaluations:
                click.echo("No evaluations found.")
                return
            for e in timeline.evaluations:
                status = "PASS" if e.composite_score.passed else "FAIL"
                click.echo(
                    f"  [{status}] {e.composite_score.value:.3f} "
                    f"({e.composite_score.dimensions_passed}/{e.composite_score.dimensions_total}) "
                    f"— {e.evaluated_at.isoformat()}"
                )
            if timeline.regressions:
                click.echo("\nRegressions detected:")
                for eval_id, r in timeline.regressions:
                    click.echo(
                        f"  [{r.severity.upper()}] delta={r.delta:+.3f} "
                        f"({r.baseline_score:.3f} -> {r.current_score:.3f})"
                    )
        finally:
            await container.shutdown()

    _run_async(_scores())


if __name__ == "__main__":
    cli()
