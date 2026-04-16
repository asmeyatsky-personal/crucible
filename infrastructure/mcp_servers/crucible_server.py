"""
CRUCIBLE MCP Server

Architectural Intent:
- Exposes CRUCIBLE capabilities as an MCP server for agent-native integration
- Tools = write operations (commands): register_agent, capture_trace,
  create_rubric, evaluate_run, generate_report
- Resources = read operations (queries): agents, traces, evaluations, rubrics
- One MCP server per bounded context — CRUCIBLE is a single context

MCP Integration:
- This is the primary MCP integration point for CRUCIBLE
- Agents can use CRUCIBLE directly through MCP tool calls
- Aligns to skill2026 Rule 6: MCP-Compliant Service Boundaries

Parallelization Notes:
- MCP tool calls are inherently parallel-safe (each is independent)
- evaluate_run internally parallelises dimension evaluation
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool

from application.dtos.evaluation_dto import EvaluateRunRequest, JudgeConfigDTO
from application.dtos.report_dto import GenerateReportRequest
from application.dtos.rubric_dto import CreateRubricRequest, RubricDimensionDTO
from application.dtos.trace_dto import CaptureTraceRequest, TrajectoryStepDTO
from infrastructure.config.dependency_injection import Container

# Tool definitions — schema for each CRUCIBLE tool
TOOLS = [
    Tool(
        name="register_agent",
        description="Register a new agent identity in CRUCIBLE for evaluation tracking.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Agent name"},
                "description": {"type": "string", "description": "Agent description"},
                "agent_type": {"type": "string", "description": "Agent type (research, coding, customer_support, data_extraction)"},
            },
            "required": ["name", "description", "agent_type"],
        },
    ),
    Tool(
        name="capture_trace",
        description="Capture an agent execution trajectory as an immutable TRACE artefact.",
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "run_id": {"type": "string"},
                "steps": {"type": "array", "items": {"type": "object"}},
                "model_id": {"type": "string"},
                "temperature": {"type": "number", "default": 0.0},
                "total_tokens": {"type": "integer", "default": 0},
                "total_latency_ms": {"type": "number", "default": 0.0},
            },
            "required": ["agent_id", "run_id", "steps", "model_id"],
        },
    ),
    Tool(
        name="create_rubric",
        description="Create a new evaluation rubric with scored dimensions.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "dimensions": {"type": "array", "items": {"type": "object"}},
                "agent_type": {"type": "string"},
            },
            "required": ["name", "description", "dimensions"],
        },
    ),
    Tool(
        name="evaluate_run",
        description="Evaluate an agent run's trace against a rubric using LLM-as-judge.",
        inputSchema={
            "type": "object",
            "properties": {
                "trace_id": {"type": "string"},
                "rubric_id": {"type": "string"},
                "judge_model": {"type": "string", "default": "claude-sonnet-4-6"},
            },
            "required": ["trace_id", "rubric_id"],
        },
    ),
    Tool(
        name="generate_report",
        description="Generate an evidence report from evaluation results.",
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "title": {"type": "string"},
                "export_format": {"type": "string", "default": "json"},
            },
            "required": ["agent_id", "title"],
        },
    ),
]


def create_crucible_mcp_server(container: Container) -> Server:
    """
    Create the CRUCIBLE MCP server.

    Each bounded context should have exactly one MCP server.
    Tools = write operations, Resources = read operations.
    """
    server = Server("crucible-service")

    @server.list_tools()
    async def handle_list_tools() -> list[Tool]:
        return TOOLS

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        result = await _dispatch_tool(container, name, arguments)
        return [TextContent(type="text", text=result)]

    return server


async def _dispatch_tool(
    container: Container, name: str, arguments: dict[str, Any],
) -> str:
    """Route MCP tool calls to the appropriate use case."""

    if name == "register_agent":
        use_case = container.register_agent()
        agent = await use_case.execute(
            name=arguments["name"],
            description=arguments["description"],
            agent_type=arguments["agent_type"],
        )
        return json.dumps({
            "id": agent.id, "name": agent.name,
            "agent_type": agent.agent_type, "status": "registered",
        })

    elif name == "capture_trace":
        request = CaptureTraceRequest(
            agent_id=arguments["agent_id"],
            run_id=arguments["run_id"],
            steps=[TrajectoryStepDTO(**s) for s in arguments["steps"]],
            model_id=arguments["model_id"],
            temperature=arguments.get("temperature", 0.0),
            total_tokens=arguments.get("total_tokens", 0),
            total_latency_ms=arguments.get("total_latency_ms", 0.0),
        )
        use_case = container.capture_trace()
        trace = await use_case.execute(request)
        return json.dumps({
            "id": trace.id, "agent_id": trace.agent_id,
            "run_id": trace.run_id, "step_count": trace.step_count,
            "status": "captured",
        })

    elif name == "create_rubric":
        request = CreateRubricRequest(
            name=arguments["name"],
            description=arguments["description"],
            dimensions=[RubricDimensionDTO(**d) for d in arguments["dimensions"]],
            agent_type=arguments.get("agent_type"),
        )
        use_case = container.create_rubric()
        rubric = await use_case.execute(request)
        return json.dumps({
            "id": rubric.id, "name": rubric.name,
            "version": rubric.version,
            "dimension_count": len(rubric.dimensions),
            "status": "created",
        })

    elif name == "evaluate_run":
        request = EvaluateRunRequest(
            trace_id=arguments["trace_id"],
            rubric_id=arguments["rubric_id"],
            judge_config=JudgeConfigDTO(
                primary_model=arguments.get("judge_model", "claude-sonnet-4-6"),
            ),
        )
        use_case = container.evaluate_run()
        evaluation = await use_case.execute(request)
        return json.dumps({
            "id": evaluation.id,
            "composite_score": evaluation.composite_score.value,
            "passed": evaluation.composite_score.passed,
            "dimensions_passed": evaluation.composite_score.dimensions_passed,
            "dimensions_total": evaluation.composite_score.dimensions_total,
            "status": "evaluated",
        })

    elif name == "generate_report":
        request = GenerateReportRequest(
            agent_id=arguments["agent_id"],
            title=arguments["title"],
            export_format=arguments.get("export_format", "json"),
        )
        use_case = container.generate_report()
        report = await use_case.execute(request)
        return json.dumps({
            "id": report.id, "title": report.title,
            "export_format": report.export_format.value,
            "status": "generated",
        })

    else:
        return json.dumps({"error": f"Unknown tool: {name}"})
