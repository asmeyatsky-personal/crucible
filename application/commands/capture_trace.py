"""
Capture Trace Use Case (TRACE Module)

Architectural Intent:
- Captures an agent execution trajectory as an immutable TRACE artefact
- Validates agent existence, converts DTOs to domain objects, persists
- Stores the raw trace artefact in blob storage for durability
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

from application.dtos.trace_dto import CaptureTraceRequest
from domain.entities.trace import Trace
from domain.ports.agent_repository_port import AgentRepositoryPort
from domain.ports.event_bus_port import EventBusPort
from domain.ports.storage_port import StoragePort
from domain.ports.trace_repository_port import TraceRepositoryPort
from domain.value_objects.tool_call import ToolCall
from domain.value_objects.trajectory_step import StepType, TrajectoryStep


class CaptureTraceUseCase:
    def __init__(
        self,
        trace_repository: TraceRepositoryPort,
        agent_repository: AgentRepositoryPort,
        storage: StoragePort,
        event_bus: EventBusPort,
    ):
        self._trace_repo = trace_repository
        self._agent_repo = agent_repository
        self._storage = storage
        self._event_bus = event_bus

    async def execute(self, request: CaptureTraceRequest) -> Trace:
        agent = await self._agent_repo.get_by_id(request.agent_id)
        if agent is None:
            raise ValueError(f"Agent not found: {request.agent_id}")

        steps = [
            TrajectoryStep(
                step_index=s.step_index,
                step_type=StepType(s.step_type),
                content=s.content,
                timestamp=s.timestamp,
                tool_calls=tuple(
                    ToolCall(
                        name=tc.name,
                        input_parameters=tc.input_parameters,
                        output=tc.output,
                        latency_ms=tc.latency_ms,
                        success=tc.success,
                        timestamp=tc.timestamp,
                        error=tc.error,
                    )
                    for tc in s.tool_calls
                ),
                token_count=s.token_count,
                model_id=s.model_id,
                metadata=s.metadata,
            )
            for s in request.steps
        ]

        trace = Trace.capture(
            id=str(uuid4()),
            agent_id=request.agent_id,
            run_id=request.run_id,
            steps=steps,
            model_id=request.model_id,
            temperature=request.temperature,
            total_tokens=request.total_tokens,
            total_latency_ms=request.total_latency_ms,
            metadata=request.metadata,
        )

        # Persist trace metadata and store raw artefact in parallel concept,
        # but sequential here for consistency
        await self._trace_repo.save(trace)
        await self._storage.store(
            f"traces/{trace.id}.json",
            json.dumps(_serialize_trace(trace)).encode(),
        )
        await self._event_bus.publish(list(trace.domain_events))

        return trace.clear_events()


def _serialize_trace(trace: Trace) -> dict:
    """Serialize trace to a JSON-compatible dict for blob storage."""
    return {
        "id": trace.id,
        "agent_id": trace.agent_id,
        "run_id": trace.run_id,
        "model_id": trace.model_id,
        "temperature": trace.temperature,
        "total_tokens": trace.total_tokens,
        "total_latency_ms": trace.total_latency_ms,
        "captured_at": trace.captured_at.isoformat(),
        "step_count": trace.step_count,
        "steps": [
            {
                "step_index": step.step_index,
                "step_type": step.step_type.value,
                "content": step.content,
                "timestamp": step.timestamp.isoformat(),
                "tool_calls": [
                    {
                        "name": tc.name,
                        "input_parameters": tc.input_parameters,
                        "output": tc.output,
                        "latency_ms": tc.latency_ms,
                        "success": tc.success,
                        "timestamp": tc.timestamp.isoformat(),
                        "error": tc.error,
                    }
                    for tc in step.tool_calls
                ],
                "token_count": step.token_count,
                "model_id": step.model_id,
            }
            for step in trace.steps
        ],
        "metadata": trace.metadata,
    }
