"""Trace API Routes — presentation layer for trajectory capture and retrieval."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from application.dtos.trace_dto import CaptureTraceRequest
from infrastructure.config.dependency_injection import Container


def create_trace_router(container: Container) -> APIRouter:
    router = APIRouter()

    @router.post("")
    async def capture_trace(request: CaptureTraceRequest):
        try:
            use_case = container.capture_trace()
            trace = await use_case.execute(request)
            return {
                "id": trace.id,
                "agent_id": trace.agent_id,
                "run_id": trace.run_id,
                "model_id": trace.model_id,
                "step_count": trace.step_count,
                "total_tokens": trace.total_tokens,
                "total_latency_ms": trace.total_latency_ms,
                "captured_at": trace.captured_at.isoformat(),
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.get("/{trace_id}")
    async def get_trace(trace_id: str):
        trace = await container.trace_repository.get_by_id(trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="Trace not found")
        return {
            "id": trace.id,
            "agent_id": trace.agent_id,
            "run_id": trace.run_id,
            "model_id": trace.model_id,
            "step_count": trace.step_count,
            "total_tokens": trace.total_tokens,
            "total_latency_ms": trace.total_latency_ms,
            "captured_at": trace.captured_at.isoformat(),
            "steps": [
                {
                    "step_index": s.step_index,
                    "step_type": s.step_type.value,
                    "content": s.content[:500],
                    "timestamp": s.timestamp.isoformat(),
                    "tool_call_count": len(s.tool_calls),
                }
                for s in trace.steps
            ],
        }

    @router.get("/agent/{agent_id}")
    async def list_traces_by_agent(agent_id: str, limit: int = 50):
        traces = await container.trace_repository.list_by_agent(agent_id, limit)
        return [
            {
                "id": t.id,
                "run_id": t.run_id,
                "model_id": t.model_id,
                "step_count": t.step_count,
                "captured_at": t.captured_at.isoformat(),
            }
            for t in traces
        ]

    return router
