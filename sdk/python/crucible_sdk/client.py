"""
CRUCIBLE SDK Client

Architectural Intent:
- High-level Python client for the CRUCIBLE REST API
- Provides the crucible.trace(), crucible.evaluate(), crucible.score() interface
- Wraps httpx for async HTTP communication
- Framework-agnostic — works with any agent orchestration
"""

from __future__ import annotations

from typing import Any

import httpx


class CrucibleClient:
    """
    CRUCIBLE™ Python SDK Client.

    Provides a clean interface for agent evaluation:
    - register_agent() — register an agent identity
    - trace() — capture an agent execution trajectory
    - evaluate() — evaluate a trace against a rubric
    - scores() — retrieve score timeline for an agent
    - report() — generate an evidence report
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8100",
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        self._base_url = base_url.rstrip("/")
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=timeout,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> CrucibleClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # --- Health ---

    async def health(self) -> dict:
        response = await self._client.get("/health")
        response.raise_for_status()
        return response.json()

    # --- Agents ---

    async def register_agent(
        self, name: str, description: str, agent_type: str,
        metadata: dict | None = None,
    ) -> dict:
        """Register a new agent identity in CRUCIBLE."""
        response = await self._client.post(
            "/api/v1/agents",
            json={
                "name": name,
                "description": description,
                "agent_type": agent_type,
                "metadata": metadata or {},
            },
        )
        response.raise_for_status()
        return response.json()

    async def list_agents(self) -> list[dict]:
        """List all registered agents."""
        response = await self._client.get("/api/v1/agents")
        response.raise_for_status()
        return response.json()

    async def get_agent(self, agent_id: str) -> dict:
        """Get agent details by ID."""
        response = await self._client.get(f"/api/v1/agents/{agent_id}")
        response.raise_for_status()
        return response.json()

    # --- Traces (TRACE Module) ---

    async def trace(
        self,
        agent_id: str,
        run_id: str,
        steps: list[dict],
        model_id: str,
        temperature: float = 0.0,
        total_tokens: int = 0,
        total_latency_ms: float = 0.0,
        metadata: dict | None = None,
    ) -> dict:
        """Capture an agent execution trajectory as a TRACE artefact."""
        response = await self._client.post(
            "/api/v1/traces",
            json={
                "agent_id": agent_id,
                "run_id": run_id,
                "steps": steps,
                "model_id": model_id,
                "temperature": temperature,
                "total_tokens": total_tokens,
                "total_latency_ms": total_latency_ms,
                "metadata": metadata or {},
            },
        )
        response.raise_for_status()
        return response.json()

    async def get_trace(self, trace_id: str) -> dict:
        """Get trace details by ID."""
        response = await self._client.get(f"/api/v1/traces/{trace_id}")
        response.raise_for_status()
        return response.json()

    # --- Rubrics (RUBRIC Module) ---

    async def create_rubric(
        self, name: str, description: str, dimensions: list[dict],
        agent_type: str | None = None, owner: str = "",
    ) -> dict:
        """Create a new evaluation rubric."""
        response = await self._client.post(
            "/api/v1/rubrics",
            json={
                "name": name,
                "description": description,
                "dimensions": dimensions,
                "agent_type": agent_type,
                "owner": owner,
            },
        )
        response.raise_for_status()
        return response.json()

    async def list_rubrics(self) -> list[dict]:
        """List all rubrics."""
        response = await self._client.get("/api/v1/rubrics")
        response.raise_for_status()
        return response.json()

    # --- Evaluations (JUDGE + SCORE Modules) ---

    async def evaluate(
        self,
        trace_id: str,
        rubric_id: str,
        judge_model: str = "claude-sonnet-4-6",
        temperature: float = 0.0,
    ) -> dict:
        """Evaluate a trace against a rubric using LLM-as-judge."""
        response = await self._client.post(
            "/api/v1/evaluations",
            json={
                "trace_id": trace_id,
                "rubric_id": rubric_id,
                "judge_config": {
                    "primary_model": judge_model,
                    "temperature": temperature,
                },
            },
        )
        response.raise_for_status()
        return response.json()

    async def get_evaluation(self, evaluation_id: str) -> dict:
        """Get evaluation details by ID."""
        response = await self._client.get(f"/api/v1/evaluations/{evaluation_id}")
        response.raise_for_status()
        return response.json()

    async def scores(self, agent_id: str, limit: int = 50) -> dict:
        """Get score timeline and regression data for an agent."""
        response = await self._client.get(
            f"/api/v1/evaluations/agent/{agent_id}/scores",
            params={"limit": limit},
        )
        response.raise_for_status()
        return response.json()

    # --- Reports (REPORT Module) ---

    async def report(
        self,
        agent_id: str,
        title: str,
        export_format: str = "json",
        evaluation_ids: list[str] | None = None,
    ) -> dict:
        """Generate an evidence report from evaluation results."""
        response = await self._client.post(
            "/api/v1/reports",
            json={
                "agent_id": agent_id,
                "title": title,
                "export_format": export_format,
                "evaluation_ids": evaluation_ids or [],
            },
        )
        response.raise_for_status()
        return response.json()
