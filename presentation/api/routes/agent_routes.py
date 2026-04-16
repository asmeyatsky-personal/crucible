"""Agent API Routes — presentation layer for agent registration and listing."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from infrastructure.config.dependency_injection import Container


class RegisterAgentRequest(BaseModel):
    name: str
    description: str
    agent_type: str
    metadata: dict | None = None


def create_agent_router(container: Container) -> APIRouter:
    router = APIRouter()

    @router.post("")
    async def register_agent(request: RegisterAgentRequest):
        try:
            use_case = container.register_agent()
            agent = await use_case.execute(
                name=request.name,
                description=request.description,
                agent_type=request.agent_type,
                metadata=request.metadata,
            )
            return {
                "id": agent.id,
                "name": agent.name,
                "description": agent.description,
                "agent_type": agent.agent_type,
                "created_at": agent.created_at.isoformat(),
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.get("")
    async def list_agents():
        query = container.list_agents_query()
        agents = await query.execute()
        return [
            {
                "id": a.id,
                "name": a.name,
                "description": a.description,
                "agent_type": a.agent_type,
                "rubric_ids": list(a.rubric_ids),
                "created_at": a.created_at.isoformat(),
            }
            for a in agents
        ]

    @router.get("/{agent_id}")
    async def get_agent(agent_id: str):
        agent = await container.agent_repository.get_by_id(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        return {
            "id": agent.id,
            "name": agent.name,
            "description": agent.description,
            "agent_type": agent.agent_type,
            "rubric_ids": list(agent.rubric_ids),
            "created_at": agent.created_at.isoformat(),
            "updated_at": agent.updated_at.isoformat(),
            "metadata": agent.metadata,
        }

    return router
