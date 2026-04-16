"""
Register Agent Use Case

Architectural Intent:
- Single-responsibility command: registers a new agent identity in CRUCIBLE
- Orchestrates domain object creation and persistence via ports
- Dispatches domain events after successful persistence
"""

from __future__ import annotations

from uuid import uuid4

from domain.entities.agent import Agent
from domain.ports.agent_repository_port import AgentRepositoryPort
from domain.ports.event_bus_port import EventBusPort


class RegisterAgentUseCase:
    def __init__(
        self,
        agent_repository: AgentRepositoryPort,
        event_bus: EventBusPort,
    ):
        self._repo = agent_repository
        self._event_bus = event_bus

    async def execute(
        self, name: str, description: str, agent_type: str,
        metadata: dict | None = None,
    ) -> Agent:
        existing = await self._repo.get_by_name(name)
        if existing is not None:
            raise ValueError(f"Agent with name '{name}' already exists")

        agent = Agent.register(
            id=str(uuid4()),
            name=name,
            description=description,
            agent_type=agent_type,
            metadata=metadata,
        )

        await self._repo.save(agent)
        await self._event_bus.publish(list(agent.domain_events))
        return agent.clear_events()
