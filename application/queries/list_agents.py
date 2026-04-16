"""
List Agents Query — retrieves all registered agents.
"""

from __future__ import annotations

from domain.entities.agent import Agent
from domain.ports.agent_repository_port import AgentRepositoryPort


class ListAgentsQuery:
    def __init__(self, agent_repository: AgentRepositoryPort):
        self._repo = agent_repository

    async def execute(self) -> list[Agent]:
        return await self._repo.list_all()
