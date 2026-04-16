"""
Agent Repository Port

Architectural Intent:
- Contract for agent identity persistence
"""

from __future__ import annotations

from typing import Protocol

from domain.entities.agent import Agent


class AgentRepositoryPort(Protocol):
    async def save(self, agent: Agent) -> None: ...
    async def get_by_id(self, agent_id: str) -> Agent | None: ...
    async def get_by_name(self, name: str) -> Agent | None: ...
    async def list_all(self) -> list[Agent]: ...
