"""
Rubric Repository Port

Architectural Intent:
- Contract for rubric persistence and retrieval
- Supports versioning — can retrieve specific or latest version
"""

from __future__ import annotations

from typing import Protocol

from domain.entities.rubric import Rubric


class RubricRepositoryPort(Protocol):
    async def save(self, rubric: Rubric) -> None: ...
    async def get_by_id(self, rubric_id: str) -> Rubric | None: ...
    async def get_by_name(self, name: str) -> Rubric | None: ...
    async def list_all(self) -> list[Rubric]: ...
    async def list_by_agent_type(self, agent_type: str) -> list[Rubric]: ...
