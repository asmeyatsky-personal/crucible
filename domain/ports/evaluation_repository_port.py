"""
Evaluation Repository Port

Architectural Intent:
- Contract for evaluation persistence and querying
- Supports retrieval by agent, run, or rubric for trend analysis
"""

from __future__ import annotations

from typing import Protocol

from domain.entities.evaluation import Evaluation


class EvaluationRepositoryPort(Protocol):
    async def save(self, evaluation: Evaluation) -> None: ...
    async def get_by_id(self, evaluation_id: str) -> Evaluation | None: ...
    async def list_by_agent(
        self, agent_id: str, limit: int = 50,
    ) -> list[Evaluation]: ...
    async def list_by_rubric(
        self, rubric_id: str, limit: int = 50,
    ) -> list[Evaluation]: ...
    async def get_latest_by_agent(self, agent_id: str) -> Evaluation | None: ...
