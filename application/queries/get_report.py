"""
Get Report Query — retrieves a generated report by ID.
"""

from __future__ import annotations

from domain.entities.evaluation import Evaluation
from domain.ports.evaluation_repository_port import EvaluationRepositoryPort


class GetEvaluationQuery:
    def __init__(self, evaluation_repository: EvaluationRepositoryPort):
        self._repo = evaluation_repository

    async def execute(self, evaluation_id: str) -> Evaluation | None:
        return await self._repo.get_by_id(evaluation_id)
