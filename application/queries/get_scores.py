"""
Get Scores Query — retrieves score timeline and version comparison data.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.entities.evaluation import Evaluation
from domain.ports.evaluation_repository_port import EvaluationRepositoryPort
from domain.services.regression_detection_service import RegressionDetectionService
from domain.value_objects.score import RegressionResult


@dataclass
class ScoreTimeline:
    evaluations: list[Evaluation]
    regressions: list[tuple[str, RegressionResult]]


class GetScoresQuery:
    def __init__(
        self,
        evaluation_repository: EvaluationRepositoryPort,
        regression_service: RegressionDetectionService,
    ):
        self._eval_repo = evaluation_repository
        self._regression = regression_service

    async def execute(self, agent_id: str, limit: int = 50) -> ScoreTimeline:
        evaluations = await self._eval_repo.list_by_agent(agent_id, limit=limit)
        regressions: list[tuple[str, RegressionResult]] = []

        for i in range(1, len(evaluations)):
            result = self._regression.check_regression(
                evaluations[i], evaluations[i - 1],
            )
            if result.regressed:
                regressions.append((evaluations[i].id, result))

        return ScoreTimeline(evaluations=evaluations, regressions=regressions)
