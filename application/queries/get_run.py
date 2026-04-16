"""
Get Run Query — retrieves trace and evaluation data for a specific run.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.entities.evaluation import Evaluation
from domain.entities.trace import Trace
from domain.ports.evaluation_repository_port import EvaluationRepositoryPort
from domain.ports.trace_repository_port import TraceRepositoryPort


@dataclass
class RunDetail:
    trace: Trace
    evaluation: Evaluation | None


class GetRunQuery:
    def __init__(
        self,
        trace_repository: TraceRepositoryPort,
        evaluation_repository: EvaluationRepositoryPort,
    ):
        self._trace_repo = trace_repository
        self._eval_repo = evaluation_repository

    async def execute(self, run_id: str) -> RunDetail | None:
        trace = await self._trace_repo.get_by_run_id(run_id)
        if trace is None:
            return None
        evaluations = await self._eval_repo.list_by_agent(trace.agent_id, limit=100)
        evaluation = next(
            (e for e in evaluations if e.run_id == run_id), None,
        )
        return RunDetail(trace=trace, evaluation=evaluation)
