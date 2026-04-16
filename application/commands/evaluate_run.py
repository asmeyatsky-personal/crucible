"""
Evaluate Run Use Case (JUDGE + SCORE Modules)

Architectural Intent:
- Orchestrates the full evaluation pipeline: fetch trace + rubric, run judge
  on each dimension (parallelised), compute composite score, persist evaluation
- This is the core use case of CRUCIBLE Phase 1
- Dimensions are evaluated concurrently (parallelism-first design)

Parallelization Notes:
- All rubric dimensions are evaluated in parallel via asyncio.gather
- Scoring and persistence are sequential (depend on judge results)
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

from application.dtos.evaluation_dto import EvaluateRunRequest
from domain.entities.evaluation import Evaluation
from domain.ports.evaluation_repository_port import EvaluationRepositoryPort
from domain.ports.event_bus_port import EventBusPort
from domain.ports.judge_port import JudgePort
from domain.ports.rubric_repository_port import RubricRepositoryPort
from domain.ports.trace_repository_port import TraceRepositoryPort
from domain.services.regression_detection_service import RegressionDetectionService
from domain.services.scoring_service import ScoringService
from domain.value_objects.judge_config import ConsensusMode, JudgeConfig, JudgeModel
from domain.value_objects.score import DimensionScore


class EvaluateRunUseCase:
    def __init__(
        self,
        trace_repository: TraceRepositoryPort,
        rubric_repository: RubricRepositoryPort,
        evaluation_repository: EvaluationRepositoryPort,
        judge: JudgePort,
        scoring_service: ScoringService,
        regression_service: RegressionDetectionService,
        event_bus: EventBusPort,
    ):
        self._trace_repo = trace_repository
        self._rubric_repo = rubric_repository
        self._eval_repo = evaluation_repository
        self._judge = judge
        self._scoring = scoring_service
        self._regression = regression_service
        self._event_bus = event_bus

    async def execute(self, request: EvaluateRunRequest) -> Evaluation:
        # Fetch trace and rubric concurrently
        trace, rubric = await asyncio.gather(
            self._trace_repo.get_by_id(request.trace_id),
            self._rubric_repo.get_by_id(request.rubric_id),
        )

        if trace is None:
            raise ValueError(f"Trace not found: {request.trace_id}")
        if rubric is None:
            raise ValueError(f"Rubric not found: {request.rubric_id}")

        judge_config = JudgeConfig(
            primary_model=JudgeModel(request.judge_config.primary_model),
            consensus_mode=ConsensusMode(request.judge_config.consensus_mode),
            secondary_models=tuple(
                JudgeModel(m) for m in request.judge_config.secondary_models
            ),
            temperature=request.judge_config.temperature,
            max_tokens=request.judge_config.max_tokens,
            custom_system_prompt=request.judge_config.custom_system_prompt,
        )

        # Evaluate all dimensions in parallel (parallelism-first)
        dimension_scores: list[DimensionScore] = await asyncio.gather(
            *(
                self._judge.evaluate_dimension(trace, dim, judge_config)
                for dim in rubric.dimensions
            )
        )

        # Compute composite score
        composite_score = self._scoring.compute_composite_score(
            dimension_scores, rubric,
        )

        # Create evaluation
        evaluation = Evaluation.create(
            id=str(uuid4()),
            trace_id=trace.id,
            rubric_id=rubric.id,
            rubric_version=rubric.version,
            agent_id=trace.agent_id,
            run_id=trace.run_id,
            judge_config=judge_config,
            composite_score=composite_score,
            dimension_scores=dimension_scores,
            judge_model_id=judge_config.primary_model.value,
        )

        # Check for regression against baseline
        baseline = await self._eval_repo.get_latest_by_agent(trace.agent_id)
        regression = self._regression.check_regression(evaluation, baseline)

        events = list(evaluation.domain_events)
        if regression.regressed:
            from domain.entities.evaluation import RegressionDetectedEvent
            events.append(
                RegressionDetectedEvent(
                    aggregate_id=evaluation.id,
                    agent_id=trace.agent_id,
                    current_score=regression.current_score,
                    baseline_score=regression.baseline_score,
                    delta=regression.delta,
                )
            )

        # Persist and publish
        await self._eval_repo.save(evaluation)
        await self._event_bus.publish(events)

        return evaluation.clear_events()
