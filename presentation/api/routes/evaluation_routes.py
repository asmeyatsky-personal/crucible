"""Evaluation API Routes — presentation layer for running evaluations and viewing scores."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from application.dtos.evaluation_dto import EvaluateRunRequest
from infrastructure.config.dependency_injection import Container


def create_evaluation_router(container: Container) -> APIRouter:
    router = APIRouter()

    @router.post("")
    async def evaluate_run(request: EvaluateRunRequest):
        try:
            use_case = container.evaluate_run()
            evaluation = await use_case.execute(request)
            return {
                "id": evaluation.id,
                "trace_id": evaluation.trace_id,
                "rubric_id": evaluation.rubric_id,
                "rubric_version": evaluation.rubric_version,
                "agent_id": evaluation.agent_id,
                "run_id": evaluation.run_id,
                "composite_score": {
                    "value": evaluation.composite_score.value,
                    "passed": evaluation.composite_score.passed,
                    "dimensions_passed": evaluation.composite_score.dimensions_passed,
                    "dimensions_total": evaluation.composite_score.dimensions_total,
                },
                "dimension_scores": [
                    {
                        "dimension_name": ds.dimension_name,
                        "score": ds.score,
                        "passed": ds.passed,
                        "rationale": ds.rationale,
                        "confidence": ds.confidence.value,
                    }
                    for ds in evaluation.dimension_scores
                ],
                "judge_model_id": evaluation.judge_model_id,
                "evaluated_at": evaluation.evaluated_at.isoformat(),
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.get("/{evaluation_id}")
    async def get_evaluation(evaluation_id: str):
        query = container.get_evaluation_query()
        evaluation = await query.execute(evaluation_id)
        if evaluation is None:
            raise HTTPException(status_code=404, detail="Evaluation not found")
        return {
            "id": evaluation.id,
            "trace_id": evaluation.trace_id,
            "rubric_id": evaluation.rubric_id,
            "rubric_version": evaluation.rubric_version,
            "agent_id": evaluation.agent_id,
            "run_id": evaluation.run_id,
            "composite_score": {
                "value": evaluation.composite_score.value,
                "passed": evaluation.composite_score.passed,
                "dimensions_passed": evaluation.composite_score.dimensions_passed,
                "dimensions_total": evaluation.composite_score.dimensions_total,
            },
            "dimension_scores": [
                {
                    "dimension_name": ds.dimension_name,
                    "score": ds.score,
                    "passed": ds.passed,
                    "rationale": ds.rationale,
                    "confidence": ds.confidence.value,
                }
                for ds in evaluation.dimension_scores
            ],
            "judge_model_id": evaluation.judge_model_id,
            "evaluated_at": evaluation.evaluated_at.isoformat(),
        }

    @router.get("/agent/{agent_id}/scores")
    async def get_agent_scores(agent_id: str, limit: int = 50):
        query = container.get_scores_query()
        timeline = await query.execute(agent_id, limit)
        return {
            "agent_id": agent_id,
            "evaluation_count": len(timeline.evaluations),
            "scores": [
                {
                    "evaluation_id": e.id,
                    "composite_score": e.composite_score.value,
                    "passed": e.composite_score.passed,
                    "evaluated_at": e.evaluated_at.isoformat(),
                }
                for e in timeline.evaluations
            ],
            "regressions": [
                {
                    "evaluation_id": eval_id,
                    "current_score": r.current_score,
                    "baseline_score": r.baseline_score,
                    "delta": r.delta,
                    "severity": r.severity,
                }
                for eval_id, r in timeline.regressions
            ],
        }

    return router
