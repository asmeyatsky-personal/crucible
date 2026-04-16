"""Rubric API Routes — presentation layer for rubric management."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from application.dtos.rubric_dto import CreateRubricRequest
from infrastructure.config.dependency_injection import Container


def create_rubric_router(container: Container) -> APIRouter:
    router = APIRouter()

    @router.post("")
    async def create_rubric(request: CreateRubricRequest):
        try:
            use_case = container.create_rubric()
            rubric = await use_case.execute(request)
            return {
                "id": rubric.id,
                "name": rubric.name,
                "description": rubric.description,
                "version": rubric.version,
                "dimension_count": len(rubric.dimensions),
                "agent_type": rubric.agent_type,
                "created_at": rubric.created_at.isoformat(),
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.get("")
    async def list_rubrics():
        rubrics = await container.rubric_repository.list_all()
        return [
            {
                "id": r.id,
                "name": r.name,
                "description": r.description,
                "version": r.version,
                "dimension_count": len(r.dimensions),
                "agent_type": r.agent_type,
            }
            for r in rubrics
        ]

    @router.get("/{rubric_id}")
    async def get_rubric(rubric_id: str):
        rubric = await container.rubric_repository.get_by_id(rubric_id)
        if rubric is None:
            raise HTTPException(status_code=404, detail="Rubric not found")
        return {
            "id": rubric.id,
            "name": rubric.name,
            "description": rubric.description,
            "version": rubric.version,
            "agent_type": rubric.agent_type,
            "owner": rubric.owner,
            "dimensions": [
                {
                    "name": d.name,
                    "description": d.description,
                    "scoring_method": d.scoring_method.value,
                    "weight": d.weight,
                    "pass_threshold": d.pass_threshold,
                }
                for d in rubric.dimensions
            ],
            "created_at": rubric.created_at.isoformat(),
            "updated_at": rubric.updated_at.isoformat(),
        }

    return router
