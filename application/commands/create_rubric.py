"""
Create Rubric Use Case (RUBRIC Module)

Architectural Intent:
- Creates a new versioned evaluation rubric from a DTO request
- Validates and converts dimensions, persists via repository port
"""

from __future__ import annotations

from uuid import uuid4

from application.dtos.rubric_dto import CreateRubricRequest
from domain.entities.rubric import Rubric
from domain.ports.event_bus_port import EventBusPort
from domain.ports.rubric_repository_port import RubricRepositoryPort
from domain.value_objects.rubric_dimension import RubricDimension, ScoringMethod


class CreateRubricUseCase:
    def __init__(
        self,
        rubric_repository: RubricRepositoryPort,
        event_bus: EventBusPort,
    ):
        self._repo = rubric_repository
        self._event_bus = event_bus

    async def execute(self, request: CreateRubricRequest) -> Rubric:
        existing = await self._repo.get_by_name(request.name)
        if existing is not None:
            raise ValueError(f"Rubric with name '{request.name}' already exists")

        dimensions = [
            RubricDimension(
                name=d.name,
                description=d.description,
                scoring_method=ScoringMethod(d.scoring_method),
                weight=d.weight,
                pass_threshold=d.pass_threshold,
                judge_prompt=d.judge_prompt,
                expected_value=d.expected_value,
            )
            for d in request.dimensions
        ]

        rubric = Rubric.create(
            id=str(uuid4()),
            name=request.name,
            description=request.description,
            dimensions=dimensions,
            agent_type=request.agent_type,
            owner=request.owner,
        )

        await self._repo.save(rubric)
        await self._event_bus.publish(list(rubric.domain_events))
        return rubric.clear_events()
