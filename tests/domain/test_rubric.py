"""
Rubric Entity Tests — pure domain logic, no mocks needed.
"""

import pytest

from domain.entities.rubric import Rubric, RubricCreatedEvent, RubricUpdatedEvent
from domain.value_objects.rubric_dimension import RubricDimension, ScoringMethod


def _dim(name: str, weight: float = 1.0) -> RubricDimension:
    return RubricDimension(
        name=name,
        description=f"Test dimension: {name}",
        scoring_method=ScoringMethod.LLM_JUDGE,
        weight=weight,
        pass_threshold=0.7,
        judge_prompt=f"Evaluate {name}",
    )


class TestRubric:
    def test_create_rubric_with_dimensions(self):
        rubric = Rubric.create(
            id="r-1", name="Test Rubric", description="A test rubric",
            dimensions=[_dim("goal_adherence"), _dim("output_quality")],
        )
        assert rubric.id == "r-1"
        assert rubric.version == 1
        assert len(rubric.dimensions) == 2

    def test_create_produces_domain_event(self):
        rubric = Rubric.create(
            id="r-1", name="Test Rubric", description="desc",
            dimensions=[_dim("d1")],
        )
        assert len(rubric.domain_events) == 1
        event = rubric.domain_events[0]
        assert isinstance(event, RubricCreatedEvent)
        assert event.rubric_name == "Test Rubric"

    def test_create_fails_with_no_dimensions(self):
        with pytest.raises(ValueError, match="at least one dimension"):
            Rubric.create(
                id="r-1", name="Empty", description="No dims",
                dimensions=[],
            )

    def test_update_dimensions_increments_version(self):
        rubric = Rubric.create(
            id="r-1", name="Test", description="desc",
            dimensions=[_dim("d1")],
        )
        updated = rubric.update_dimensions([_dim("d1"), _dim("d2")])
        assert updated.version == 2
        assert len(updated.dimensions) == 2
        assert rubric.version == 1  # Original unchanged

    def test_update_produces_domain_event(self):
        rubric = Rubric.create(
            id="r-1", name="Test", description="desc",
            dimensions=[_dim("d1")],
        )
        updated = rubric.update_dimensions([_dim("d1", 0.5), _dim("d2", 0.5)])
        # Should have both creation and update events
        assert len(updated.domain_events) == 2
        assert isinstance(updated.domain_events[1], RubricUpdatedEvent)
        assert updated.domain_events[1].new_version == 2

    def test_normalised_weights(self):
        rubric = Rubric.create(
            id="r-1", name="Test", description="desc",
            dimensions=[_dim("d1", 2.0), _dim("d2", 3.0)],
        )
        weights = rubric.normalised_weights
        assert abs(weights["d1"] - 0.4) < 0.001
        assert abs(weights["d2"] - 0.6) < 0.001
        assert abs(sum(weights.values()) - 1.0) < 0.001

    def test_normalised_weights_zero_total(self):
        """When all dimension weights are 0, normalised_weights returns 0.0 for each."""
        dim_zero = RubricDimension(
            name="d1",
            description="desc",
            scoring_method=ScoringMethod.EXACT_MATCH,
            weight=0.0,
            pass_threshold=0.5,
        )
        # Bypass Rubric.create which validates; build directly to test the property.
        rubric = Rubric(
            id="r-1",
            name="Zero weights",
            description="desc",
            dimensions=(dim_zero,),
        )
        weights = rubric.normalised_weights
        assert weights == {"d1": 0.0}

    def test_clear_events(self):
        rubric = Rubric.create(
            id="r-1", name="Test", description="desc",
            dimensions=[_dim("d1")],
        )
        assert len(rubric.domain_events) == 1
        cleared = rubric.clear_events()
        assert len(cleared.domain_events) == 0
        assert len(rubric.domain_events) == 1  # original unchanged

    def test_rubric_is_immutable(self):
        rubric = Rubric.create(
            id="r-1", name="Test", description="desc",
            dimensions=[_dim("d1")],
        )
        try:
            rubric.name = "Changed"
            assert False, "Should not be able to modify frozen dataclass"
        except AttributeError:
            pass
