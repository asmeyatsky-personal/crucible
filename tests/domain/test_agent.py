"""
Agent Entity Tests — pure domain logic, no mocks needed.
"""

from domain.entities.agent import Agent, AgentRegisteredEvent, RubricAssignedEvent


class TestAgent:
    def test_register_creates_agent_with_event(self):
        agent = Agent.register(
            id="a-1", name="Test Agent",
            description="A test agent", agent_type="research",
        )
        assert agent.id == "a-1"
        assert agent.name == "Test Agent"
        assert agent.agent_type == "research"
        assert len(agent.domain_events) == 1
        event = agent.domain_events[0]
        assert isinstance(event, AgentRegisteredEvent)
        assert event.agent_name == "Test Agent"

    def test_assign_rubric(self):
        agent = Agent.register(
            id="a-1", name="Agent", description="desc", agent_type="coding",
        )
        updated = agent.assign_rubric("r-1")
        assert "r-1" in updated.rubric_ids
        assert len(updated.domain_events) == 2
        assert isinstance(updated.domain_events[1], RubricAssignedEvent)
        assert updated.domain_events[1].rubric_id == "r-1"

    def test_assign_rubric_duplicate_returns_self(self):
        """Assigning a rubric that is already assigned returns the same instance."""
        agent = Agent.register(
            id="a-1", name="Agent", description="desc", agent_type="coding",
        )
        with_rubric = agent.assign_rubric("r-1")
        duplicate = with_rubric.assign_rubric("r-1")
        assert duplicate is with_rubric  # exact same object returned
        assert duplicate.rubric_ids == ("r-1",)
        # No new event added
        assert len(duplicate.domain_events) == len(with_rubric.domain_events)

    def test_clear_events(self):
        agent = Agent.register(
            id="a-1", name="Agent", description="desc", agent_type="research",
        )
        assert len(agent.domain_events) == 1
        cleared = agent.clear_events()
        assert len(cleared.domain_events) == 0
        assert len(agent.domain_events) == 1  # original unchanged
