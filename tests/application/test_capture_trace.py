"""
Capture Trace Use Case Tests — mocked ports, verify orchestration.
"""

import pytest
from unittest.mock import AsyncMock

from application.commands.capture_trace import CaptureTraceUseCase
from application.dtos.trace_dto import CaptureTraceRequest, TrajectoryStepDTO
from domain.entities.agent import Agent
from datetime import UTC, datetime


class TestCaptureTraceUseCase:
    @pytest.fixture
    def deps(self):
        return {
            "trace_repository": AsyncMock(),
            "agent_repository": AsyncMock(),
            "storage": AsyncMock(),
            "event_bus": AsyncMock(),
        }

    @pytest.fixture
    def use_case(self, deps):
        return CaptureTraceUseCase(**deps)

    @pytest.mark.asyncio
    async def test_successful_capture(self, use_case, deps):
        agent = Agent.register(
            id="a-1", name="test-agent",
            description="Test", agent_type="research",
        ).clear_events()
        deps["agent_repository"].get_by_id.return_value = agent

        request = CaptureTraceRequest(
            agent_id="a-1",
            run_id="run-1",
            steps=[
                TrajectoryStepDTO(
                    step_index=0, step_type="user_message",
                    content="Hello", timestamp=datetime.now(UTC),
                ),
                TrajectoryStepDTO(
                    step_index=1, step_type="assistant_response",
                    content="Hi there", timestamp=datetime.now(UTC),
                ),
            ],
            model_id="claude-sonnet-4-6",
        )

        trace = await use_case.execute(request)

        assert trace.agent_id == "a-1"
        assert trace.step_count == 2
        deps["trace_repository"].save.assert_awaited_once()
        deps["storage"].store.assert_awaited_once()
        deps["event_bus"].publish.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_agent_not_found_raises(self, use_case, deps):
        deps["agent_repository"].get_by_id.return_value = None

        request = CaptureTraceRequest(
            agent_id="bad",
            run_id="run-1",
            steps=[
                TrajectoryStepDTO(
                    step_index=0, step_type="user_message",
                    content="Hello", timestamp=datetime.now(UTC),
                ),
            ],
            model_id="model",
        )

        with pytest.raises(ValueError, match="Agent not found"):
            await use_case.execute(request)
