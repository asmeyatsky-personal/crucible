"""
Trace Entity Tests — pure domain logic, no mocks needed.
"""

from datetime import UTC, datetime

from domain.entities.trace import Trace, TraceCapturedEvent
from domain.value_objects.tool_call import ToolCall
from domain.value_objects.trajectory_step import StepType, TrajectoryStep


def _make_step(index: int, step_type: StepType = StepType.REASONING) -> TrajectoryStep:
    return TrajectoryStep(
        step_index=index,
        step_type=step_type,
        content=f"Step {index} content",
        timestamp=datetime.now(UTC),
    )


def _make_tool_step(index: int) -> TrajectoryStep:
    return TrajectoryStep(
        step_index=index,
        step_type=StepType.TOOL_CALL,
        content="Calling search tool",
        timestamp=datetime.now(UTC),
        tool_calls=(
            ToolCall(
                name="search",
                input_parameters={"query": "test"},
                output="results",
                latency_ms=150.0,
                success=True,
                timestamp=datetime.now(UTC),
            ),
        ),
    )


class TestTrace:
    def test_capture_creates_immutable_trace(self):
        steps = [_make_step(0, StepType.SYSTEM_PROMPT), _make_step(1)]
        trace = Trace.capture(
            id="t-1", agent_id="a-1", run_id="r-1", steps=steps,
            model_id="claude-sonnet-4-6", temperature=0.0,
            total_tokens=500, total_latency_ms=1200.0,
        )
        assert trace.id == "t-1"
        assert trace.agent_id == "a-1"
        assert trace.step_count == 2
        assert trace.steps[0].step_type == StepType.SYSTEM_PROMPT

    def test_capture_sorts_steps_by_index(self):
        steps = [_make_step(2), _make_step(0), _make_step(1)]
        trace = Trace.capture(
            id="t-1", agent_id="a-1", run_id="r-1", steps=steps,
            model_id="model", temperature=0.0,
            total_tokens=0, total_latency_ms=0.0,
        )
        assert [s.step_index for s in trace.steps] == [0, 1, 2]

    def test_capture_produces_domain_event(self):
        steps = [_make_step(0)]
        trace = Trace.capture(
            id="t-1", agent_id="a-1", run_id="r-1", steps=steps,
            model_id="model", temperature=0.0,
            total_tokens=0, total_latency_ms=0.0,
        )
        assert len(trace.domain_events) == 1
        event = trace.domain_events[0]
        assert isinstance(event, TraceCapturedEvent)
        assert event.agent_id == "a-1"
        assert event.step_count == 1

    def test_tool_calls_extracted_across_steps(self):
        steps = [_make_tool_step(0), _make_tool_step(1), _make_step(2)]
        trace = Trace.capture(
            id="t-1", agent_id="a-1", run_id="r-1", steps=steps,
            model_id="model", temperature=0.0,
            total_tokens=0, total_latency_ms=0.0,
        )
        assert len(trace.tool_calls) == 2
        assert all(tc.name == "search" for tc in trace.tool_calls)

    def test_clear_events_returns_new_instance(self):
        steps = [_make_step(0)]
        trace = Trace.capture(
            id="t-1", agent_id="a-1", run_id="r-1", steps=steps,
            model_id="model", temperature=0.0,
            total_tokens=0, total_latency_ms=0.0,
        )
        cleared = trace.clear_events()
        assert len(cleared.domain_events) == 0
        assert len(trace.domain_events) == 1  # Original unchanged
        assert cleared is not trace

    def test_trace_is_immutable(self):
        steps = [_make_step(0)]
        trace = Trace.capture(
            id="t-1", agent_id="a-1", run_id="r-1", steps=steps,
            model_id="model", temperature=0.0,
            total_tokens=0, total_latency_ms=0.0,
        )
        try:
            trace.id = "changed"
            assert False, "Should not be able to modify frozen dataclass"
        except AttributeError:
            pass
