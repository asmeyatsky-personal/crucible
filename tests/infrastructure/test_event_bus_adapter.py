"""
Tests for InMemoryEventBusAdapter.

Tests subscribe+publish and handler error handling.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from domain.events.event_base import DomainEvent
from domain.entities.trace import TraceCapturedEvent
from domain.entities.evaluation import EvaluationCompletedEvent
from infrastructure.adapters.event_bus_adapter import InMemoryEventBusAdapter


class TestSubscribeAndPublish:
    @pytest.mark.asyncio
    async def test_handler_called_on_publish(self):
        bus = InMemoryEventBusAdapter()
        handler = AsyncMock()

        await bus.subscribe(TraceCapturedEvent, handler)
        event = TraceCapturedEvent(aggregate_id="t-1", agent_id="a-1", run_id="r-1", step_count=5)
        await bus.publish([event])

        handler.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_multiple_handlers_for_same_event(self):
        bus = InMemoryEventBusAdapter()
        handler1 = AsyncMock()
        handler2 = AsyncMock()

        await bus.subscribe(TraceCapturedEvent, handler1)
        await bus.subscribe(TraceCapturedEvent, handler2)

        event = TraceCapturedEvent(aggregate_id="t-1", agent_id="a-1", run_id="r-1", step_count=3)
        await bus.publish([event])

        handler1.assert_called_once_with(event)
        handler2.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_handler_not_called_for_different_event_type(self):
        bus = InMemoryEventBusAdapter()
        handler = AsyncMock()

        await bus.subscribe(TraceCapturedEvent, handler)

        event = EvaluationCompletedEvent(
            aggregate_id="e-1", trace_id="t-1", rubric_id="r-1",
            composite_score=0.9, passed=True,
        )
        await bus.publish([event])

        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_publish_multiple_events(self):
        bus = InMemoryEventBusAdapter()
        handler = AsyncMock()

        await bus.subscribe(TraceCapturedEvent, handler)

        events = [
            TraceCapturedEvent(aggregate_id=f"t-{i}", agent_id="a-1", run_id=f"r-{i}", step_count=i)
            for i in range(3)
        ]
        await bus.publish(events)

        assert handler.call_count == 3

    @pytest.mark.asyncio
    async def test_publish_empty_list(self):
        bus = InMemoryEventBusAdapter()
        handler = AsyncMock()

        await bus.subscribe(TraceCapturedEvent, handler)
        await bus.publish([])

        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_publish_with_no_subscribers(self):
        bus = InMemoryEventBusAdapter()
        event = TraceCapturedEvent(aggregate_id="t-1", agent_id="a-1", run_id="r-1", step_count=1)

        # Should not raise
        await bus.publish([event])


class TestHandlerErrorHandling:
    @pytest.mark.asyncio
    async def test_handler_error_does_not_stop_other_handlers(self):
        bus = InMemoryEventBusAdapter()

        failing_handler = AsyncMock(side_effect=ValueError("Handler broke"))
        succeeding_handler = AsyncMock()

        await bus.subscribe(TraceCapturedEvent, failing_handler)
        await bus.subscribe(TraceCapturedEvent, succeeding_handler)

        event = TraceCapturedEvent(aggregate_id="t-1", agent_id="a-1", run_id="r-1", step_count=1)
        await bus.publish([event])

        failing_handler.assert_called_once()
        succeeding_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_handler_error_does_not_stop_other_events(self):
        bus = InMemoryEventBusAdapter()

        call_count = 0

        async def counting_handler(event):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("First event fails")

        await bus.subscribe(TraceCapturedEvent, counting_handler)

        events = [
            TraceCapturedEvent(aggregate_id="t-1", agent_id="a-1", run_id="r-1", step_count=1),
            TraceCapturedEvent(aggregate_id="t-2", agent_id="a-1", run_id="r-2", step_count=2),
        ]
        await bus.publish(events)

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_handler_error_is_logged(self, caplog):
        bus = InMemoryEventBusAdapter()
        failing_handler = AsyncMock(side_effect=TypeError("Bad type"))

        await bus.subscribe(TraceCapturedEvent, failing_handler)

        event = TraceCapturedEvent(aggregate_id="t-1", agent_id="a-1", run_id="r-1", step_count=1)

        import logging
        with caplog.at_level(logging.ERROR):
            await bus.publish([event])

        assert "Event handler failed" in caplog.text
        assert "TraceCapturedEvent" in caplog.text
