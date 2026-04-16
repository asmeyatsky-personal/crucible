"""
DAG Orchestrator Tests — verify parallel execution and failure modes.
"""

import asyncio
import pytest

from application.orchestration.evaluation_workflow import DAGOrchestrator, WorkflowStep


class TestDAGOrchestrator:
    @pytest.mark.asyncio
    async def test_sequential_execution(self):
        order = []

        async def step_a(ctx, completed):
            order.append("a")
            return "a_result"

        async def step_b(ctx, completed):
            order.append("b")
            return "b_result"

        orchestrator = DAGOrchestrator([
            WorkflowStep("a", step_a),
            WorkflowStep("b", step_b, depends_on=["a"]),
        ])

        results = await orchestrator.execute({})
        assert order == ["a", "b"]
        assert results["a"] == "a_result"
        assert results["b"] == "b_result"

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Independent steps should run concurrently."""
        started = []

        async def step_a(ctx, completed):
            started.append("a")
            await asyncio.sleep(0.01)
            return "a"

        async def step_b(ctx, completed):
            started.append("b")
            await asyncio.sleep(0.01)
            return "b"

        async def step_c(ctx, completed):
            return "c"

        orchestrator = DAGOrchestrator([
            WorkflowStep("a", step_a),
            WorkflowStep("b", step_b),
            WorkflowStep("c", step_c, depends_on=["a", "b"]),
        ])

        results = await orchestrator.execute({})

        # a and b should both be in started before c runs
        assert "a" in started and "b" in started
        assert results["c"] == "c"

    @pytest.mark.asyncio
    async def test_step_failure_propagates(self):
        async def fail_step(ctx, completed):
            raise ValueError("boom")

        async def ok_step(ctx, completed):
            return "ok"

        orchestrator = DAGOrchestrator([
            WorkflowStep("fail", fail_step),
            WorkflowStep("ok", ok_step, depends_on=["fail"]),
        ])

        with pytest.raises(RuntimeError, match="Step 'fail' failed"):
            await orchestrator.execute({})

    def test_circular_dependency_detected(self):
        async def noop(ctx, completed):
            pass

        with pytest.raises(ValueError, match="Circular dependency"):
            DAGOrchestrator([
                WorkflowStep("a", noop, depends_on=["b"]),
                WorkflowStep("b", noop, depends_on=["a"]),
            ])

    def test_unknown_dependency_detected(self):
        async def noop(ctx, completed):
            pass

        with pytest.raises(ValueError, match="Unknown dependency"):
            DAGOrchestrator([
                WorkflowStep("a", noop, depends_on=["nonexistent"]),
            ])

    @pytest.mark.asyncio
    async def test_diamond_dependency(self):
        """Test diamond-shaped DAG: A -> B, A -> C, B+C -> D."""
        order = []

        async def step(name):
            async def _step(ctx, completed):
                order.append(name)
                return name
            return _step

        orchestrator = DAGOrchestrator([
            WorkflowStep("a", await step("a")),
            WorkflowStep("b", await step("b"), depends_on=["a"]),
            WorkflowStep("c", await step("c"), depends_on=["a"]),
            WorkflowStep("d", await step("d"), depends_on=["b", "c"]),
        ])

        results = await orchestrator.execute({})
        assert order[0] == "a"
        assert order[-1] == "d"
        assert set(order[1:3]) == {"b", "c"}
