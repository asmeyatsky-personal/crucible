"""
Evaluation Workflow Orchestrator

Architectural Intent:
- DAG-based orchestration for the full evaluation pipeline
- Decomposes evaluation into independent steps, parallelises where possible
- Implements the Parallel-Safe Orchestration pattern from skill2026

Parallelization Notes:
- validate_trace and validate_rubric run concurrently (no data dependency)
- evaluate_dimensions fans out all dimensions in parallel
- compute_score and persist_evaluation are sequential (depend on judge results)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine


@dataclass
class WorkflowStep:
    """A single step in a DAG-based workflow."""

    name: str
    execute: Callable[..., Coroutine[Any, Any, Any]]
    depends_on: list[str] = field(default_factory=list)


class DAGOrchestrator:
    """
    Executes workflow steps respecting dependency order,
    parallelising independent steps automatically.
    """

    def __init__(self, steps: list[WorkflowStep]):
        self.steps = {s.name: s for s in steps}
        self._validate_no_cycles()

    def _validate_no_cycles(self) -> None:
        """Detect circular dependencies via topological sort attempt."""
        visited: set[str] = set()
        in_progress: set[str] = set()

        def visit(name: str) -> None:
            if name in in_progress:
                raise ValueError(f"Circular dependency detected involving: {name}")
            if name in visited:
                return
            in_progress.add(name)
            for dep in self.steps[name].depends_on:
                if dep not in self.steps:
                    raise ValueError(f"Unknown dependency: {dep}")
                visit(dep)
            in_progress.discard(name)
            visited.add(name)

        for name in self.steps:
            visit(name)

    async def execute(self, context: dict) -> dict[str, Any]:
        """Execute all steps, parallelising where dependencies allow."""
        completed: dict[str, Any] = {}
        pending = set(self.steps.keys())

        while pending:
            ready = [
                name for name in pending
                if all(dep in completed for dep in self.steps[name].depends_on)
            ]
            if not ready:
                raise RuntimeError("Deadlock: no steps can proceed")

            results = await asyncio.gather(
                *(
                    self.steps[name].execute(context, completed)
                    for name in ready
                ),
                return_exceptions=True,
            )

            for name, result in zip(ready, results):
                if isinstance(result, Exception):
                    raise RuntimeError(f"Step '{name}' failed: {result}") from result
                completed[name] = result
                pending.discard(name)

        return completed
