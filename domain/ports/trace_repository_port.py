"""
Trace Repository Port

Architectural Intent:
- Defines the contract for trace persistence
- Lives in domain layer — implementations in infrastructure
- Supports async operations for parallelism-first design
"""

from __future__ import annotations

from typing import Protocol

from domain.entities.trace import Trace


class TraceRepositoryPort(Protocol):
    async def save(self, trace: Trace) -> None: ...
    async def get_by_id(self, trace_id: str) -> Trace | None: ...
    async def get_by_run_id(self, run_id: str) -> Trace | None: ...
    async def list_by_agent(self, agent_id: str, limit: int = 50) -> list[Trace]: ...
