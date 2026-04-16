"""
Tool Call Value Object

Architectural Intent:
- Immutable representation of a single tool invocation within an agent trajectory
- Captures name, input, output, timing, and success/failure status
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ToolCall:
    """A single tool call within an agent trajectory step."""

    name: str
    input_parameters: dict
    output: str | dict | None
    latency_ms: float
    success: bool
    timestamp: datetime
    error: str | None = None
