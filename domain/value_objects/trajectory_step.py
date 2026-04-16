"""
Trajectory Step Value Object

Architectural Intent:
- Immutable representation of a single step in an agent execution trajectory
- Each step may contain reasoning, tool calls, or final output
- Steps are ordered and compose a full TRACE artefact
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from domain.value_objects.tool_call import ToolCall


class StepType(Enum):
    SYSTEM_PROMPT = "system_prompt"
    USER_MESSAGE = "user_message"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ASSISTANT_RESPONSE = "assistant_response"
    FINAL_OUTPUT = "final_output"


@dataclass(frozen=True)
class TrajectoryStep:
    """A single step in an agent's execution trajectory."""

    step_index: int
    step_type: StepType
    content: str
    timestamp: datetime
    tool_calls: tuple[ToolCall, ...] = field(default=())
    token_count: int | None = None
    model_id: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError(f"step_index must be non-negative, got {self.step_index}")
