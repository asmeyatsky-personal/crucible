"""
Judge Configuration Value Object

Architectural Intent:
- Immutable configuration for the JUDGE module's LLM-as-judge orchestration
- Specifies model(s), consensus mode, and evaluation parameters
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class JudgeModel(Enum):
    CLAUDE_SONNET = "claude-sonnet-4-6"
    GPT_4O = "gpt-4o"
    CUSTOM = "custom"


class ConsensusMode(Enum):
    SINGLE = "single"  # Use primary judge only
    MULTI = "multi"  # Run multiple judges, flag disagreements


@dataclass(frozen=True)
class JudgeConfig:
    """Configuration for LLM-as-judge evaluation."""

    primary_model: JudgeModel
    consensus_mode: ConsensusMode = ConsensusMode.SINGLE
    secondary_models: tuple[JudgeModel, ...] = field(default=())
    temperature: float = 0.0
    max_tokens: int = 4096
    custom_system_prompt: str | None = None

    def __post_init__(self) -> None:
        if (
            self.consensus_mode == ConsensusMode.MULTI
            and not self.secondary_models
        ):
            raise ValueError(
                "secondary_models required when consensus_mode is MULTI"
            )
