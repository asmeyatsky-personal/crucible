"""
OpenAI Judge Adapter (JUDGE Module)

Architectural Intent:
- Infrastructure adapter implementing JudgePort via OpenAI GPT-4o API
- Alternative judge for cross-model consensus evaluation
- Same interface contract as ClaudeJudgeAdapter — ports make them interchangeable
"""

from __future__ import annotations

import json

import openai

from domain.entities.trace import Trace
from domain.value_objects.judge_config import JudgeConfig
from domain.value_objects.rubric_dimension import RubricDimension
from domain.value_objects.score import Confidence, DimensionScore


class OpenAIJudgeAdapter:
    """Implements JudgePort using OpenAI GPT-4o as the judge model."""

    def __init__(self, api_key: str):
        self._client = openai.AsyncOpenAI(api_key=api_key)

    async def evaluate_dimension(
        self,
        trace: Trace,
        dimension: RubricDimension,
        judge_config: JudgeConfig,
    ) -> DimensionScore:
        system_prompt = judge_config.custom_system_prompt or self._default_system_prompt()
        user_prompt = self._build_evaluation_prompt(trace, dimension)

        response = await self._client.chat.completions.create(
            model="gpt-4o",
            temperature=judge_config.temperature,
            max_tokens=judge_config.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return self._parse_response(response, dimension)

    def _default_system_prompt(self) -> str:
        return """You are an expert AI agent evaluator for the CRUCIBLE evaluation harness.
Your role is to evaluate a specific dimension of an agent's execution trajectory.

You must respond with valid JSON containing exactly these fields:
- "score": a float between 0.0 and 1.0
- "passed": a boolean indicating if the score meets the threshold
- "rationale": a concise explanation of your assessment (2-3 sentences)
- "confidence": one of "high", "medium", or "low"

Be rigorous but fair. Focus solely on the dimension being evaluated."""

    def _build_evaluation_prompt(
        self, trace: Trace, dimension: RubricDimension,
    ) -> str:
        trajectory_text = self._format_trajectory(trace)
        return f"""## Evaluation Dimension
**Name**: {dimension.name}
**Description**: {dimension.description}
**Pass Threshold**: {dimension.pass_threshold}

## Specific Evaluation Criteria
{dimension.judge_prompt}

## Agent Trajectory
{trajectory_text}

## Instructions
Evaluate the agent's trajectory ONLY on the dimension described above.
Respond with a JSON object containing: score, passed, rationale, confidence."""

    def _format_trajectory(self, trace: Trace) -> str:
        parts = [
            f"Agent: {trace.agent_id} | Model: {trace.model_id} | "
            f"Steps: {trace.step_count} | Tokens: {trace.total_tokens}",
            "",
        ]
        for step in trace.steps:
            parts.append(f"### Step {step.step_index}: {step.step_type.value}")
            parts.append(step.content[:2000])
            if step.tool_calls:
                for tc in step.tool_calls:
                    status = "OK" if tc.success else f"FAILED: {tc.error}"
                    parts.append(
                        f"  Tool: {tc.name}({json.dumps(tc.input_parameters)[:500]}) "
                        f"-> {status} [{tc.latency_ms:.0f}ms]"
                    )
            parts.append("")
        return "\n".join(parts)

    def _parse_response(
        self, response: openai.types.chat.ChatCompletion, dimension: RubricDimension,
    ) -> DimensionScore:
        text = response.choices[0].message.content or "{}"
        try:
            data = json.loads(text.strip())
        except json.JSONDecodeError:
            return DimensionScore(
                dimension_name=dimension.name,
                score=0.5,
                passed=False,
                rationale=f"Judge response could not be parsed: {text[:200]}",
                confidence=Confidence.LOW,
            )

        score = max(0.0, min(1.0, float(data.get("score", 0.5))))
        return DimensionScore(
            dimension_name=dimension.name,
            score=score,
            passed=score >= dimension.pass_threshold,
            rationale=str(data.get("rationale", "No rationale provided")),
            confidence=Confidence(data.get("confidence", "medium")),
        )
