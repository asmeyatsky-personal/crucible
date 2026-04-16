"""
CRUCIBLE™ Python SDK

Usage:
    from crucible_sdk import CrucibleClient

    client = CrucibleClient(base_url="http://localhost:8100")

    # Register an agent
    agent = await client.register_agent("my-agent", "Research assistant", "research")

    # Capture a trace
    trace = await client.trace(agent_id=agent["id"], run_id="run-1", steps=[...], model_id="claude-sonnet-4-6")

    # Evaluate
    evaluation = await client.evaluate(trace_id=trace["id"], rubric_id="rubric-1")

    # Get scores
    scores = await client.scores(agent_id=agent["id"])
"""

from crucible_sdk.client import CrucibleClient

__all__ = ["CrucibleClient"]
__version__ = "1.0.0"
