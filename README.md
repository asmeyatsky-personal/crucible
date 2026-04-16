# CRUCIBLE™

**Agentic Evaluation Harness — Heat-test your agents. Ship what survives.**

CRUCIBLE™ is a standalone evaluation harness for agentic AI systems. It provides infrastructure to define what "correct" looks like across a full agent trajectory, score it automatically using LLM-as-judge evaluation, track quality over time, and export evidence for governance and compliance.

> A crucible is a vessel in which materials are subjected to high heat to test their purity and composition. Agents enter CRUCIBLE™. Heat is applied. Only what is sound comes out.

## Phase 1 Modules

| Module | Purpose |
|--------|---------|
| **TRACE** | Captures full agent execution trajectories as immutable, versioned artefacts |
| **RUBRIC** | Defines and versions evaluation criteria as structured, YAML-based rubric sets |
| **JUDGE** | Orchestrates LLM-as-judge evaluation (Claude Sonnet, GPT-4o, multi-model consensus) |
| **SCORE** | Computes weighted composite scores with regression detection |
| **REPORT** | Generates evidence reports (JSON, CSV) for dashboards and governance |

## Architecture

Built on clean/hexagonal architecture principles with domain-driven design:

```
crucible/
├── domain/           # Entities, value objects, events, ports — zero infrastructure deps
├── application/      # Use cases, queries, DTOs, DAG orchestrator
├── infrastructure/   # SQLite repos, judge adapters, MCP server, DI container
├── presentation/     # FastAPI REST API, Click CLI
├── sdk/              # Python SDK client
├── tests/            # Domain, application, infrastructure, integration tests
└── rubric_templates/ # Pre-built rubrics for common agent archetypes
```

Key patterns:
- **Immutable domain models** — all entities are frozen dataclasses
- **Interface-first** — Protocol ports in domain, adapters in infrastructure
- **MCP-native** — full MCP server with tools (write) and resources (read)
- **Parallelism-first** — rubric dimensions evaluated concurrently, DAG-based workflow orchestration
- **Framework-agnostic** — works with any agent orchestration (LangChain, CrewAI, custom)

## Quick Start

### Install

```bash
pip install -e ".[dev]"
```

### Start the API Server

```bash
crucible serve --port 8100
```

### Register an Agent

```bash
crucible register --name "my-research-agent" --description "Research assistant" --type research
```

### Load a Rubric

```bash
crucible load-rubric rubric_templates/research_agent.yaml
```

### Using the Python SDK

```python
from crucible_sdk import CrucibleClient

async with CrucibleClient(base_url="http://localhost:8100") as client:
    # Register an agent
    agent = await client.register_agent(
        "my-agent", "Research assistant", "research"
    )

    # Capture a trace
    trace = await client.trace(
        agent_id=agent["id"],
        run_id="run-001",
        steps=[
            {"step_index": 0, "step_type": "user_message",
             "content": "Research Python history", "timestamp": "2025-01-01T00:00:00Z"},
            {"step_index": 1, "step_type": "assistant_response",
             "content": "Python was created by Guido van Rossum...", "timestamp": "2025-01-01T00:00:01Z"},
        ],
        model_id="claude-sonnet-4-6",
    )

    # Evaluate against a rubric
    evaluation = await client.evaluate(
        trace_id=trace["id"],
        rubric_id="<rubric-id>",
    )

    # Check scores and regressions
    scores = await client.scores(agent_id=agent["id"])
```

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/agents` | Register an agent |
| `GET` | `/api/v1/agents` | List all agents |
| `POST` | `/api/v1/traces` | Capture a trace |
| `GET` | `/api/v1/traces/{id}` | Get trace details |
| `POST` | `/api/v1/rubrics` | Create a rubric |
| `GET` | `/api/v1/rubrics` | List rubrics |
| `POST` | `/api/v1/evaluations` | Evaluate a run |
| `GET` | `/api/v1/evaluations/{id}` | Get evaluation details |
| `GET` | `/api/v1/evaluations/agent/{id}/scores` | Score timeline |
| `POST` | `/api/v1/reports` | Generate a report |
| `GET` | `/health` | Health check |

## MCP Server

CRUCIBLE exposes an MCP server (`crucible-service`) for agent-native integration:

**Tools** (write operations):
- `register_agent` — register an agent identity
- `capture_trace` — capture an execution trajectory
- `create_rubric` — create evaluation criteria
- `evaluate_run` — evaluate a trace against a rubric
- `generate_report` — generate an evidence report

Add to your MCP configuration:
```json
{
  "mcpServers": {
    "crucible": {
      "command": "python",
      "args": ["-m", "infrastructure.mcp_servers.crucible_server"],
      "env": { "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}" }
    }
  }
}
```

## Rubric Templates

Pre-built evaluation rubrics for common agent archetypes:

| Template | Dimensions |
|----------|-----------|
| `research_agent.yaml` | Goal adherence, source quality, tool appropriateness, reasoning coherence, output quality, scope compliance |
| `coding_agent.yaml` | Goal adherence, code correctness, tool sequencing, code quality, safety compliance, output quality |
| `customer_support_agent.yaml` | Goal adherence, tone/empathy, accuracy, tool appropriateness, scope compliance, resolution efficiency |
| `data_extraction_agent.yaml` | Extraction accuracy, schema compliance, tool appropriateness, scope compliance, reasoning coherence |

## Running Tests

```bash
python -m pytest tests/ -v
```

48 tests covering domain logic (no mocks), application use cases (mocked ports), MCP schema compliance, and full integration flows.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DATABASE_URL` | `sqlite+aiosqlite:///crucible.db` | Database connection |
| `STORAGE_PATH` | `./storage` | Artefact blob storage path |
| `ANTHROPIC_API_KEY` | — | Anthropic API key for Claude judge |
| `OPENAI_API_KEY` | — | OpenAI API key for GPT-4o judge |
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8100` | API server port |

## Roadmap

- **Phase 1** (current) — Evaluation harness: TRACE, RUBRIC, JUDGE, SCORE, REPORT
- **Phase 2** — Replay and adversarial: deterministic trajectory replay, adversarial injection engine
- **Phase 3** — Portfolio integration: VAID binding, CODEX™ evidence export, SENTINEL™ baseline integration

## Tech Stack

- **Backend**: Python 3.11+ (FastAPI, aiosqlite, Anthropic SDK, OpenAI SDK)
- **Architecture**: Clean/hexagonal, DDD, MCP-native
- **Testing**: pytest, pytest-asyncio
- **Storage**: SQLite (dev), Cloud Spanner (prod)

---

*CRUCIBLE™ — Smeyatsky Labs Ltd — Proprietary*
