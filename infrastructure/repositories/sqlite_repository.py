"""
SQLite Repository Implementations

Architectural Intent:
- Infrastructure adapters implementing domain repository ports
- Uses aiosqlite for async SQLite operations
- Handles serialisation/deserialisation between domain objects and relational storage
- Single file for all repositories to share connection management

MCP Integration:
- These repositories are consumed by use cases, which are exposed via MCP server
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import aiosqlite

from domain.entities.agent import Agent
from domain.entities.evaluation import Evaluation
from domain.entities.rubric import Rubric
from domain.entities.trace import Trace
from domain.value_objects.judge_config import ConsensusMode, JudgeConfig, JudgeModel
from domain.value_objects.rubric_dimension import RubricDimension, ScoringMethod
from domain.value_objects.score import CompositeScore, Confidence, DimensionScore
from domain.value_objects.tool_call import ToolCall
from domain.value_objects.trajectory_step import StepType, TrajectoryStep


class SQLiteDatabase:
    """Manages the SQLite connection and schema initialisation."""

    def __init__(self, db_path: str = "crucible.db"):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._create_tables()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._db

    async def _create_tables(self) -> None:
        await self.db.executescript("""
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL,
                agent_type TEXT NOT NULL,
                rubric_ids TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS traces (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                temperature REAL NOT NULL,
                total_tokens INTEGER NOT NULL,
                total_latency_ms REAL NOT NULL,
                captured_at TEXT NOT NULL,
                steps_json TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (agent_id) REFERENCES agents(id)
            );

            CREATE TABLE IF NOT EXISTS rubrics (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL,
                dimensions_json TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                agent_type TEXT,
                owner TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evaluations (
                id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL,
                rubric_id TEXT NOT NULL,
                rubric_version INTEGER NOT NULL,
                agent_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                judge_config_json TEXT NOT NULL,
                composite_score_json TEXT NOT NULL,
                dimension_scores_json TEXT NOT NULL,
                judge_model_id TEXT NOT NULL,
                evaluated_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (trace_id) REFERENCES traces(id),
                FOREIGN KEY (rubric_id) REFERENCES rubrics(id)
            );

            CREATE INDEX IF NOT EXISTS idx_traces_agent_id ON traces(agent_id);
            CREATE INDEX IF NOT EXISTS idx_traces_run_id ON traces(run_id);
            CREATE INDEX IF NOT EXISTS idx_evaluations_agent_id ON evaluations(agent_id);
            CREATE INDEX IF NOT EXISTS idx_evaluations_rubric_id ON evaluations(rubric_id);
            CREATE INDEX IF NOT EXISTS idx_evaluations_evaluated_at ON evaluations(evaluated_at);
        """)
        await self.db.commit()


class SQLiteAgentRepository:
    """Implements AgentRepositoryPort using SQLite."""

    def __init__(self, database: SQLiteDatabase):
        self._db = database

    async def save(self, agent: Agent) -> None:
        await self._db.db.execute(
            """INSERT OR REPLACE INTO agents
               (id, name, description, agent_type, rubric_ids, created_at, updated_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                agent.id, agent.name, agent.description, agent.agent_type,
                json.dumps(list(agent.rubric_ids)),
                agent.created_at.isoformat(),
                agent.updated_at.isoformat(),
                json.dumps(agent.metadata),
            ),
        )
        await self._db.db.commit()

    async def get_by_id(self, agent_id: str) -> Agent | None:
        cursor = await self._db.db.execute(
            "SELECT * FROM agents WHERE id = ?", (agent_id,),
        )
        row = await cursor.fetchone()
        return self._to_agent(row) if row else None

    async def get_by_name(self, name: str) -> Agent | None:
        cursor = await self._db.db.execute(
            "SELECT * FROM agents WHERE name = ?", (name,),
        )
        row = await cursor.fetchone()
        return self._to_agent(row) if row else None

    async def list_all(self) -> list[Agent]:
        cursor = await self._db.db.execute("SELECT * FROM agents ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [self._to_agent(row) for row in rows]

    @staticmethod
    def _to_agent(row: aiosqlite.Row) -> Agent:
        return Agent(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            agent_type=row["agent_type"],
            rubric_ids=tuple(json.loads(row["rubric_ids"])),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            metadata=json.loads(row["metadata"]),
        )


class SQLiteTraceRepository:
    """Implements TraceRepositoryPort using SQLite."""

    def __init__(self, database: SQLiteDatabase):
        self._db = database

    async def save(self, trace: Trace) -> None:
        steps_json = json.dumps([_serialize_step(s) for s in trace.steps])
        await self._db.db.execute(
            """INSERT OR REPLACE INTO traces
               (id, agent_id, run_id, model_id, temperature, total_tokens,
                total_latency_ms, captured_at, steps_json, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trace.id, trace.agent_id, trace.run_id, trace.model_id,
                trace.temperature, trace.total_tokens, trace.total_latency_ms,
                trace.captured_at.isoformat(), steps_json,
                json.dumps(trace.metadata),
            ),
        )
        await self._db.db.commit()

    async def get_by_id(self, trace_id: str) -> Trace | None:
        cursor = await self._db.db.execute(
            "SELECT * FROM traces WHERE id = ?", (trace_id,),
        )
        row = await cursor.fetchone()
        return self._to_trace(row) if row else None

    async def get_by_run_id(self, run_id: str) -> Trace | None:
        cursor = await self._db.db.execute(
            "SELECT * FROM traces WHERE run_id = ?", (run_id,),
        )
        row = await cursor.fetchone()
        return self._to_trace(row) if row else None

    async def list_by_agent(self, agent_id: str, limit: int = 50) -> list[Trace]:
        cursor = await self._db.db.execute(
            "SELECT * FROM traces WHERE agent_id = ? ORDER BY captured_at DESC LIMIT ?",
            (agent_id, limit),
        )
        rows = await cursor.fetchall()
        return [self._to_trace(row) for row in rows]

    @staticmethod
    def _to_trace(row: aiosqlite.Row) -> Trace:
        steps_data = json.loads(row["steps_json"])
        steps = tuple(_deserialize_step(s) for s in steps_data)
        return Trace(
            id=row["id"],
            agent_id=row["agent_id"],
            run_id=row["run_id"],
            steps=steps,
            model_id=row["model_id"],
            temperature=row["temperature"],
            total_tokens=row["total_tokens"],
            total_latency_ms=row["total_latency_ms"],
            captured_at=datetime.fromisoformat(row["captured_at"]),
            metadata=json.loads(row["metadata"]),
        )


class SQLiteRubricRepository:
    """Implements RubricRepositoryPort using SQLite."""

    def __init__(self, database: SQLiteDatabase):
        self._db = database

    async def save(self, rubric: Rubric) -> None:
        dims_json = json.dumps([_serialize_dimension(d) for d in rubric.dimensions])
        await self._db.db.execute(
            """INSERT OR REPLACE INTO rubrics
               (id, name, description, dimensions_json, version, agent_type, owner,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rubric.id, rubric.name, rubric.description, dims_json,
                rubric.version, rubric.agent_type, rubric.owner,
                rubric.created_at.isoformat(), rubric.updated_at.isoformat(),
            ),
        )
        await self._db.db.commit()

    async def get_by_id(self, rubric_id: str) -> Rubric | None:
        cursor = await self._db.db.execute(
            "SELECT * FROM rubrics WHERE id = ?", (rubric_id,),
        )
        row = await cursor.fetchone()
        return self._to_rubric(row) if row else None

    async def get_by_name(self, name: str) -> Rubric | None:
        cursor = await self._db.db.execute(
            "SELECT * FROM rubrics WHERE name = ?", (name,),
        )
        row = await cursor.fetchone()
        return self._to_rubric(row) if row else None

    async def list_all(self) -> list[Rubric]:
        cursor = await self._db.db.execute(
            "SELECT * FROM rubrics ORDER BY created_at DESC",
        )
        rows = await cursor.fetchall()
        return [self._to_rubric(row) for row in rows]

    async def list_by_agent_type(self, agent_type: str) -> list[Rubric]:
        cursor = await self._db.db.execute(
            "SELECT * FROM rubrics WHERE agent_type = ? ORDER BY created_at DESC",
            (agent_type,),
        )
        rows = await cursor.fetchall()
        return [self._to_rubric(row) for row in rows]

    @staticmethod
    def _to_rubric(row: aiosqlite.Row) -> Rubric:
        dims_data = json.loads(row["dimensions_json"])
        dimensions = tuple(_deserialize_dimension(d) for d in dims_data)
        return Rubric(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            dimensions=dimensions,
            version=row["version"],
            agent_type=row["agent_type"],
            owner=row["owner"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )


class SQLiteEvaluationRepository:
    """Implements EvaluationRepositoryPort using SQLite."""

    def __init__(self, database: SQLiteDatabase):
        self._db = database

    async def save(self, evaluation: Evaluation) -> None:
        judge_config_json = json.dumps(_serialize_judge_config(evaluation.judge_config))
        composite_json = json.dumps(_serialize_composite_score(evaluation.composite_score))
        dims_json = json.dumps([_serialize_dim_score(ds) for ds in evaluation.dimension_scores])
        await self._db.db.execute(
            """INSERT OR REPLACE INTO evaluations
               (id, trace_id, rubric_id, rubric_version, agent_id, run_id,
                judge_config_json, composite_score_json, dimension_scores_json,
                judge_model_id, evaluated_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                evaluation.id, evaluation.trace_id, evaluation.rubric_id,
                evaluation.rubric_version, evaluation.agent_id, evaluation.run_id,
                judge_config_json, composite_json, dims_json,
                evaluation.judge_model_id, evaluation.evaluated_at.isoformat(),
                json.dumps(evaluation.metadata),
            ),
        )
        await self._db.db.commit()

    async def get_by_id(self, evaluation_id: str) -> Evaluation | None:
        cursor = await self._db.db.execute(
            "SELECT * FROM evaluations WHERE id = ?", (evaluation_id,),
        )
        row = await cursor.fetchone()
        return self._to_evaluation(row) if row else None

    async def list_by_agent(
        self, agent_id: str, limit: int = 50,
    ) -> list[Evaluation]:
        cursor = await self._db.db.execute(
            "SELECT * FROM evaluations WHERE agent_id = ? ORDER BY evaluated_at DESC LIMIT ?",
            (agent_id, limit),
        )
        rows = await cursor.fetchall()
        return [self._to_evaluation(row) for row in rows]

    async def list_by_rubric(
        self, rubric_id: str, limit: int = 50,
    ) -> list[Evaluation]:
        cursor = await self._db.db.execute(
            "SELECT * FROM evaluations WHERE rubric_id = ? ORDER BY evaluated_at DESC LIMIT ?",
            (rubric_id, limit),
        )
        rows = await cursor.fetchall()
        return [self._to_evaluation(row) for row in rows]

    async def get_latest_by_agent(self, agent_id: str) -> Evaluation | None:
        cursor = await self._db.db.execute(
            "SELECT * FROM evaluations WHERE agent_id = ? ORDER BY evaluated_at DESC LIMIT 1",
            (agent_id,),
        )
        row = await cursor.fetchone()
        return self._to_evaluation(row) if row else None

    @staticmethod
    def _to_evaluation(row: aiosqlite.Row) -> Evaluation:
        judge_config_data = json.loads(row["judge_config_json"])
        composite_data = json.loads(row["composite_score_json"])
        dims_data = json.loads(row["dimension_scores_json"])

        judge_config = JudgeConfig(
            primary_model=JudgeModel(judge_config_data["primary_model"]),
            consensus_mode=ConsensusMode(judge_config_data["consensus_mode"]),
            secondary_models=tuple(
                JudgeModel(m) for m in judge_config_data.get("secondary_models", [])
            ),
            temperature=judge_config_data.get("temperature", 0.0),
            max_tokens=judge_config_data.get("max_tokens", 4096),
            custom_system_prompt=judge_config_data.get("custom_system_prompt"),
        )

        dimension_scores = tuple(
            DimensionScore(
                dimension_name=ds["dimension_name"],
                score=ds["score"],
                passed=ds["passed"],
                rationale=ds["rationale"],
                confidence=Confidence(ds["confidence"]),
            )
            for ds in dims_data
        )

        composite_score = CompositeScore(
            value=composite_data["value"],
            dimension_scores=dimension_scores,
            passed=composite_data["passed"],
            dimensions_passed=composite_data["dimensions_passed"],
            dimensions_total=composite_data["dimensions_total"],
        )

        return Evaluation(
            id=row["id"],
            trace_id=row["trace_id"],
            rubric_id=row["rubric_id"],
            rubric_version=row["rubric_version"],
            agent_id=row["agent_id"],
            run_id=row["run_id"],
            judge_config=judge_config,
            composite_score=composite_score,
            dimension_scores=dimension_scores,
            judge_model_id=row["judge_model_id"],
            evaluated_at=datetime.fromisoformat(row["evaluated_at"]),
            metadata=json.loads(row["metadata"]),
        )


# --- Serialisation Helpers ---

def _serialize_step(step: TrajectoryStep) -> dict:
    return {
        "step_index": step.step_index,
        "step_type": step.step_type.value,
        "content": step.content,
        "timestamp": step.timestamp.isoformat(),
        "tool_calls": [
            {
                "name": tc.name,
                "input_parameters": tc.input_parameters,
                "output": tc.output,
                "latency_ms": tc.latency_ms,
                "success": tc.success,
                "timestamp": tc.timestamp.isoformat(),
                "error": tc.error,
            }
            for tc in step.tool_calls
        ],
        "token_count": step.token_count,
        "model_id": step.model_id,
        "metadata": step.metadata,
    }


def _deserialize_step(data: dict) -> TrajectoryStep:
    return TrajectoryStep(
        step_index=data["step_index"],
        step_type=StepType(data["step_type"]),
        content=data["content"],
        timestamp=datetime.fromisoformat(data["timestamp"]),
        tool_calls=tuple(
            ToolCall(
                name=tc["name"],
                input_parameters=tc["input_parameters"],
                output=tc.get("output"),
                latency_ms=tc["latency_ms"],
                success=tc["success"],
                timestamp=datetime.fromisoformat(tc["timestamp"]),
                error=tc.get("error"),
            )
            for tc in data.get("tool_calls", [])
        ),
        token_count=data.get("token_count"),
        model_id=data.get("model_id"),
        metadata=data.get("metadata", {}),
    )


def _serialize_dimension(dim: RubricDimension) -> dict:
    return {
        "name": dim.name,
        "description": dim.description,
        "scoring_method": dim.scoring_method.value,
        "weight": dim.weight,
        "pass_threshold": dim.pass_threshold,
        "judge_prompt": dim.judge_prompt,
        "expected_value": dim.expected_value,
    }


def _deserialize_dimension(data: dict) -> RubricDimension:
    return RubricDimension(
        name=data["name"],
        description=data["description"],
        scoring_method=ScoringMethod(data["scoring_method"]),
        weight=data["weight"],
        pass_threshold=data["pass_threshold"],
        judge_prompt=data.get("judge_prompt"),
        expected_value=data.get("expected_value"),
    )


def _serialize_judge_config(config: JudgeConfig) -> dict:
    return {
        "primary_model": config.primary_model.value,
        "consensus_mode": config.consensus_mode.value,
        "secondary_models": [m.value for m in config.secondary_models],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "custom_system_prompt": config.custom_system_prompt,
    }


def _serialize_composite_score(score: CompositeScore) -> dict:
    return {
        "value": score.value,
        "passed": score.passed,
        "dimensions_passed": score.dimensions_passed,
        "dimensions_total": score.dimensions_total,
    }


def _serialize_dim_score(ds: DimensionScore) -> dict:
    return {
        "dimension_name": ds.dimension_name,
        "score": ds.score,
        "passed": ds.passed,
        "rationale": ds.rationale,
        "confidence": ds.confidence.value,
    }
