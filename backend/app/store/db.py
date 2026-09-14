"""Persistencia en SQLite.

Jerarquia de identificadores, que es la misma que se replica en MLflow:

    session_id        -> una sesion de trabajo (N conversaciones)
      conversation_id -> un hilo con memoria (N interacciones)
        interaction_id-> un mensaje del usuario + todo el bucle del agente
          tool_event  -> cada llamada concreta al servidor MCP

Las evaluaciones reutilizan esa jerarquia: `eval_runs` apunta a su sesion y
cada fila de `eval_results` a la conversacion donde se ejecuto el caso.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

import aiosqlite

from ..config import settings

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sessions (
    id                TEXT PRIMARY KEY,
    title             TEXT NOT NULL DEFAULT '',
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL,
    metadata          TEXT NOT NULL DEFAULT '{}',
    -- Experimento de MLflow donde se registran los runs de esta sesion.
    -- Vacio => el de la configuracion del backend.
    mlflow_experiment TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS conversations (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    title        TEXT NOT NULL DEFAULT '',
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    provider     TEXT NOT NULL DEFAULT '',
    model        TEXT NOT NULL DEFAULT '',
    mcp_servers  TEXT NOT NULL DEFAULT '[]',
    system_prompt TEXT NOT NULL DEFAULT '',
    mlflow_run_id TEXT NOT NULL DEFAULT '',
    metadata     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    interaction_id  TEXT NOT NULL DEFAULT '',
    seq             INTEGER NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL DEFAULT '',
    thinking        TEXT NOT NULL DEFAULT '',
    tool_calls      TEXT NOT NULL DEFAULT '[]',
    tool_call_id    TEXT NOT NULL DEFAULT '',
    name            TEXT NOT NULL DEFAULT '',
    usage           TEXT NOT NULL DEFAULT '{}',
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, seq);

CREATE TABLE IF NOT EXISTS interactions (
    id                TEXT PRIMARY KEY,
    conversation_id   TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    session_id        TEXT NOT NULL,
    user_message      TEXT NOT NULL DEFAULT '',
    final_answer      TEXT NOT NULL DEFAULT '',
    status            TEXT NOT NULL DEFAULT 'running',
    error             TEXT NOT NULL DEFAULT '',
    started_at        REAL NOT NULL,
    ended_at          REAL,
    latency_ms        REAL NOT NULL DEFAULT 0,
    llm_latency_ms    REAL NOT NULL DEFAULT 0,
    mcp_latency_ms    REAL NOT NULL DEFAULT 0,
    llm_calls         INTEGER NOT NULL DEFAULT 0,
    tool_calls        INTEGER NOT NULL DEFAULT 0,
    tool_errors       INTEGER NOT NULL DEFAULT 0,
    iterations        INTEGER NOT NULL DEFAULT 0,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens      INTEGER NOT NULL DEFAULT 0,
    provider          TEXT NOT NULL DEFAULT '',
    model             TEXT NOT NULL DEFAULT '',
    mcp_servers       TEXT NOT NULL DEFAULT '[]',
    mlflow_trace_id   TEXT NOT NULL DEFAULT '',
    mlflow_run_id     TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_inter_conv ON interactions(conversation_id, started_at);
CREATE INDEX IF NOT EXISTS idx_inter_session ON interactions(session_id, started_at);

CREATE TABLE IF NOT EXISTS tool_events (
    id              TEXT PRIMARY KEY,
    interaction_id  TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    iteration       INTEGER NOT NULL DEFAULT 0,
    tool_name       TEXT NOT NULL,
    real_tool_name  TEXT NOT NULL DEFAULT '',
    server_name     TEXT NOT NULL DEFAULT '',
    server_url      TEXT NOT NULL DEFAULT '',
    conn_id         TEXT NOT NULL DEFAULT '',
    arguments       TEXT NOT NULL DEFAULT '{}',
    result_text     TEXT NOT NULL DEFAULT '',
    structured      TEXT NOT NULL DEFAULT 'null',
    ok              INTEGER NOT NULL DEFAULT 1,
    error           TEXT NOT NULL DEFAULT '',
    latency_ms      REAL NOT NULL DEFAULT 0,
    frames          TEXT NOT NULL DEFAULT '[]',
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tool_inter ON tool_events(interaction_id, seq);
CREATE INDEX IF NOT EXISTS idx_tool_session ON tool_events(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_tool_name ON tool_events(tool_name);

-- Ejecucion de una bateria de evaluacion. Vive dentro de su propia sesion
-- (session_id), y cada caso x persona es una conversacion de esa sesion.
CREATE TABLE IF NOT EXISTS eval_runs (
    id                TEXT PRIMARY KEY,
    session_id        TEXT NOT NULL,
    name              TEXT NOT NULL DEFAULT '',
    suite             TEXT NOT NULL DEFAULT '',
    status            TEXT NOT NULL DEFAULT 'running',
    created_at        REAL NOT NULL,
    ended_at          REAL,
    -- Modelos, MCP y opciones. Nunca incluye claves de API.
    config            TEXT NOT NULL DEFAULT '{}',
    personas_yaml     TEXT NOT NULL DEFAULT '',
    cases_yaml        TEXT NOT NULL DEFAULT '',
    summary           TEXT NOT NULL DEFAULT '{}',
    error             TEXT NOT NULL DEFAULT '',
    mlflow_experiment TEXT NOT NULL DEFAULT '',
    mlflow_run_id     TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_evalrun_created ON eval_runs(created_at);

CREATE TABLE IF NOT EXISTS eval_results (
    id              TEXT PRIMARY KEY,
    eval_run_id     TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    case_id         TEXT NOT NULL,
    case_title      TEXT NOT NULL DEFAULT '',
    persona_id      TEXT NOT NULL,
    persona_name    TEXT NOT NULL DEFAULT '',
    repetition      INTEGER NOT NULL DEFAULT 1,
    conversation_id TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'pending',
    score           REAL,
    passed          INTEGER,
    turns           INTEGER NOT NULL DEFAULT 0,
    end_reason      TEXT NOT NULL DEFAULT '',
    items           TEXT NOT NULL DEFAULT '[]',
    verdict         TEXT NOT NULL DEFAULT '{}',
    transcript      TEXT NOT NULL DEFAULT '[]',
    metrics         TEXT NOT NULL DEFAULT '{}',
    error           TEXT NOT NULL DEFAULT '',
    started_at      REAL,
    ended_at        REAL,
    mlflow_run_id   TEXT NOT NULL DEFAULT '',
    mlflow_trace_ids TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_evalres_run ON eval_results(eval_run_id, seq);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(value), ensure_ascii=False)


def loads(value: str | None, fallback: Any = None) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


# Columnas anadidas despues de la primera version. SQLite no tiene
# "ADD COLUMN IF NOT EXISTS", asi que se comprueba con PRAGMA table_info.
MIGRATIONS: list[tuple[str, str, str]] = [
    ("sessions", "mlflow_experiment", "TEXT NOT NULL DEFAULT ''"),
]


class Database:
    """Wrapper fino sobre aiosqlite con una unica conexion serializada."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(SCHEMA)
        await self._migrate()
        await self._conn.commit()

    async def _migrate(self) -> None:
        """Anade columnas nuevas a bases creadas con versiones anteriores."""
        assert self._conn is not None
        for table, column, ddl in MIGRATIONS:
            cursor = await self._conn.execute(f"PRAGMA table_info({table})")
            columns = {row[1] for row in await cursor.fetchall()}
            await cursor.close()
            if column not in columns:
                await self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("La base de datos no esta inicializada")
        return self._conn

    async def execute(self, sql: str, params: tuple = ()) -> None:
        async with self._lock:
            await self.conn.execute(sql, params)
            await self.conn.commit()

    async def fetch_all(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        async with self._lock:
            cursor = await self.conn.execute(sql, params)
            rows = await cursor.fetchall()
            await cursor.close()
        return [dict(r) for r in rows]

    async def fetch_one(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        rows = await self.fetch_all(sql, params)
        return rows[0] if rows else None


db = Database(settings.db_path)


def now() -> float:
    return time.time()
