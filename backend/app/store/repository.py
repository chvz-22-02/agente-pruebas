"""Operaciones de alto nivel sobre la base de datos."""

from __future__ import annotations

import time
from typing import Any

from .db import db, dumps, loads, new_id, now


def default_session_title() -> str:
    """Nombre por defecto de una sesion: corto y ordenable.

    Solo es una etiqueta: quien identifica la sesion es su id, tanto en SQLite
    como en MLflow (`tags.session_id`), asi que dos sesiones pueden compartir
    nombre sin ningun conflicto. Se puede cambiar desde la UI.
    """
    return time.strftime("%d/%m %H:%M")


# --------------------------------------------------------------- sesiones ---
async def create_session(
    title: str = "", metadata: dict | None = None, mlflow_experiment: str = ""
) -> dict[str, Any]:
    session_id = new_id("ses")
    ts = now()
    await db.execute(
        """INSERT INTO sessions (id, title, created_at, updated_at, metadata, mlflow_experiment)
           VALUES (?,?,?,?,?,?)""",
        (session_id, title or default_session_title(), ts, ts, dumps(metadata or {}), mlflow_experiment),
    )
    return await get_session(session_id)  # type: ignore[return-value]


async def get_session(session_id: str) -> dict[str, Any] | None:
    row = await db.fetch_one("SELECT * FROM sessions WHERE id = ?", (session_id,))
    if row:
        row["metadata"] = loads(row["metadata"], {})
    return row


async def ensure_session(
    session_id: str | None, title: str = "", mlflow_experiment: str = ""
) -> str:
    """Devuelve una sesion utilizable, creandola solo si de verdad hace falta.

    Las sesiones se crean *bajo demanda* (al mandar el primer mensaje o al
    pulsar "+ Sesion"), nunca de forma periodica: asi no se acumulan sesiones
    vacias en la barra lateral ni runs huerfanos en MLflow.
    """
    if session_id and await get_session(session_id):
        return session_id
    if session_id:
        ts = now()
        await db.execute(
            """INSERT INTO sessions (id, title, created_at, updated_at, metadata, mlflow_experiment)
               VALUES (?,?,?,?,?,?)""",
            (session_id, title or default_session_title(), ts, ts, "{}", mlflow_experiment),
        )
        return session_id
    return (await create_session(title, mlflow_experiment=mlflow_experiment))["id"]


async def update_session(session_id: str, **fields: Any) -> None:
    allowed = {"title", "mlflow_experiment", "metadata"}
    sets, params = [], []
    for key, value in fields.items():
        if key not in allowed:
            continue
        sets.append(f"{key} = ?")
        params.append(dumps(value) if key == "metadata" else value)
    if not sets:
        return
    sets.append("updated_at = ?")
    params.extend([now(), session_id])
    await db.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE id = ?", tuple(params))


async def delete_session(session_id: str) -> None:
    """Borra la sesion y todo lo que cuelga de ella.

    `tool_events` no tiene clave foranea (se consulta por sesion sin pasar por
    la conversacion), y los borrados en cascada dependen de que el PRAGMA este
    activo, asi que se limpia tabla por tabla de forma explicita.
    """
    await db.execute(
        "DELETE FROM messages WHERE conversation_id IN "
        "(SELECT id FROM conversations WHERE session_id = ?)",
        (session_id,),
    )
    await db.execute("DELETE FROM tool_events WHERE session_id = ?", (session_id,))
    await db.execute("DELETE FROM interactions WHERE session_id = ?", (session_id,))
    await db.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
    await db.execute(
        "DELETE FROM eval_results WHERE eval_run_id IN (SELECT id FROM eval_runs WHERE session_id = ?)",
        (session_id,),
    )
    await db.execute("DELETE FROM eval_runs WHERE session_id = ?", (session_id,))
    await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


async def list_empty_sessions(exclude: str = "") -> list[dict[str, Any]]:
    """Sesiones sin ninguna interaccion registrada.

    Es lo que deja atras un banco de pruebas: hilos que se abrieron y nunca se
    usaron. La sesion abierta se puede excluir para no borrarla en caliente.
    """
    rows = await db.fetch_all(
        """SELECT s.id, s.title FROM sessions s
           WHERE NOT EXISTS (SELECT 1 FROM interactions i WHERE i.session_id = s.id)
             AND s.id != ?
           ORDER BY s.updated_at DESC""",
        (exclude,),
    )
    return rows


async def list_conversation_ids(session_id: str) -> list[str]:
    rows = await db.fetch_all("SELECT id FROM conversations WHERE session_id = ?", (session_id,))
    return [row["id"] for row in rows]


async def list_sessions(limit: int = 100) -> list[dict[str, Any]]:
    rows = await db.fetch_all(
        """
        SELECT s.*,
               (SELECT COUNT(*) FROM conversations c WHERE c.session_id = s.id) AS conversations,
               (SELECT COUNT(*) FROM interactions i WHERE i.session_id = s.id)  AS interactions,
               (SELECT COALESCE(SUM(i.total_tokens),0) FROM interactions i WHERE i.session_id = s.id) AS total_tokens
        FROM sessions s ORDER BY s.updated_at DESC LIMIT ?
        """,
        (limit,),
    )
    for row in rows:
        row["metadata"] = loads(row["metadata"], {})
    return rows


async def touch_session(session_id: str) -> None:
    await db.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now(), session_id))


# ---------------------------------------------------------- conversaciones --
async def create_conversation(
    session_id: str,
    title: str = "",
    provider: str = "",
    model: str = "",
    mcp_servers: list | None = None,
    system_prompt: str = "",
) -> dict[str, Any]:
    conv_id = new_id("conv")
    ts = now()
    await db.execute(
        """INSERT INTO conversations
           (id, session_id, title, created_at, updated_at, provider, model, mcp_servers, system_prompt)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            conv_id,
            session_id,
            title or "Nueva conversacion",
            ts,
            ts,
            provider,
            model,
            dumps(mcp_servers or []),
            system_prompt,
        ),
    )
    await touch_session(session_id)
    return await get_conversation(conv_id)  # type: ignore[return-value]


async def get_conversation(conv_id: str) -> dict[str, Any] | None:
    row = await db.fetch_one("SELECT * FROM conversations WHERE id = ?", (conv_id,))
    if row:
        row["mcp_servers"] = loads(row["mcp_servers"], [])
        row["metadata"] = loads(row["metadata"], {})
    return row


async def list_conversations(session_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    if session_id:
        sql = """SELECT c.*, (SELECT COUNT(*) FROM interactions i WHERE i.conversation_id = c.id) AS interactions
                 FROM conversations c WHERE c.session_id = ? ORDER BY c.updated_at DESC LIMIT ?"""
        params: tuple = (session_id, limit)
    else:
        sql = """SELECT c.*, (SELECT COUNT(*) FROM interactions i WHERE i.conversation_id = c.id) AS interactions
                 FROM conversations c ORDER BY c.updated_at DESC LIMIT ?"""
        params = (limit,)
    rows = await db.fetch_all(sql, params)
    for row in rows:
        row["mcp_servers"] = loads(row["mcp_servers"], [])
        row["metadata"] = loads(row["metadata"], {})
    return rows


async def update_conversation(conv_id: str, **fields: Any) -> None:
    allowed = {"title", "provider", "model", "mcp_servers", "system_prompt", "mlflow_run_id", "metadata"}
    sets, params = [], []
    for key, value in fields.items():
        if key not in allowed:
            continue
        sets.append(f"{key} = ?")
        params.append(dumps(value) if key in {"mcp_servers", "metadata"} else value)
    if not sets:
        return
    sets.append("updated_at = ?")
    params.extend([now(), conv_id])
    await db.execute(f"UPDATE conversations SET {', '.join(sets)} WHERE id = ?", tuple(params))


async def delete_conversation(conv_id: str) -> None:
    await db.execute("DELETE FROM tool_events WHERE conversation_id = ?", (conv_id,))
    await db.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    await db.execute("DELETE FROM interactions WHERE conversation_id = ?", (conv_id,))
    await db.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))


async def clear_conversation_messages(conv_id: str) -> None:
    """Reinicia la memoria conservando la conversacion y su historico de trazas."""
    await db.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))


# ---------------------------------------------------------------- mensajes --
async def next_seq(conv_id: str) -> int:
    row = await db.fetch_one(
        "SELECT COALESCE(MAX(seq), -1) AS m FROM messages WHERE conversation_id = ?", (conv_id,)
    )
    return int(row["m"]) + 1 if row else 0


async def add_message(
    conv_id: str,
    role: str,
    content: str = "",
    *,
    interaction_id: str = "",
    thinking: str = "",
    tool_calls: list | None = None,
    tool_call_id: str = "",
    name: str = "",
    usage: dict | None = None,
) -> dict[str, Any]:
    seq = await next_seq(conv_id)
    msg_id = new_id("msg")
    ts = now()
    await db.execute(
        """INSERT INTO messages
           (id, conversation_id, interaction_id, seq, role, content, thinking,
            tool_calls, tool_call_id, name, usage, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            msg_id,
            conv_id,
            interaction_id,
            seq,
            role,
            content,
            thinking,
            dumps(tool_calls or []),
            tool_call_id,
            name,
            dumps(usage or {}),
            ts,
        ),
    )
    await db.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (ts, conv_id))
    return {"id": msg_id, "seq": seq, "created_at": ts}


async def get_messages(conv_id: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows = await db.fetch_all(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY seq ASC", (conv_id,)
    )
    for row in rows:
        row["tool_calls"] = loads(row["tool_calls"], [])
        row["usage"] = loads(row["usage"], {})
    if limit is not None and len(rows) > limit:
        # Conserva siempre el system inicial al recortar por ventana.
        head = [r for r in rows[:1] if r["role"] == "system"]
        tail = rows[-(limit - len(head)):]
        # Un mensaje 'tool' sin el 'assistant' que lo pidio deja el historial
        # incoherente y las APIs tipo OpenAI lo rechazan: se descartan los
        # huerfanos que hayan quedado al principio del recorte.
        while tail and tail[0]["role"] == "tool":
            tail.pop(0)
        rows = head + tail
    return rows


# ------------------------------------------------------------ interacciones -
async def create_interaction(
    interaction_id: str,
    conv_id: str,
    session_id: str,
    user_message: str,
    provider: str,
    model: str,
    mcp_servers: list,
) -> None:
    await db.execute(
        """INSERT INTO interactions
           (id, conversation_id, session_id, user_message, started_at, provider, model, mcp_servers)
           VALUES (?,?,?,?,?,?,?,?)""",
        (interaction_id, conv_id, session_id, user_message, now(), provider, model, dumps(mcp_servers)),
    )


async def finish_interaction(interaction_id: str, **fields: Any) -> None:
    allowed = {
        "final_answer", "status", "error", "ended_at", "latency_ms", "llm_latency_ms",
        "mcp_latency_ms", "llm_calls", "tool_calls", "tool_errors", "iterations",
        "prompt_tokens", "completion_tokens", "total_tokens", "mlflow_trace_id", "mlflow_run_id",
    }
    sets, params = [], []
    for key, value in fields.items():
        if key in allowed:
            sets.append(f"{key} = ?")
            params.append(value)
    if not sets:
        return
    params.append(interaction_id)
    await db.execute(f"UPDATE interactions SET {', '.join(sets)} WHERE id = ?", tuple(params))


async def get_interaction(interaction_id: str) -> dict[str, Any] | None:
    row = await db.fetch_one("SELECT * FROM interactions WHERE id = ?", (interaction_id,))
    if row:
        row["mcp_servers"] = loads(row["mcp_servers"], [])
    return row


async def list_interactions(
    session_id: str | None = None, conversation_id: str | None = None, limit: int = 200
) -> list[dict[str, Any]]:
    clauses, params = [], []
    if session_id:
        clauses.append("session_id = ?")
        params.append(session_id)
    if conversation_id:
        clauses.append("conversation_id = ?")
        params.append(conversation_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    rows = await db.fetch_all(
        f"SELECT * FROM interactions {where} ORDER BY started_at DESC LIMIT ?", tuple(params)
    )
    for row in rows:
        row["mcp_servers"] = loads(row["mcp_servers"], [])
    return rows


# ------------------------------------------------------ eventos herramienta -
async def add_tool_event(
    *,
    interaction_id: str,
    conversation_id: str,
    session_id: str,
    seq: int,
    iteration: int,
    tool_name: str,
    real_tool_name: str,
    server_name: str,
    server_url: str,
    conn_id: str,
    arguments: dict,
    result_text: str,
    structured: Any,
    ok: bool,
    error: str,
    latency_ms: float,
    frames: list,
) -> str:
    event_id = new_id("tev")
    await db.execute(
        """INSERT INTO tool_events
           (id, interaction_id, conversation_id, session_id, seq, iteration, tool_name,
            real_tool_name, server_name, server_url, conn_id, arguments, result_text,
            structured, ok, error, latency_ms, frames, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            event_id, interaction_id, conversation_id, session_id, seq, iteration, tool_name,
            real_tool_name, server_name, server_url, conn_id, dumps(arguments), result_text,
            dumps(structured), 1 if ok else 0, error, latency_ms, dumps(frames), now(),
        ),
    )
    return event_id


async def list_tool_events(
    interaction_id: str | None = None,
    conversation_id: str | None = None,
    session_id: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    clauses, params = [], []
    for column, value in (
        ("interaction_id", interaction_id),
        ("conversation_id", conversation_id),
        ("session_id", session_id),
    ):
        if value:
            clauses.append(f"{column} = ?")
            params.append(value)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    rows = await db.fetch_all(
        f"SELECT * FROM tool_events {where} ORDER BY created_at DESC LIMIT ?", tuple(params)
    )
    for row in rows:
        row["arguments"] = loads(row["arguments"], {})
        row["structured"] = loads(row["structured"], None)
        row["frames"] = loads(row["frames"], [])
        row["ok"] = bool(row["ok"])
    return rows


async def cancel_running_interactions(session_id: str) -> None:
    """Cierra las interacciones que quedaron a medias al cancelar una evaluacion."""
    await db.execute(
        "UPDATE interactions SET status = 'cancelled', ended_at = ? "
        "WHERE session_id = ? AND status = 'running'",
        (now(), session_id),
    )


# ------------------------------------------------------------- evaluaciones -
_EVAL_RUN_JSON = {"config": {}, "summary": {}}
_EVAL_RESULT_JSON = {"items": [], "verdict": {}, "transcript": [], "metrics": {}, "mlflow_trace_ids": []}


def _decode(row: dict[str, Any] | None, fields: dict[str, Any]) -> dict[str, Any] | None:
    if row is None:
        return None
    for key, fallback in fields.items():
        if key in row:
            row[key] = loads(row[key], fallback)
    if "passed" in row and row["passed"] is not None:
        row["passed"] = bool(row["passed"])
    return row


async def create_eval_run(
    eval_run_id: str,
    session_id: str,
    *,
    name: str,
    suite: str,
    config: dict[str, Any],
    personas_yaml: str,
    cases_yaml: str,
    mlflow_experiment: str,
) -> None:
    await db.execute(
        """INSERT INTO eval_runs
           (id, session_id, name, suite, status, created_at, config, personas_yaml, cases_yaml,
            mlflow_experiment)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            eval_run_id, session_id, name, suite, "running", now(), dumps(config),
            personas_yaml, cases_yaml, mlflow_experiment,
        ),
    )


async def update_eval_run(eval_run_id: str, **fields: Any) -> None:
    allowed = {"status", "ended_at", "summary", "error", "mlflow_run_id", "name"}
    sets, params = [], []
    for key, value in fields.items():
        if key in allowed:
            sets.append(f"{key} = ?")
            params.append(dumps(value) if key == "summary" else value)
    if not sets:
        return
    params.append(eval_run_id)
    await db.execute(f"UPDATE eval_runs SET {', '.join(sets)} WHERE id = ?", tuple(params))


async def get_eval_run(eval_run_id: str) -> dict[str, Any] | None:
    row = await db.fetch_one("SELECT * FROM eval_runs WHERE id = ?", (eval_run_id,))
    return _decode(row, _EVAL_RUN_JSON)


async def list_eval_runs(limit: int = 100) -> list[dict[str, Any]]:
    rows = await db.fetch_all(
        """SELECT id, session_id, name, suite, status, created_at, ended_at, summary, error,
                  mlflow_experiment, mlflow_run_id
           FROM eval_runs ORDER BY created_at DESC LIMIT ?""",
        (limit,),
    )
    return [_decode(r, {"summary": {}}) for r in rows]  # type: ignore[misc]


async def mark_stale_eval_runs() -> None:
    """Al arrancar, lo que figuraba en marcha ya no lo esta: el proceso murio."""
    await db.execute(
        "UPDATE eval_runs SET status = 'interrupted', ended_at = ? WHERE status = 'running'",
        (now(),),
    )
    await db.execute(
        "UPDATE eval_results SET status = 'interrupted' WHERE status IN ('pending', 'running')"
    )


async def create_eval_result(
    result_id: str,
    eval_run_id: str,
    *,
    seq: int,
    case_id: str,
    case_title: str,
    persona_id: str,
    persona_name: str,
    repetition: int,
) -> None:
    await db.execute(
        """INSERT INTO eval_results
           (id, eval_run_id, seq, case_id, case_title, persona_id, persona_name, repetition, status)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (result_id, eval_run_id, seq, case_id, case_title, persona_id, persona_name, repetition, "pending"),
    )


async def update_eval_result(result_id: str, **fields: Any) -> None:
    allowed = {
        "conversation_id", "status", "score", "passed", "turns", "end_reason", "items", "verdict",
        "transcript", "metrics", "error", "started_at", "ended_at", "mlflow_run_id", "mlflow_trace_ids",
    }
    sets, params = [], []
    for key, value in fields.items():
        if key not in allowed:
            continue
        sets.append(f"{key} = ?")
        if key in _EVAL_RESULT_JSON:
            params.append(dumps(value))
        elif key == "passed" and value is not None:
            params.append(1 if value else 0)
        else:
            params.append(value)
    if not sets:
        return
    params.append(result_id)
    await db.execute(f"UPDATE eval_results SET {', '.join(sets)} WHERE id = ?", tuple(params))


async def list_eval_results(eval_run_id: str) -> list[dict[str, Any]]:
    rows = await db.fetch_all(
        "SELECT * FROM eval_results WHERE eval_run_id = ? ORDER BY seq ASC", (eval_run_id,)
    )
    return [_decode(r, _EVAL_RESULT_JSON) for r in rows]  # type: ignore[misc]


# ---------------------------------------------------------------- metricas --
async def session_stats(session_id: str) -> dict[str, Any]:
    totals = await db.fetch_one(
        """SELECT COUNT(*) AS interactions,
                  COALESCE(SUM(total_tokens),0)      AS total_tokens,
                  COALESCE(SUM(prompt_tokens),0)     AS prompt_tokens,
                  COALESCE(SUM(completion_tokens),0) AS completion_tokens,
                  COALESCE(SUM(tool_calls),0)        AS tool_calls,
                  COALESCE(SUM(tool_errors),0)       AS tool_errors,
                  COALESCE(AVG(latency_ms),0)        AS avg_latency_ms,
                  COALESCE(MAX(latency_ms),0)        AS max_latency_ms
           FROM interactions WHERE session_id = ?""",
        (session_id,),
    ) or {}
    per_tool = await db.fetch_all(
        """SELECT tool_name, server_name, COUNT(*) AS calls,
                  SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END) AS errors,
                  AVG(latency_ms) AS avg_latency_ms
           FROM tool_events WHERE session_id = ? GROUP BY tool_name, server_name
           ORDER BY calls DESC""",
        (session_id,),
    )
    conversations = await db.fetch_one(
        "SELECT COUNT(*) AS n FROM conversations WHERE session_id = ?", (session_id,)
    )
    return {
        "session_id": session_id,
        "conversations": (conversations or {}).get("n", 0),
        **totals,
        "per_tool": per_tool,
    }
