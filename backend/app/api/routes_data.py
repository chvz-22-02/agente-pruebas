"""Sesiones, conversaciones, mensajes y trazas almacenadas."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from ..config import settings
from ..evals.runner import eval_manager
from ..observability.mlflow_tracker import tracker
from ..store import repository as repo
from .schemas import (
    CreateConversationRequest,
    CreateSessionRequest,
    UpdateConversationRequest,
    UpdateSessionRequest,
)

router = APIRouter(prefix="/api", tags=["datos"])


# ---------------------------------------------------------------- sesiones --
@router.post("/sessions")
async def create_session(payload: CreateSessionRequest) -> dict:
    """Crea una sesion. Solo se llama desde el boton "+ Sesion" de la UI.

    El run de MLflow **no** se crea aqui a proposito: se crea con la primera
    interaccion. Asi una sesion que se abre y no se usa no deja rastro en el
    experimento.
    """
    session = await repo.create_session(
        payload.title, payload.metadata, payload.mlflow_experiment
    )
    return {**session, "mlflow": await tracker.experiment_info(payload.mlflow_experiment)}


@router.get("/sessions")
async def list_sessions(limit: int = Query(100, ge=1, le=500)) -> dict:
    return {"sessions": await repo.list_sessions(limit)}


@router.post("/sessions/prune")
async def prune_sessions(
    keep: str = Query("", description="Sesion a conservar aunque este vacia"),
    purge_mlflow: bool = Query(True),
) -> dict:
    """Borra de golpe las sesiones que no llegaron a registrar nada.

    Existe porque una version anterior de la UI creaba una sesion en cada
    recarga: sin esto habria que ir borrandolas de una en una.
    """
    # Una evaluacion recien lanzada aun no tiene interacciones: no esta vacia.
    empty = [
        s for s in await repo.list_empty_sessions(exclude=keep) if not eval_manager.is_session_busy(s["id"])
    ]
    purged = {"runs": 0, "traces": 0}
    for session in empty:
        for conv_id in await repo.list_conversation_ids(session["id"]):
            await tracker.close_conversation_run(conv_id)
        if purge_mlflow:
            result = await tracker.delete_session(session["id"])
            purged["runs"] += result["runs"]
            purged["traces"] += result["traces"]
        await repo.delete_session(session["id"])
    return {"deleted": len(empty), "sessions": [s["id"] for s in empty], "mlflow": purged}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    session = await repo.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Sesion no encontrada")
    return session


@router.patch("/sessions/{session_id}")
async def update_session(session_id: str, payload: UpdateSessionRequest) -> dict:
    """Renombra la sesion o cambia su experimento de MLflow.

    El nombre se propaga al run de MLflow (`mlflow.runName`). No hace falta que
    sea unico: los runs se identifican por `run_id` y aqui siempre se buscan
    por `tags.session_id`, asi que dos sesiones pueden llamarse igual.

    Cambiar el experimento no mueve lo ya registrado: a partir de la siguiente
    interaccion se crean runs nuevos en el experimento indicado.
    """
    if await repo.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Sesion no encontrada")
    fields = {k: v for k, v in payload.model_dump().items() if v is not None}
    if "mlflow_experiment" in fields:
        await tracker.ensure_experiment(fields["mlflow_experiment"])
    if title := (fields.get("title") or "").strip():
        # El run de MLflow, si ya existe, se renombra para no quedar desfasado.
        fields["title"] = title
        await tracker.rename_session(session_id, title)
    await repo.update_session(session_id, **fields)
    session = await repo.get_session(session_id)
    return {
        **(session or {}),
        "mlflow": await tracker.experiment_info((session or {}).get("mlflow_experiment", "")),
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, purge_mlflow: bool = Query(True)) -> dict:
    """Borra la sesion completa y, por defecto, tambien su rastro en MLflow.

    En MLflow se eliminan el run de la sesion, los de sus conversaciones y las
    trazas de cada interaccion, buscandolos por `tags.session_id`.
    """
    if await repo.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Sesion no encontrada")
    if eval_manager.is_session_busy(session_id):
        raise HTTPException(
            status_code=409, detail="La sesion es de una evaluacion en marcha: cancelala antes de borrarla"
        )

    for conv_id in await repo.list_conversation_ids(session_id):
        await tracker.close_conversation_run(conv_id)

    purged = {"runs": 0, "traces": 0}
    if purge_mlflow:
        purged = await tracker.delete_session(session_id)

    await repo.delete_session(session_id)
    return {"deleted": True, "session_id": session_id, "mlflow": purged}


@router.get("/sessions/{session_id}/stats")
async def session_stats(session_id: str) -> dict:
    return await repo.session_stats(session_id)


# ----------------------------------------------------------- conversaciones -
@router.post("/conversations")
async def create_conversation(payload: CreateConversationRequest) -> dict:
    session_id = await repo.ensure_session(payload.session_id)
    return await repo.create_conversation(
        session_id,
        title=payload.title,
        provider=payload.provider or settings.llm_provider,
        model=payload.model or settings.llm_model,
        system_prompt=payload.system_prompt,
        mcp_servers=payload.mcp_conn_ids,
    )


@router.get("/conversations")
async def list_conversations(session_id: str | None = None) -> dict:
    return {"conversations": await repo.list_conversations(session_id)}


@router.get("/conversations/{conv_id}")
async def get_conversation(conv_id: str) -> dict:
    conversation = await repo.get_conversation(conv_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversacion no encontrada")
    return conversation


@router.patch("/conversations/{conv_id}")
async def update_conversation(conv_id: str, payload: UpdateConversationRequest) -> dict:
    fields = {k: v for k, v in payload.model_dump().items() if v is not None}
    await repo.update_conversation(conv_id, **fields)
    return await get_conversation(conv_id)


@router.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str, purge_mlflow: bool = Query(True)) -> dict:
    await tracker.close_conversation_run(conv_id)
    purged = {"runs": 0, "traces": 0}
    if purge_mlflow:
        purged = await tracker.delete_conversation(conv_id)
    await repo.delete_conversation(conv_id)
    return {"deleted": True, "mlflow": purged}


@router.post("/conversations/{conv_id}/reset")
async def reset_conversation(conv_id: str, new_thread: bool = Query(True)) -> dict:
    """Boton 'reiniciar conversacion' de la UI.

    Por defecto abre un hilo nuevo (asi el historico y sus trazas se conservan).
    Con `new_thread=false` se limpia la memoria del hilo actual manteniendo su id.
    """
    conversation = await repo.get_conversation(conv_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversacion no encontrada")

    if not new_thread:
        await repo.clear_conversation_messages(conv_id)
        return {"reset": True, "conversation": await repo.get_conversation(conv_id)}

    await tracker.close_conversation_run(conv_id)
    fresh = await repo.create_conversation(
        conversation["session_id"],
        title="Nueva conversacion",
        provider=conversation["provider"],
        model=conversation["model"],
        system_prompt=conversation["system_prompt"],
        mcp_servers=conversation["mcp_servers"],
    )
    return {"reset": True, "conversation": fresh}


@router.get("/conversations/{conv_id}/messages")
async def get_messages(conv_id: str) -> dict:
    return {"messages": await repo.get_messages(conv_id)}


# ------------------------------------------------------------------ trazas --
@router.get("/traces/interactions")
async def list_interactions(
    session_id: str | None = None,
    conversation_id: str | None = None,
    limit: int = Query(200, ge=1, le=1000),
) -> dict:
    return {"interactions": await repo.list_interactions(session_id, conversation_id, limit)}


@router.get("/traces/interactions/{interaction_id}")
async def interaction_detail(interaction_id: str) -> dict:
    interaction = await repo.get_interaction(interaction_id)
    if interaction is None:
        raise HTTPException(status_code=404, detail="Interaccion no encontrada")
    tool_events = await repo.list_tool_events(interaction_id=interaction_id)
    session = await repo.get_session(interaction["session_id"]) or {}
    experiment = await tracker.experiment_info(session.get("mlflow_experiment", ""))
    return {
        "interaction": interaction,
        "tool_events": sorted(tool_events, key=lambda e: e["seq"]),
        "mlflow": {
            "experiment": experiment["name"],
            "experiment_url": experiment["url"],
            "trace_id": interaction.get("mlflow_trace_id", ""),
            "run_id": interaction.get("mlflow_run_id", ""),
            # Carpeta de artifacts con el volcado integro de cada llamada MCP.
            "tool_artifacts": f"mcp_tool_calls/{interaction_id}/",
        },
    }


@router.get("/traces/tool-events")
async def tool_events(
    session_id: str | None = None,
    conversation_id: str | None = None,
    interaction_id: str | None = None,
    limit: int = Query(300, ge=1, le=2000),
) -> dict:
    return {
        "tool_events": await repo.list_tool_events(
            interaction_id=interaction_id,
            conversation_id=conversation_id,
            session_id=session_id,
            limit=limit,
        )
    }


@router.get("/traces/export")
async def export_traces(session_id: str) -> JSONResponse:
    """Volcado completo de una sesion, listo para adjuntar a un informe."""
    session = await repo.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Sesion no encontrada")

    conversations = await repo.list_conversations(session_id)
    payload = {
        "session": session,
        "stats": await repo.session_stats(session_id),
        "mlflow": {
            **tracker.info(),
            **await tracker.experiment_info(session.get("mlflow_experiment", "")),
        },
        "conversations": [],
    }
    for conversation in conversations:
        interactions = await repo.list_interactions(conversation_id=conversation["id"])
        payload["conversations"].append(
            {
                **conversation,
                "messages": await repo.get_messages(conversation["id"]),
                "interactions": [
                    {
                        **interaction,
                        "tool_events": await repo.list_tool_events(interaction_id=interaction["id"]),
                    }
                    for interaction in interactions
                ],
            }
        )
    return JSONResponse(
        payload,
        headers={"Content-Disposition": f'attachment; filename="sesion-{session_id}.json"'},
    )
