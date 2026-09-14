"""Endpoint de chat con streaming SSE."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from ..agent.loop import AgentRunner, RunConfig
from ..config import settings
from ..store import repository as repo
from .schemas import ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


async def _resolve_context(payload: ChatRequest) -> tuple[str, str, str]:
    """Garantiza que existan sesion y conversacion; las crea si hacen falta.

    Devuelve `(session_id, conversation_id, experimento de MLflow)`. Las
    sesiones se crean aqui, con el primer mensaje: no hay ningun proceso que
    las cree por su cuenta.
    """
    experiment = payload.mlflow_experiment.strip()
    session_id = await repo.ensure_session(payload.session_id, mlflow_experiment=experiment)

    session = await repo.get_session(session_id) or {}
    if experiment and session.get("mlflow_experiment", "") != experiment:
        # La eleccion de la UI manda y queda guardada en la sesion.
        await repo.update_session(session_id, mlflow_experiment=experiment)
    else:
        experiment = session.get("mlflow_experiment", "")

    conversation_id = payload.conversation_id
    if conversation_id:
        conversation = await repo.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail=f"Conversacion no encontrada: {conversation_id}")
        session_id = conversation["session_id"]
    else:
        conversation = await repo.create_conversation(
            session_id,
            title=payload.message[:60],
            provider=payload.provider or settings.llm_provider,
            model=payload.model or settings.llm_model,
            system_prompt=payload.system_prompt,
        )
        conversation_id = conversation["id"]

    return session_id, conversation_id, experiment


@router.post("/chat")
async def chat(payload: ChatRequest, request: Request) -> EventSourceResponse:
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacio")

    session_id, conversation_id, experiment = await _resolve_context(payload)

    runner = AgentRunner(
        RunConfig(
            session_id=session_id,
            conversation_id=conversation_id,
            message=payload.message,
            mcp_conn_ids=payload.mcp_conn_ids,
            provider=payload.provider,
            base_url=payload.base_url,
            model=payload.model,
            api_key=payload.api_key,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
            thinking=payload.thinking,
            system_prompt=payload.system_prompt,
            max_iterations=payload.max_iterations,
            mlflow_experiment=experiment,
        )
    )

    async def event_stream() -> AsyncIterator[dict]:
        try:
            async for event in runner.run():
                if await request.is_disconnected():
                    logger.info("Cliente desconectado; se aborta %s", runner.interaction_id)
                    break
                yield {"event": event["type"], "data": json.dumps(event, ensure_ascii=False, default=str)}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error en el stream de chat")
            yield {
                "event": "error",
                "data": json.dumps({"type": "error", "message": f"{type(exc).__name__}: {exc}"}),
            }

    return EventSourceResponse(
        event_stream(),
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
