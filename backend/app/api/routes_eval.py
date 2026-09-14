"""Evaluaciones: validar YAML, lanzar baterias, seguirlas y consultarlas."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from sse_starlette.sse import EventSourceResponse

from ..evals.runner import AgentModel, EvalConfigError, EvalRequest, RoleModel, eval_manager
from ..evals.spec import parse_suite
from ..observability.mlflow_tracker import tracker
from ..store import repository as repo
from .schemas import EvalRoleModel, StartEvalRequest, ValidateEvalRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/eval", tags=["evaluacion"])

TEMPLATES = Path(__file__).resolve().parent.parent / "evals" / "templates"


def _role(payload: EvalRoleModel | None, fallback: AgentModel) -> RoleModel:
    if payload is None:
        # Mismo modelo que el agente, pero sin su prompt ni sus limites.
        return RoleModel(
            provider=fallback.provider,
            base_url=fallback.base_url,
            model=fallback.model,
            api_key=fallback.api_key,
            temperature=fallback.temperature,
            thinking=fallback.thinking,
            max_tokens=fallback.max_tokens,
        )
    return RoleModel(**payload.model_dump())


async def _mlflow_links(run: dict) -> dict:
    info = await tracker.experiment_info(run.get("mlflow_experiment", ""))
    base = info.get("url", "")
    run_id = run.get("mlflow_run_id", "")
    return {
        "experiment": info.get("name", ""),
        "experiment_url": base,
        "run_id": run_id,
        "run_url": f"{base}/runs/{run_id}" if base.startswith("http") and run_id else "",
        # Filtro para la vista de traces de MLflow.
        "trace_filter": f"tags.eval_run_id = '{run['id']}'",
    }


@router.get("/templates")
async def templates() -> dict:
    """Plantillas de ejemplo con la estructura propuesta de los dos YAML."""
    return {
        "personas_yaml": (TEMPLATES / "personas.yaml").read_text(encoding="utf-8"),
        "cases_yaml": (TEMPLATES / "casos.yaml").read_text(encoding="utf-8"),
    }


@router.post("/validate")
async def validate(payload: ValidateEvalRequest) -> dict:
    """Valida la pareja de ficheros y devuelve lo que se ejecutaria."""
    return parse_suite(payload.personas_yaml, payload.cases_yaml).summary()


@router.post("/runs")
async def start_run(payload: StartEvalRequest) -> dict:
    agent = AgentModel(**payload.agent.model_dump())
    request = EvalRequest(
        personas_yaml=payload.personas_yaml,
        cases_yaml=payload.cases_yaml,
        agent=agent,
        simulator=_role(payload.simulator, agent),
        judge=_role(payload.judge, agent),
        mcp_conn_ids=payload.mcp_conn_ids,
        name=payload.name,
        mlflow_experiment=payload.mlflow_experiment,
        case_ids=payload.case_ids,
        persona_ids=payload.persona_ids,
        repetitions=payload.repetitions,
        max_turns_override=payload.max_turns_override,
    )
    try:
        job = await eval_manager.start(request)
    except EvalConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "eval_run_id": job.id,
        "session_id": job.session_id,
        "name": request.name,
        "items": len(job.items),
    }


@router.get("/runs")
async def list_runs(limit: int = Query(50, ge=1, le=500)) -> dict:
    runs = await repo.list_eval_runs(limit)
    live = set(eval_manager.running())
    for run in runs:
        run["live"] = run["id"] in live
    return {"runs": runs}


@router.get("/runs/{eval_run_id}")
async def get_run(eval_run_id: str) -> dict:
    run = await repo.get_eval_run(eval_run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluacion no encontrada")
    job = eval_manager.get(eval_run_id)
    return {
        "run": run,
        "results": await repo.list_eval_results(eval_run_id),
        "live": bool(job and not job.finished),
        "events": len(job.events) if job else 0,
        "mlflow": await _mlflow_links(run),
    }


@router.get("/runs/{eval_run_id}/events")
async def run_events(eval_run_id: str, request: Request, since: int = Query(0, ge=0)) -> EventSourceResponse:
    """Progreso en vivo por SSE (GET, para poder usar EventSource).

    `since` (o la cabecera Last-Event-ID que manda EventSource al reconectar)
    permite reengancharse sin perder ni repetir eventos.
    """
    job = eval_manager.get(eval_run_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail="La evaluacion no esta en marcha en este proceso; consulta su resultado guardado",
        )
    last = request.headers.get("last-event-id")
    if last and last.isdigit():
        since = max(since, int(last) + 1)

    async def stream() -> AsyncIterator[dict]:
        async for event in eval_manager.stream(job, since):
            if await request.is_disconnected():
                break
            yield {
                "event": event["type"],
                "id": str(event["seq"]),
                "data": json.dumps(event, ensure_ascii=False, default=str),
            }

    return EventSourceResponse(
        stream(),
        ping=15,
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/runs/{eval_run_id}/cancel")
async def cancel_run(eval_run_id: str) -> dict:
    job = eval_manager.get(eval_run_id)
    if job is None or job.finished:
        raise HTTPException(status_code=409, detail="La evaluacion no esta en marcha")
    job.cancel()
    return {"cancelled": True}


@router.delete("/runs/{eval_run_id}")
async def delete_run(eval_run_id: str, purge_mlflow: bool = Query(True)) -> dict:
    """Borra la ejecucion con su sesion, sus conversaciones y su rastro en MLflow."""
    run = await repo.get_eval_run(eval_run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluacion no encontrada")
    job = eval_manager.get(eval_run_id)
    if job and not job.finished:
        raise HTTPException(status_code=409, detail="Cancela la evaluacion antes de borrarla")

    session_id = run["session_id"]
    for conv_id in await repo.list_conversation_ids(session_id):
        await tracker.close_conversation_run(conv_id)
    purged = {"runs": 0, "traces": 0}
    if purge_mlflow:
        purged = await tracker.delete_session(session_id)
    await repo.delete_session(session_id)
    return {"deleted": True, "session_id": session_id, "mlflow": purged}
