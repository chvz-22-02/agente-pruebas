"""Estado del sistema: LLM, MLflow y configuracion efectiva."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from ..config import settings
from ..llm import catalog
from ..llm.registry import PROVIDERS, MissingAPIKey, get_provider
from ..mcpclient.manager import mcp_manager
from ..observability.mlflow_tracker import tracker
from .schemas import EnsureExperimentRequest, ProbeLLMRequest, PullModelRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["sistema"])


@router.get("/health")
async def health() -> dict:
    try:
        llm = await (await get_provider()).health()
    except MissingAPIKey as exc:
        llm = {"ok": False, "needs_api_key": True, "error": str(exc)}
    return {
        "ok": True,
        "llm": llm,
        "mlflow": tracker.info(),
        "mcp_connections": len(mcp_manager.list_connections()),
    }


@router.get("/config")
async def config() -> dict:
    """Valores por defecto que la UI usa para prerrellenar sus formularios."""
    return {
        "providers": sorted(PROVIDERS),
        "defaults": {
            "provider": settings.llm_provider,
            "base_url": settings.llm_base_url,
            "model": settings.llm_model,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
            "num_ctx": settings.llm_num_ctx,
            "thinking": settings.llm_thinking,
            "max_iterations": settings.agent_max_iterations,
        },
        "mlflow": tracker.info(),
        "mlflow_url": tracker.experiment_url(),
        # Experimentos ya existentes: la UI los ofrece en un desplegable y
        # ademas acepta un nombre nuevo, que se crea al vuelo.
        "mlflow_experiments": await tracker.list_experiments(),
        # Motores que saben descargarse modelos solos: la UI usa esto para
        # mostrar u ocultar el boton de descarga.
        "pull_capable": sorted(name for name, cls in PROVIDERS.items() if cls.supports_pull),
        # Proveedores y modelos que la UI ofrece para elegir.
        "catalog": catalog.as_json(),
        # Sugerencias para este perfil de maquina (CPU, 32 GB, sin GPU).
        "suggested_models": [
            {"name": "qwen3:8b", "size": "5,2 GB", "note": "equilibrio recomendado"},
            {"name": "qwen3:4b", "size": "2,6 GB", "note": "el doble de rapido"},
            {"name": "qwen3:14b", "size": "9,3 GB", "note": "mas preciso, mas lento"},
            {"name": "granite3.3:8b", "size": "4,9 GB", "note": "alternativa con buen tool calling"},
        ],
    }


@router.get("/llm/models")
async def models(provider: str | None = None, base_url: str | None = None) -> dict:
    try:
        engine = await get_provider(provider, base_url)
        return {"provider": engine.name, "base_url": engine.base_url, "models": await engine.list_models()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"No se pudo consultar el motor LLM: {type(exc).__name__}: {exc}",
        ) from exc


@router.post("/llm/pull")
async def pull_model(payload: PullModelRequest, request: Request) -> EventSourceResponse:
    """Descarga un modelo en el motor local y retransmite el progreso por SSE.

    Ollama deduplica descargas por capas, asi que relanzar una descarga
    interrumpida reaprovecha lo ya bajado.
    """
    model = payload.model.strip()
    if not model:
        raise HTTPException(status_code=400, detail="Indica la etiqueta del modelo")

    try:
        engine = await get_provider(payload.provider, payload.base_url, model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not engine.supports_pull:
        raise HTTPException(
            status_code=400,
            detail=(
                f"El proveedor '{engine.name}' no descarga modelos desde su API. "
                "Cambia a Ollama o carga el modelo en tu motor manualmente."
            ),
        )

    async def event_stream() -> AsyncIterator[dict]:
        # Bytes ya descargados por capa: Ollama informa por capa, no del total.
        layers: dict[str, dict[str, int]] = {}
        try:
            async for update in engine.pull_model(model):
                if await request.is_disconnected():
                    # El pull sigue en Ollama; al reabrirlo se retoma donde iba.
                    logger.info("Cliente desconectado durante la descarga de %s", model)
                    break

                if error := update.get("error"):
                    yield {"event": "error", "data": json.dumps({"type": "error", "message": error})}
                    return

                digest = update.get("digest")
                if digest and update.get("total"):
                    layers[digest] = {
                        "completed": int(update.get("completed") or 0),
                        "total": int(update["total"]),
                    }
                total = sum(v["total"] for v in layers.values())
                completed = sum(v["completed"] for v in layers.values())

                yield {
                    "event": "progress",
                    "data": json.dumps(
                        {
                            "type": "progress",
                            "model": model,
                            "status": update.get("status", ""),
                            "completed": completed,
                            "total": total,
                            "percent": round(completed / total * 100, 1) if total else None,
                        },
                        ensure_ascii=False,
                    ),
                }

            installed = await engine.list_models()
            yield {
                "event": "done",
                "data": json.dumps(
                    {"type": "done", "model": model, "installed": model in installed, "models": installed}
                ),
            }
        except Exception as exc:  # noqa: BLE001 - el error viaja al cliente, no tumba el server
            logger.exception("Fallo descargando %s", model)
            yield {
                "event": "error",
                "data": json.dumps({"type": "error", "message": f"{type(exc).__name__}: {exc}"}),
            }

    return EventSourceResponse(
        event_stream(),
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/llm/probe")
async def probe_llm(payload: ProbeLLMRequest) -> dict:
    """Valida credenciales y devuelve la lista real de modelos del proveedor.

    Es lo que la UI llama al pegar una clave: si responde `ok`, el desplegable
    de modelos se rellena con lo que la cuenta tiene disponible de verdad, no
    con el catalogo estatico.
    """
    try:
        engine = await get_provider(
            payload.provider, payload.base_url, payload.model, payload.api_key
        )
    except MissingAPIKey as exc:
        return {"ok": False, "needs_api_key": True, "error": str(exc), "models": []}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    health = await engine.health()
    return {**health, "models": health.get("models", [])}


@router.get("/llm/features")
async def llm_features(
    provider: str | None = None, base_url: str | None = None, model: str | None = None
) -> dict:
    """Capacidades reales del modelo seleccionado (razonamiento, tools, ...).

    Es lo que permite que el interruptor de razonamiento siga funcionando con
    modelos recien descargados que no estan en el catalogo estatico: en vez de
    suponerlo, se le pregunta al motor.
    """
    try:
        engine = await get_provider(provider, base_url, model)
    except MissingAPIKey as exc:
        return {"ok": False, "needs_api_key": True, "error": str(exc)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    features = await engine.model_features(model)
    catalog_entry = catalog.model_info(engine.name, model or engine.model)
    return {
        "ok": True,
        "provider": engine.name,
        "model": model or engine.model,
        "in_catalog": catalog_entry is not None,
        **features,
    }


@router.get("/llm/health")
async def llm_health(provider: str | None = None, base_url: str | None = None, model: str | None = None) -> dict:
    engine = await get_provider(provider, base_url, model)
    return await engine.health()


@router.get("/mlflow")
async def mlflow_info() -> dict:
    return {**tracker.info(), "experiment_url": tracker.experiment_url()}


@router.get("/mlflow/experiments")
async def list_experiments() -> dict:
    """Experimentos existentes, para el desplegable de la UI."""
    return {
        "experiments": await tracker.list_experiments(),
        "default": tracker.default_experiment,
        "available": tracker.available,
    }


@router.post("/mlflow/experiments")
async def ensure_experiment(payload: EnsureExperimentRequest) -> dict:
    """Selecciona un experimento desde la UI; lo crea si el nombre no existe."""
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Indica el nombre del experimento")
    if not tracker.available:
        raise HTTPException(
            status_code=503,
            detail=f"MLflow no esta disponible: {tracker.status}",
        )
    known = {e["name"] for e in await tracker.list_experiments()}
    info = await tracker.experiment_info(name)
    if not info.get("experiment_id"):
        raise HTTPException(status_code=502, detail=f"MLflow rechazo el experimento '{name}'")
    return {**info, "created": name not in known}
