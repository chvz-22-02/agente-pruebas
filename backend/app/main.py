"""Punto de entrada del backend.

Arranca en 0.0.0.0 y con CORS abierto por defecto para que el frontend pueda
vivir en otra maquina de la misma red.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import routes_chat, routes_data, routes_eval, routes_mcp, routes_system
from .config import settings
from .evals.runner import eval_manager
from .llm.registry import close_all
from .mcpclient.manager import mcp_manager
from .observability.mlflow_tracker import tracker
from .store import repository as repo
from .store.db import db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("agente-pruebas")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()
    # Las evaluaciones viven en memoria: si el proceso se reinicio, las que
    # figuraban en marcha ya no lo estan.
    await repo.mark_stale_eval_runs()
    logger.info("SQLite listo en %s", settings.db_path)
    await tracker.init()
    logger.info("Escuchando en http://%s:%s", settings.host, settings.port)
    try:
        yield
    finally:
        await eval_manager.shutdown()
        await mcp_manager.disconnect_all()
        await close_all()
        await db.close()
        logger.info("Backend detenido")


app = FastAPI(
    title="Agente de pruebas MCP",
    description="Backend de agentes locales para testear servidores MCP, con trazabilidad en MLflow.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,  # con allow_origins=["*"] los navegadores exigen esto en false
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

app.include_router(routes_system.router)
app.include_router(routes_mcp.router)
app.include_router(routes_chat.router)
app.include_router(routes_data.router)
app.include_router(routes_eval.router)


@app.get("/")
async def root() -> dict:
    return {
        "service": "agente-pruebas-mcp",
        "docs": "/docs",
        "health": "/api/health",
    }


def run() -> None:
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run()
