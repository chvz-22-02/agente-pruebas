"""Endpoints de gestion de servidores MCP."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..mcpclient.connection import MCPConnectionError
from ..mcpclient.manager import mcp_manager
from ..mcpclient.models import MCPServerConfig
from .schemas import ConnectMCPRequest, DirectToolCallRequest, ReadResourceRequest

router = APIRouter(prefix="/api/mcp", tags=["mcp"])


@router.post("/connect")
async def connect(payload: ConnectMCPRequest) -> dict:
    config = MCPServerConfig(
        url=payload.url.strip(),
        name=payload.name.strip(),
        transport=payload.transport,
        headers=payload.headers,
        timeout_s=payload.timeout_s,
        capture_raw=payload.capture_raw,
    )
    try:
        return await mcp_manager.connect(config, reuse=payload.reuse)
    except MCPConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc


@router.get("/connections")
async def list_connections() -> dict:
    return {"connections": mcp_manager.list_connections()}


@router.delete("/connections/{conn_id}")
async def disconnect(conn_id: str) -> dict:
    return {"disconnected": await mcp_manager.disconnect(conn_id)}


@router.get("/connections/{conn_id}")
async def get_connection(conn_id: str) -> dict:
    try:
        return mcp_manager.get(conn_id).describe()
    except MCPConnectionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/connections/{conn_id}/refresh")
async def refresh_tools(conn_id: str) -> dict:
    try:
        conn = mcp_manager.get(conn_id)
        await conn.refresh_tools()
        return conn.describe()
    except MCPConnectionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/connections/{conn_id}/frames")
async def frames(conn_id: str, limit: int = Query(100, ge=1, le=500)) -> dict:
    """Ultimas tramas JSON-RPC intercambiadas con este servidor."""
    try:
        return {"frames": mcp_manager.get(conn_id).recent_frames(limit)}
    except MCPConnectionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/connections/{conn_id}/call")
async def call_tool(conn_id: str, payload: DirectToolCallRequest) -> dict:
    """Llamada manual a una herramienta, util para aislar fallos sin el LLM."""
    try:
        result = await mcp_manager.get(conn_id).call_tool(payload.tool, payload.arguments)
        return result.as_dict()
    except MCPConnectionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/connections/{conn_id}/resources")
async def resources(conn_id: str) -> dict:
    try:
        return {"resources": await mcp_manager.get(conn_id).list_resources()}
    except MCPConnectionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - el servidor puede no soportar resources
        return {"resources": [], "unsupported": f"{type(exc).__name__}: {exc}"}


@router.get("/connections/{conn_id}/prompts")
async def prompts(conn_id: str) -> dict:
    try:
        return {"prompts": await mcp_manager.get(conn_id).list_prompts()}
    except MCPConnectionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        return {"prompts": [], "unsupported": f"{type(exc).__name__}: {exc}"}


@router.post("/connections/{conn_id}/read-resource")
async def read_resource(conn_id: str, payload: ReadResourceRequest) -> dict:
    try:
        return await mcp_manager.get(conn_id).read_resource(payload.uri)
    except MCPConnectionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/connections/{conn_id}/ping")
async def ping(conn_id: str) -> dict:
    try:
        return {"ok": await mcp_manager.get(conn_id).ping()}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
