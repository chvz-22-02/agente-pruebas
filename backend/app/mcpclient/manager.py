"""Gestor de conexiones MCP.

Permite tener varios servidores MCP conectados a la vez y enruta cada llamada
de herramienta al servidor correcto. Los nombres se desambiguan solo cuando
colisionan, para que el modelo vea nombres cortos siempre que sea posible.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from typing import Any

from ..llm.base import ToolSpec
from .connection import MCPConnection, MCPConnectionError
from .models import MCPServerConfig, MCPToolResult, RawFrame

logger = logging.getLogger(__name__)

SAFE_NAME = re.compile(r"[^a-zA-Z0-9_-]")


def _slug(value: str) -> str:
    return SAFE_NAME.sub("_", value.strip())[:32] or "mcp"


class MCPManager:
    """Registro global de conexiones MCP vivas."""

    def __init__(self) -> None:
        self._connections: dict[str, MCPConnection] = {}
        self._lock = asyncio.Lock()
        self._frame_listeners: list[Callable[[str, RawFrame], None]] = []

    # ------------------------------------------------------------ listeners -
    def add_frame_listener(self, listener: Callable[[str, RawFrame], None]) -> None:
        self._frame_listeners.append(listener)

    def remove_frame_listener(self, listener: Callable[[str, RawFrame], None]) -> None:
        if listener in self._frame_listeners:
            self._frame_listeners.remove(listener)

    def _broadcast(self, conn_id: str, frame: RawFrame) -> None:
        for listener in list(self._frame_listeners):
            try:
                listener(conn_id, frame)
            except Exception:  # noqa: BLE001
                logger.debug("listener de tramas fallido", exc_info=True)

    # ---------------------------------------------------------- conexiones --
    async def connect(self, config: MCPServerConfig, reuse: bool = True) -> dict[str, Any]:
        """Conecta (o reutiliza) un servidor MCP y devuelve su descripcion."""
        async with self._lock:
            if reuse:
                for conn in self._connections.values():
                    if conn.config.url == config.url and conn.is_alive:
                        await conn.refresh_tools()
                        return conn.describe()
            conn = MCPConnection(config, on_frame=self._broadcast)
            description = await conn.start()
            self._connections[conn.conn_id] = conn
            logger.info(
                "MCP conectado: %s (%s) via %s con %d herramientas",
                config.url,
                conn.server_info.get("name"),
                conn.active_transport,
                len(conn.tools),
            )
            return description

    async def disconnect(self, conn_id: str) -> bool:
        async with self._lock:
            conn = self._connections.pop(conn_id, None)
        if conn is None:
            return False
        await conn.stop()
        return True

    async def reconnect(self, conn_id: str) -> dict[str, Any]:
        """Levanta de nuevo una conexion caida **con el mismo identificador**.

        Una bateria de evaluacion, las conversaciones guardadas y la UI
        referencian la conexion por `conn_id`; si al reconectar cambiara, todo
        eso quedaria apuntando a una conexion muerta. Primero se cierra la
        vieja del todo -aunque su sesion siguiera medio abierta- y despues se
        abre otra con la misma configuracion y se registra bajo el mismo id.
        """
        async with self._lock:
            old = self._connections.pop(conn_id, None)
        if old is None:
            raise MCPConnectionError(f"Conexion MCP no encontrada: {conn_id}")
        await old.stop()

        conn = MCPConnection(old.config, on_frame=self._broadcast)
        conn.conn_id = conn_id
        description = await conn.start()
        async with self._lock:
            self._connections[conn_id] = conn
        logger.info(
            "MCP reconectado: %s (%s) via %s con %d herramientas",
            old.config.url,
            conn.server_info.get("name"),
            conn.active_transport,
            len(conn.tools),
        )
        return description

    async def disconnect_all(self) -> None:
        async with self._lock:
            conns = list(self._connections.values())
            self._connections.clear()
        for conn in conns:
            await conn.stop()

    def get(self, conn_id: str) -> MCPConnection:
        conn = self._connections.get(conn_id)
        if conn is None:
            raise MCPConnectionError(f"Conexion MCP no encontrada: {conn_id}")
        return conn

    def list_connections(self) -> list[dict[str, Any]]:
        return [c.describe() for c in self._connections.values()]

    def alive_ids(self, conn_ids: list[str] | None = None) -> list[str]:
        """Filtra las conexiones vivas.

        `None` significa "todas las conectadas"; una lista vacia significa
        "ninguna", que es lo que envia la UI cuando no hay servidor marcado.
        """
        candidates = list(self._connections) if conn_ids is None else conn_ids
        return [
            cid for cid in candidates if cid in self._connections and self._connections[cid].is_alive
        ]


class ToolRouter:
    """Vista de herramientas que se le entrega al LLM en una interaccion."""

    def __init__(self, manager: MCPManager, conn_ids: list[str]) -> None:
        self.manager = manager
        self.conn_ids = conn_ids
        self._map: dict[str, tuple[str, str]] = {}  # nombre expuesto -> (conn_id, nombre real)
        self._specs: list[ToolSpec] = []
        self._build()

    def _build(self) -> None:
        counts: dict[str, int] = {}
        for conn_id in self.conn_ids:
            for tool in self.manager.get(conn_id).tools:
                counts[tool.name] = counts.get(tool.name, 0) + 1

        for conn_id in self.conn_ids:
            conn = self.manager.get(conn_id)
            prefix = _slug(conn.server_info.get("name") or conn.conn_id)
            for tool in conn.tools:
                exposed = tool.name if counts[tool.name] == 1 else f"{prefix}__{tool.name}"
                self._map[exposed] = (conn_id, tool.name)
                self._specs.append(
                    ToolSpec(
                        name=exposed,
                        description=tool.description,
                        input_schema=tool.input_schema,
                    )
                )

    @property
    def specs(self) -> list[ToolSpec]:
        return self._specs

    @property
    def is_empty(self) -> bool:
        return not self._specs

    def catalog(self) -> list[dict[str, Any]]:
        return [{"name": s.name, "description": s.description} for s in self._specs]

    async def call(self, exposed_name: str, arguments: dict[str, Any]) -> MCPToolResult:
        target = self._map.get(exposed_name)
        if target is None:
            available = ", ".join(sorted(self._map)) or "(ninguna)"
            return MCPToolResult(
                tool_name=exposed_name,
                arguments=arguments,
                ok=False,
                text="",
                error=f"La herramienta '{exposed_name}' no existe. Disponibles: {available}",
            )
        conn_id, real_name = target
        try:
            result = await self.manager.get(conn_id).call_tool(real_name, arguments)
        except MCPConnectionError as exc:
            # El servidor se cayo a mitad del turno. Se le devuelve al modelo
            # como observacion, igual que cualquier otro error de herramienta:
            # el turno termina y se puede evaluar, en vez de reventar el bucle.
            return MCPToolResult(
                tool_name=exposed_name,
                arguments=arguments,
                ok=False,
                text="",
                error=str(exc),
            )
        result.tool_name = exposed_name
        return result

    def server_of(self, exposed_name: str) -> dict[str, str]:
        target = self._map.get(exposed_name)
        if target is None:
            return {"conn_id": "", "url": "", "server_name": "", "real_tool": exposed_name}
        conn_id, real_name = target
        conn = self.manager.get(conn_id)
        return {
            "conn_id": conn_id,
            "url": conn.config.url,
            "server_name": conn.server_info.get("name", ""),
            "real_tool": real_name,
        }


mcp_manager = MCPManager()
