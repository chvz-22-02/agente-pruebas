"""Que pasa cuando el servidor MCP se cae con trabajo en vuelo.

Regresion de un fallo real: una bateria de 55 casos se detuvo sola en el 13
tras casi una hora, la UI la dio por "cancelada" sin que nadie pulsara nada, el
evaluador no llego a juzgar la conversacion y su run de MLflow quedo abierto.

La cadena era esta:

1. la conexion MCP se cayo y su tarea supervisora cancelo el future del
   comando en vuelo;
2. `CancelledError` es `BaseException`, asi que se colo por el `except
   Exception` del bucle del agente sin cerrar la traza ni escribir metricas;
3. arriba del todo, la evaluacion lo confundio con un "detener" del usuario.

    python -m pytest tests/test_mcp_connection.py -q
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.mcpclient.connection import MCPConnection, MCPConnectionError  # noqa: E402
from app.mcpclient.models import MCPServerConfig  # noqa: E402


def _connection() -> MCPConnection:
    return MCPConnection(MCPServerConfig(url="https://ejemplo.invalid/mcp", timeout_s=5))


def test_dying_connection_fails_the_caller_instead_of_cancelling_it() -> None:
    """Quien espera una herramienta recibe un error, no una cancelacion."""

    async def scenario() -> None:
        conn = _connection()
        conn.last_error = "streamable_http: peer closed connection"

        async def dispatch_dies(*_: Any, **__: Any) -> Any:
            raise asyncio.CancelledError

        conn._dispatch = dispatch_dies  # type: ignore[method-assign]

        future: asyncio.Future = asyncio.get_running_loop().create_future()
        await conn._queue.put(("call_tool", {"name": "x", "arguments": {}}, future))

        loop_task = asyncio.create_task(conn._command_loop(None))
        try:
            await loop_task
        except asyncio.CancelledError:
            pass  # el supervisor si muere: es su cancelacion

        assert not future.cancelled(), "al que esperaba se le cancelo el future"
        exc = future.exception()
        assert isinstance(exc, MCPConnectionError), exc
        # El motivo real viaja con el error, que es lo que vera el modelo.
        assert "call_tool" in str(exc) and "peer closed" in str(exc), exc

    asyncio.run(scenario())


def test_queued_commands_do_not_wait_for_the_timeout() -> None:
    """Lo que quedo en la cola falla ya, en vez de esperar 60 s a nadie."""

    async def scenario() -> None:
        conn = _connection()
        pending: asyncio.Future = asyncio.get_running_loop().create_future()
        await conn._queue.put(("call_tool", {"name": "y", "arguments": {}}, pending))

        conn._fail_pending()

        assert pending.done() and not pending.cancelled()
        assert isinstance(pending.exception(), MCPConnectionError)
        assert conn._queue.empty()

    asyncio.run(scenario())


def test_reconnect_keeps_the_connection_id() -> None:
    """Reconectar cierra la sesion vieja y registra la nueva bajo el mismo id.

    Una bateria en marcha, las conversaciones guardadas y la UI referencian la
    conexion por `conn_id`: si cambiara al reconectar, todo eso apuntaria a
    una conexion muerta.
    """

    async def scenario() -> None:
        from app.mcpclient import manager as manager_mod
        from app.mcpclient.manager import MCPManager

        stopped: list[str] = []

        class FakeConnection:
            def __init__(self, config: MCPServerConfig, on_frame: Any = None) -> None:
                self.conn_id = f"mcp_{id(self)}"
                self.config = config
                self.server_info = {"name": "fake"}
                self.tools = ["a", "b"]
                self.active_transport = "streamable_http"
                self.alive = False

            @property
            def is_alive(self) -> bool:
                return self.alive

            async def start(self) -> dict[str, Any]:
                self.alive = True
                return {"conn_id": self.conn_id, "config": {"url": self.config.url}, "tools": self.tools}

            async def stop(self) -> None:
                stopped.append(self.conn_id)
                self.alive = False

            def describe(self) -> dict[str, Any]:
                return {"conn_id": self.conn_id}

        saved = manager_mod.MCPConnection
        manager_mod.MCPConnection = FakeConnection  # type: ignore[misc]
        try:
            manager = MCPManager()
            conn_id = (await manager.connect(MCPServerConfig(url="https://x/mcp")))["conn_id"]
            old = manager.get(conn_id)
            old.alive = False  # se cayo
            assert manager.alive_ids([conn_id]) == []

            info = await manager.reconnect(conn_id)
            new = manager.get(conn_id)
            assert new is not old and new.conn_id == conn_id
            assert stopped == [conn_id], "la sesion vieja tiene que cerrarse antes"
            assert manager.alive_ids([conn_id]) == [conn_id]
            assert info["conn_id"] == conn_id and info["config"]["url"] == "https://x/mcp"

            try:
                await manager.reconnect("mcp_inexistente")
            except MCPConnectionError as exc:
                assert "no encontrada" in str(exc)
            else:
                raise AssertionError("reconectar un id desconocido debe fallar")
        finally:
            manager_mod.MCPConnection = saved  # type: ignore[misc]

    asyncio.run(scenario())


def test_router_turns_a_dead_server_into_an_observation() -> None:
    """Un MCP caido es un resultado de herramienta con error, no una excepcion.

    Asi el turno termina, se puede evaluar y el modelo tiene ocasion de
    reaccionar, que es como se tratan el resto de errores de herramienta.
    """

    async def scenario() -> None:
        from app.mcpclient.manager import ToolRouter

        conn = _connection()

        async def call_tool_dies(*_: Any, **__: Any) -> Any:
            raise MCPConnectionError("La conexion se cerro")

        conn.call_tool = call_tool_dies  # type: ignore[method-assign]

        class _Manager:
            def get(self, _conn_id: str) -> MCPConnection:
                return conn

        router = ToolRouter.__new__(ToolRouter)
        router.manager = _Manager()  # type: ignore[assignment]
        router._map = {"consultar": (conn.conn_id, "consultar")}

        result = await router.call("consultar", {})
        assert result.ok is False
        assert result.tool_name == "consultar"
        assert "se cerro" in (result.error or ""), result.error

    asyncio.run(scenario())
