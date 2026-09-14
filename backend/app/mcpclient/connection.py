"""Conexion viva contra un servidor MCP remoto.

Usa el cliente de alto nivel del SDK de MCP 2.x (`mcp.client.Client`). Ese
cliente se apoya en context managers y task groups de anyio, asi que la sesion
debe vivir entera dentro de una unica task. Por eso cada conexion arranca una
task supervisora y el resto del backend habla con ella por una cola de
comandos: asi la conexion sobrevive entre peticiones HTTP y conserva su estado
(session id, version de protocolo negociada, cache de la sesion).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from collections.abc import Callable
from typing import Any

import httpx2

from ..llm.base import ToolSpec
from .models import MCPServerConfig, MCPToolResult, RawFrame
from .tee import TeeReceiveStream, TeeSendStream

logger = logging.getLogger(__name__)

MAX_FRAME_BUFFER = 500


class MCPConnectionError(RuntimeError):
    pass


def _content_to_text(blocks: list[Any]) -> tuple[str, list[dict[str, Any]]]:
    """Aplana los bloques de contenido MCP a texto utilizable por el LLM."""
    parts: list[str] = []
    raw_blocks: list[dict[str, Any]] = []
    for block in blocks or []:
        try:
            dumped = block.model_dump(by_alias=True, exclude_none=True, mode="json")
        except AttributeError:
            dumped = {"type": "unknown", "value": str(block)}
        raw_blocks.append(dumped)
        kind = dumped.get("type")
        if kind == "text":
            parts.append(dumped.get("text", ""))
        elif kind == "resource":
            resource = dumped.get("resource") or {}
            parts.append(resource.get("text") or json.dumps(resource, ensure_ascii=False))
        elif kind in {"image", "audio"}:
            mime = dumped.get("mimeType") or dumped.get("mime_type") or "binario"
            parts.append("[" + str(kind) + ": " + str(mime) + "]")
        else:
            parts.append(json.dumps(dumped, ensure_ascii=False))
    return "\n".join(p for p in parts if p), raw_blocks


def _model_list(items: list[Any]) -> list[dict[str, Any]]:
    out = []
    for item in items or []:
        try:
            out.append(item.model_dump(by_alias=True, exclude_none=True, mode="json"))
        except AttributeError:
            out.append({"value": str(item)})
    return out


class MCPConnection:
    """Conexion persistente a un servidor MCP, identificada por `conn_id`."""

    def __init__(
        self,
        config: MCPServerConfig,
        on_frame: Callable[[str, RawFrame], None] | None = None,
    ) -> None:
        self.conn_id = f"mcp_{uuid.uuid4().hex[:10]}"
        self.config = config
        self.connected_at: float | None = None
        self.server_info: dict[str, Any] = {}
        self.tools: list[ToolSpec] = []
        self.active_transport: str | None = None
        self.last_error: str | None = None

        self._on_frame = on_frame
        self._frames: deque[RawFrame] = deque(maxlen=MAX_FRAME_BUFFER)
        self._collectors: list[list[RawFrame]] = []
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._ready: asyncio.Future | None = None
        self._closed = False

    # ----------------------------------------------------------- captura ----
    def _sink(self, direction: str, payload: dict[str, Any]) -> None:
        frame = RawFrame(direction=direction, payload=payload)  # type: ignore[arg-type]
        self._frames.append(frame)
        for collector in self._collectors:
            collector.append(frame)
        if self._on_frame:
            try:
                self._on_frame(self.conn_id, frame)
            except Exception:  # noqa: BLE001
                logger.debug("fallo en el listener de tramas", exc_info=True)

    def recent_frames(self, limit: int = 100) -> list[dict[str, Any]]:
        return [f.as_dict() for f in list(self._frames)[-limit:]]

    # -------------------------------------------------------- ciclo de vida -
    async def start(self) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        self._ready = loop.create_future()
        self._task = asyncio.create_task(self._run(), name=f"mcp-{self.conn_id}")
        try:
            return await asyncio.wait_for(self._ready, timeout=self.config.timeout_s)
        except TimeoutError as exc:
            await self.stop()
            raise MCPConnectionError(
                f"Timeout ({self.config.timeout_s}s) conectando a {self.config.url}"
            ) from exc

    async def stop(self) -> None:
        self._closed = True
        task = self._task
        if task and not task.done():
            await self._queue.put(None)
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=10)
            except (TimeoutError, asyncio.CancelledError):
                task.cancel()
        self._task = None

    @property
    def is_alive(self) -> bool:
        return bool(self._task and not self._task.done()) and not self._closed

    # ---------------------------------------------------------- transporte --
    def _transport_order(self) -> list[str]:
        if self.config.transport in {"streamable_http", "sse"}:
            return [self.config.transport]
        # auto: por convencion, una ruta terminada en /sse usa el transporte legado.
        if self.config.url.rstrip("/").endswith("/sse"):
            return ["sse", "streamable_http"]
        return ["streamable_http", "sse"]

    def _open_transport(self, kind: str):
        from mcp.client.sse import sse_client
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared._httpx_utils import create_mcp_http_client

        headers = dict(self.config.headers or {})
        if kind == "streamable_http":
            # Las cabeceras (auth) viajan en el cliente httpx que usa el transporte.
            http_client = create_mcp_http_client(
                headers=headers or None,
                timeout=httpx2.Timeout(self.config.timeout_s),
            )
            return streamable_http_client(self.config.url, http_client=http_client)
        if kind == "sse":
            return sse_client(
                url=self.config.url,
                headers=headers or None,
                timeout=self.config.timeout_s,
            )
        raise MCPConnectionError(f"Transporte no soportado: {kind}")

    async def _run(self) -> None:
        errors: list[str] = []
        for kind in self._transport_order():
            try:
                await self._serve(kind)
                return
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # noqa: BLE001 - anyio agrupa en ExceptionGroup
                detail = f"{kind}: {_flatten(exc)}"
                errors.append(detail)
                logger.warning("Conexion MCP fallida (%s)", detail)
                if self._ready is not None and self._ready.done():
                    # Ya habia conectado: la caida es posterior, no probamos otro transporte.
                    self.last_error = detail
                    return
        self.last_error = " | ".join(errors)
        if self._ready is not None and not self._ready.done():
            self._ready.set_exception(
                MCPConnectionError(f"No se pudo conectar a {self.config.url} -> {self.last_error}")
            )

    async def _serve(self, kind: str) -> None:
        from mcp import Implementation
        from mcp.client import Client

        transport = self._open_transport(kind)
        async with transport as streams:
            read, write = streams[0], streams[1]
            if self.config.capture_raw:
                read = TeeReceiveStream(read, self._sink)
                write = TeeSendStream(write, self._sink)

            client = Client(
                _PreopenedTransport(read, write),
                read_timeout_seconds=self.config.timeout_s,
                client_info=Implementation(name="agente-pruebas-mcp", version="0.1.0"),
            )
            async with client:
                self.active_transport = kind
                self.connected_at = time.time()
                info = client.server_info
                caps = client.server_capabilities
                self.server_info = {
                    "name": getattr(info, "name", "desconocido"),
                    "title": getattr(info, "title", None) or "",
                    "version": getattr(info, "version", ""),
                    "protocol_version": client.protocol_version or "",
                    "instructions": client.instructions or "",
                    "capabilities": caps.model_dump(exclude_none=True, mode="json") if caps else {},
                }
                self.tools = await self._fetch_tools(client)

                if self._ready is not None and not self._ready.done():
                    self._ready.set_result(self.describe())

                await self._command_loop(client)

    async def _fetch_tools(self, client: Any, refresh: bool = False) -> list[ToolSpec]:
        result = await client.list_tools(cache_mode="refresh" if refresh else "use")
        return [
            ToolSpec(
                name=t.name,
                description=t.description or "",
                input_schema=(t.input_schema or {"type": "object", "properties": {}}),
            )
            for t in result.tools
        ]

    async def _command_loop(self, client: Any) -> None:
        while True:
            command = await self._queue.get()
            if command is None:
                return
            op, payload, future = command
            if future.cancelled():
                continue
            try:
                result = await self._dispatch(client, op, payload)
                if not future.done():
                    future.set_result(result)
            except asyncio.CancelledError:
                if not future.done():
                    future.cancel()
                raise
            except Exception as exc:  # noqa: BLE001
                if not future.done():
                    future.set_exception(exc)

    async def _dispatch(self, client: Any, op: str, payload: Any) -> Any:
        if op == "call_tool":
            return await self._call_tool(client, payload["name"], payload["arguments"])
        if op == "list_tools":
            self.tools = await self._fetch_tools(client, refresh=True)
            return self.tools
        if op == "list_resources":
            return _model_list((await client.list_resources(cache_mode="refresh")).resources)
        if op == "list_prompts":
            return _model_list((await client.list_prompts(cache_mode="refresh")).prompts)
        if op == "read_resource":
            result = await client.read_resource(payload["uri"], cache_mode="refresh")
            return result.model_dump(by_alias=True, exclude_none=True, mode="json")
        if op == "ping":
            # `ping` desaparecio del protocolo 2026-07-28: un list_tools sirve
            # igual como prueba de vida de extremo a extremo.
            await client.list_tools(cache_mode="refresh")
            return True
        raise MCPConnectionError(f"Operacion desconocida: {op}")

    async def _call_tool(self, client: Any, name: str, arguments: dict[str, Any]) -> MCPToolResult:
        collector: list[RawFrame] = []
        self._collectors.append(collector)
        started = time.perf_counter()
        try:
            result = await client.call_tool(name, arguments or {})
            text, blocks = _content_to_text(getattr(result, "content", []))
            is_error = bool(getattr(result, "is_error", False))
            return MCPToolResult(
                tool_name=name,
                arguments=arguments,
                ok=not is_error,
                text=text,
                structured=getattr(result, "structured_content", None),
                content_blocks=blocks,
                error=text or "el servidor devolvio isError" if is_error else None,
                latency_ms=(time.perf_counter() - started) * 1000,
                frames=[f.as_dict() for f in collector],
            )
        except Exception as exc:  # noqa: BLE001 - el error se devuelve al modelo como observacion
            return MCPToolResult(
                tool_name=name,
                arguments=arguments,
                ok=False,
                text="",
                error=_flatten(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
                frames=[f.as_dict() for f in collector],
            )
        finally:
            self._collectors.remove(collector)

    # ---------------------------------------------------------- api publica -
    async def _submit(self, op: str, payload: Any = None, timeout: float | None = None) -> Any:
        if not self.is_alive:
            raise MCPConnectionError(
                f"La conexion {self.conn_id} no esta activa. {self.last_error or ''}".strip()
            )
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._queue.put((op, payload, future))
        return await asyncio.wait_for(future, timeout=timeout or self.config.timeout_s)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPToolResult:
        return await self._submit("call_tool", {"name": name, "arguments": arguments})

    async def refresh_tools(self) -> list[ToolSpec]:
        return await self._submit("list_tools")

    async def list_resources(self) -> list[dict[str, Any]]:
        return await self._submit("list_resources")

    async def list_prompts(self) -> list[dict[str, Any]]:
        return await self._submit("list_prompts")

    async def read_resource(self, uri: str) -> dict[str, Any]:
        return await self._submit("read_resource", {"uri": uri})

    async def ping(self) -> bool:
        return await self._submit("ping", timeout=15)

    def describe(self) -> dict[str, Any]:
        return {
            "conn_id": self.conn_id,
            "config": self.config.public(),
            "alive": self.is_alive,
            "transport": self.active_transport,
            "connected_at": self.connected_at,
            "server_info": self.server_info,
            "last_error": self.last_error,
            "tools": [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in self.tools
            ],
        }


class _PreopenedTransport:
    """Adapta unos streams ya abiertos (y ya interceptados) al protocolo Transport.

    `Client` espera un context manager que produzca los streams. Como aqui el
    transporte real ya esta abierto -para poder envolverlo con el tee-, este
    adaptador se limita a devolverlos y a no cerrarlos: de eso se encarga el
    `async with` del transporte original.
    """

    def __init__(self, read: Any, write: Any) -> None:
        self._streams = (read, write)

    async def __aenter__(self) -> tuple[Any, Any]:
        return self._streams

    async def __aexit__(self, *exc: object) -> None:
        return None


def _flatten(exc: BaseException, depth: int = 0) -> str:
    """Aplana los ExceptionGroup de anyio a un mensaje legible."""
    if depth > 4:
        return f"{type(exc).__name__}: {exc}"
    inner = getattr(exc, "exceptions", None)
    if inner:
        return " ; ".join(_flatten(e, depth + 1) for e in inner)
    return f"{type(exc).__name__}: {exc}"
