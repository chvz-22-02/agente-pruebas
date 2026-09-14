"""Intercepta las tramas JSON-RPC que cruzan entre el agente y el servidor MCP.

El SDK de MCP entrega dos streams (lectura/escritura). Envolviendolos podemos
registrar el protocolo real sin tocar la libreria ni el servidor remoto.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

FrameSink = Callable[[str, dict[str, Any]], None]


def _dump(item: Any) -> dict[str, Any] | None:
    """Convierte un SessionMessage en dict JSON. Nunca lanza."""
    try:
        if isinstance(item, BaseException):
            return {"_transport_error": f"{type(item).__name__}: {item}"}
        message = getattr(item, "message", item)
        dump = getattr(message, "model_dump", None)
        if dump is None:
            return None
        return dump(by_alias=True, exclude_none=True, mode="json")
    except Exception:  # noqa: BLE001 - la observabilidad jamas rompe el transporte
        return None


class TeeReceiveStream:
    """Envuelve el stream de lectura (servidor MCP -> agente)."""

    def __init__(self, inner: Any, sink: FrameSink) -> None:
        self._inner = inner
        self._sink = sink

    async def __aenter__(self) -> TeeReceiveStream:
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> Any:
        return await self._inner.__aexit__(*exc)

    def __aiter__(self) -> TeeReceiveStream:
        return self

    async def __anext__(self) -> Any:
        item = await self._inner.__anext__()
        payload = _dump(item)
        if payload is not None:
            self._sink("mcp->agent", payload)
        return item

    async def receive(self) -> Any:
        item = await self._inner.receive()
        payload = _dump(item)
        if payload is not None:
            self._sink("mcp->agent", payload)
        return item

    async def aclose(self) -> None:
        await self._inner.aclose()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class TeeSendStream:
    """Envuelve el stream de escritura (agente -> servidor MCP)."""

    def __init__(self, inner: Any, sink: FrameSink) -> None:
        self._inner = inner
        self._sink = sink

    async def __aenter__(self) -> TeeSendStream:
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> Any:
        return await self._inner.__aexit__(*exc)

    async def send(self, item: Any) -> None:
        payload = _dump(item)
        if payload is not None:
            self._sink("agent->mcp", payload)
        await self._inner.send(item)

    async def aclose(self) -> None:
        await self._inner.aclose()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
