"""Modelos de datos del subsistema MCP."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

Transport = Literal["auto", "streamable_http", "sse"]


@dataclass(slots=True)
class MCPServerConfig:
    """Configuracion que llega desde la UI. Nada de esto se hardcodea."""

    url: str
    name: str = ""
    transport: Transport = "auto"
    headers: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 60.0
    capture_raw: bool = True

    def public(self) -> dict[str, Any]:
        """Version segura para logs y UI: oculta credenciales en cabeceras."""
        redacted = {
            k: ("***" if k.lower() in {"authorization", "x-api-key", "api-key", "cookie"} else v)
            for k, v in self.headers.items()
        }
        return {
            "url": self.url,
            "name": self.name,
            "transport": self.transport,
            "headers": redacted,
            "timeout_s": self.timeout_s,
        }


@dataclass(slots=True)
class RawFrame:
    """Trama JSON-RPC intercambiada con el servidor MCP."""

    direction: Literal["agent->mcp", "mcp->agent"]
    payload: dict[str, Any]
    ts: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {"direction": self.direction, "payload": self.payload, "ts": self.ts}


@dataclass(slots=True)
class MCPToolResult:
    """Resultado normalizado de una invocacion de herramienta."""

    tool_name: str
    arguments: dict[str, Any]
    ok: bool
    text: str
    structured: Any = None
    content_blocks: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    latency_ms: float = 0.0
    call_id: str = field(default_factory=lambda: f"mcp_{uuid.uuid4().hex[:12]}")
    frames: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "ok": self.ok,
            "text": self.text,
            "structured": self.structured,
            "content_blocks": self.content_blocks,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "frames": self.frames,
        }
