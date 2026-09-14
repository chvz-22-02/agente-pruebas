"""Contrato comun para cualquier proveedor de LLM.

El objetivo es que cambiar de modelo (o de motor de inferencia) sea cambiar
una variable de entorno, no tocar codigo del agente.
"""

from __future__ import annotations

import json
import re
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


@dataclass(slots=True)
class ToolSpec:
    """Herramienta expuesta al modelo (viene del servidor MCP)."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        schema = self.input_schema or {"type": "object", "properties": {}}
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (self.description or "")[:1024],
                "parameters": schema,
            },
        }


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    # Campos propios del proveedor que hay que devolverle intactos en el
    # siguiente turno. Hoy lo usa Gemini 3, que firma cada functionCall con un
    # `thought_signature` y rechaza la peticion si no vuelve. Viaja tambien a
    # SQLite dentro de `tool_calls`, para que sobreviva a recargar el historial.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def merge(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(slots=True)
class LLMResponse:
    content: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    # Bloques de contenido tal y como los devolvio el proveedor. Claude exige
    # que el turno se le devuelva intacto (con sus bloques de razonamiento
    # firmados) al mandarle los resultados de las herramientas.
    raw_blocks: list[Any] | None = None


def split_thinking(text: str) -> tuple[str, str]:
    """Separa el bloque <think>...</think> del contenido visible.

    Necesario para modelos que no exponen el razonamiento en un campo aparte.
    """
    if not text:
        return "", ""
    thoughts = "\n".join(m.strip() for m in THINK_RE.findall(text))
    visible = THINK_RE.sub("", text)
    # Cierre sin apertura: modelos que razonan siempre (qwen3:4b en Ollama,
    # variantes *-thinking-2507) llevan el <think> en la plantilla del prompt,
    # asi que la salida solo trae el razonamiento y el </think>. Pasa incluso
    # pidiendo think=false, que esos modelos no pueden desactivar.
    if not thoughts and THINK_CLOSE_RE.search(visible):
        parts = THINK_CLOSE_RE.split(visible)
        thoughts = "\n".join(p.strip() for p in parts[:-1] if p.strip())
        visible = parts[-1]
    # Bloque de pensamiento sin cerrar (respuesta truncada).
    if not thoughts and THINK_OPEN_RE.search(visible):
        head, _, tail = visible.partition("<think>")
        thoughts = tail.strip()
        visible = head
    return visible.strip(), thoughts.strip()


def coerce_arguments(raw: Any) -> dict[str, Any]:
    """Los modelos locales devuelven los argumentos como dict o como string."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"__raw__": raw}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {}


class LLMProvider(ABC):
    """Interfaz minima que debe cumplir cualquier motor de inferencia."""

    name: str = "base"
    # Solo algunos motores (Ollama) exponen una API para descargar modelos.
    # La UI lo consulta para decidir si ofrece el boton de descarga.
    supports_pull: bool = False

    def __init__(self, base_url: str, model: str, **options: Any) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.options = options

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        thinking: bool | None = None,
    ) -> LLMResponse:
        """Una llamada al modelo. `messages` usa el formato canonico interno."""

    @abstractmethod
    async def list_models(self) -> list[str]:
        """Modelos disponibles en el motor local."""

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Estado del motor: alcanzable, version, modelo cargado."""

    async def model_features(self, model: str | None = None) -> dict[str, Any]:
        """Que sabe hacer el modelo indicado: razonamiento, herramientas, ...

        La respuesta por defecto sale del catalogo estatico. Los motores que
        publican esta informacion (Ollama lo hace en `/api/show`) la
        sobrescriben preguntandole al motor, que es lo unico fiable para un
        modelo recien descargado que todavia no esta en el catalogo.
        """
        from .catalog import model_info

        info = model_info(self.name, model or self.model)
        if info is None:
            # Modelo escrito a mano: no se sabe nada de el, asi que se deja el
            # interruptor en manos del usuario en vez de bloquearlo.
            return {
                "thinking": "unknown",
                "thinking_supported": True,
                "can_disable_thinking": True,
                "tools": True,
                "capabilities": [],
                "source": "desconocido",
            }
        return {
            "thinking": info.thinking,
            "thinking_supported": info.thinking != "none",
            "can_disable_thinking": info.thinking in {"toggle", "adaptive", "effort"},
            "tools": True,
            "capabilities": [],
            "source": "catalog",
        }

    def pull_model(self, model: str) -> AsyncIterator[dict[str, Any]]:
        """Descarga un modelo en el motor local, emitiendo el progreso.

        Devuelve un generador asincrono de dicts con, al menos, `status`; los
        motores que reportan bytes anaden `digest`, `completed` y `total`.
        """
        raise NotImplementedError(
            f"El proveedor '{self.name}' no permite descargar modelos desde su API. "
            "Descarga el modelo con las herramientas propias del motor."
        )

    async def aclose(self) -> None:  # pragma: no cover - opcional
        return None
