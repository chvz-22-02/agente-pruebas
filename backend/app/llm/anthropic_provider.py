"""Proveedor para la API de Anthropic (Claude), sobre el SDK oficial.

Notas de la API que condicionan este codigo:

* `temperature` / `top_p` fueron retirados: los modelos Claude 4.7+ devuelven
  400 si se envian. La profundidad se controla con `output_config.effort`.
* El razonamiento se pide con `thinking={"type": "adaptive"}`; `budget_tokens`
  ya no existe. Claude Fable razona siempre y rechaza el parametro.
* Los resultados de herramientas van como bloques `tool_result` dentro de un
  *unico* mensaje de usuario: repartirlos en varios mensajes le ensena al
  modelo a dejar de pedir herramientas en paralelo.
* Los bloques `thinking` de un turno deben devolverse tal cual junto con los
  `tool_result` de ese mismo turno, por eso se conservan sin tocar.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from .base import LLMProvider, LLMResponse, ToolCall, ToolSpec, Usage
from .catalog import CATALOG, model_info

logger = logging.getLogger(__name__)

# Modelos con fallback por refusal en servidor: si un clasificador rechaza la
# peticion, Anthropic la reencamina a otro modelo en vez de devolver un error.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = ("claude-opus-5", "claude-fable-5")


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, base_url: str, model: str, **options: Any) -> None:
        super().__init__(base_url, model, **options)
        from anthropic import AsyncAnthropic

        kwargs: dict[str, Any] = {
            "api_key": options.get("api_key") or None,
            "timeout": float(options.get("request_timeout", 600.0)),
            "max_retries": 2,
        }
        # base_url solo si el usuario apunta a un gateway propio. Una URL de
        # loopback casi siempre es la del motor local arrastrada al cambiar de
        # proveedor, y mandar ahi el trafico de Claude solo genera confusion.
        if base_url and "anthropic.com" not in base_url and base_url.startswith("http"):
            if not any(h in base_url for h in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]")):
                kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)

    # ------------------------------------------------------------- mensajes -
    def _split_system(self, messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """Claude lleva el system aparte, no como un mensaje mas."""
        system_parts = [m.get("content", "") for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]
        return "\n\n".join(p for p in system_parts if p), rest

    def _serialize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []

        for msg in messages:
            role = msg["role"]

            if role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", "") or "(sin contenido)",
                }
                # Se acumulan en el ultimo mensaje de usuario si ya es de
                # resultados: todos los tool_result de un turno van juntos.
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
                continue

            if role == "assistant":
                # Si tenemos los bloques originales de la respuesta (incluidos
                # los de razonamiento firmados), se devuelven intactos.
                raw = msg.get("_provider_blocks")
                if raw:
                    out.append({"role": "assistant", "content": raw})
                    continue

                blocks: list[dict[str, Any]] = []
                if msg.get("content"):
                    blocks.append({"type": "text", "text": msg["content"]})
                for call in msg.get("tool_calls") or []:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call["id"],
                            "name": call["name"],
                            "input": call.get("arguments", {}),
                        }
                    )
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "..."}]})
                continue

            # Usuario: Claude no admite dos turnos de usuario seguidos.
            text = msg.get("content", "") or ""
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append({"type": "text", "text": text})
            else:
                out.append({"role": "user", "content": [{"type": "text", "text": text}]})

        return out

    # ---------------------------------------------------------- parametros --
    def _thinking_and_effort(self, model: str, thinking: bool | None) -> tuple[Any, str | None]:
        """Traduce el interruptor de razonamiento de la UI.

        Desactivar el razonamiento en Opus 5 tiene un modo de fallo conocido:
        el modelo escribe a veces la llamada a la herramienta como texto en vez
        de emitir un bloque `tool_use`, y en un banco de pruebas de MCP eso es
        justo lo que no queremos. Por eso "sin razonamiento" se traduce a
        esfuerzo bajo manteniendo el razonamiento adaptativo.
        """
        want = self.options.get("thinking", True) if thinking is None else thinking
        info = model_info("anthropic", model)
        mode = info.thinking if info else "adaptive"

        if mode == "always":
            return None, None  # Fable razona siempre y rechaza el parametro.
        if mode == "none":
            return None, None
        if want:
            return {"type": "adaptive", "display": "summarized"}, None
        return {"type": "adaptive", "display": "omitted"}, "low"

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None,
        model: str,
        max_tokens: int | None,
        thinking: bool | None,
    ) -> dict[str, Any]:
        system, conversation = self._split_system(messages)
        think, effort = self._thinking_and_effort(model, thinking)

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens or self.options.get("max_tokens") or 16000,
            "messages": self._serialize_messages(conversation),
        }
        if system:
            kwargs["system"] = system
        if tools:
            # El esquema de herramienta de Claude es plano: nada de {"function": ...}.
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": (t.description or "")[:1024],
                    "input_schema": t.input_schema or {"type": "object", "properties": {}},
                }
                for t in tools
            ]
        if think is not None:
            kwargs["thinking"] = think
        if effort:
            kwargs["output_config"] = {"effort": effort}
        return kwargs

    # ------------------------------------------------------------------ api -
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        *,
        model: str | None = None,
        temperature: float | None = None,  # noqa: ARG002 - Claude ya no lo acepta
        max_tokens: int | None = None,
        thinking: bool | None = None,
    ) -> LLMResponse:
        target = model or self.model
        kwargs = self._build_kwargs(messages, tools, target, max_tokens, thinking)

        use_fallbacks = target.startswith(FALLBACK_MODELS)
        started = time.perf_counter()
        if use_fallbacks:
            message = await self._client.beta.messages.create(
                **kwargs, betas=[FALLBACK_BETA], fallbacks="default"
            )
        else:
            message = await self._client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - started) * 1000

        text_parts: list[str] = []
        thoughts: list[str] = []
        calls: list[ToolCall] = []
        raw_blocks: list[dict[str, Any]] = []

        for block in message.content:
            raw_blocks.append(block.model_dump(exclude_none=True, mode="json"))
            kind = getattr(block, "type", "")
            if kind == "text":
                text_parts.append(block.text)
            elif kind == "thinking":
                if block.thinking:
                    thoughts.append(block.thinking)
            elif kind == "tool_use":
                calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {}))
                )

        # Un rechazo del clasificador llega como 200 con stop_reason "refusal".
        if message.stop_reason == "refusal" and not text_parts:
            detail = getattr(message.stop_details, "explanation", "") or ""
            category = getattr(message.stop_details, "category", "") or "desconocida"
            text_parts.append(
                f"[El modelo declino la peticion. Categoria: {category}. {detail}]".strip()
            )

        usage = message.usage
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        billed_input = prompt_tokens + cache_read + cache_write

        return LLMResponse(
            content="\n".join(text_parts).strip(),
            thinking="\n".join(thoughts).strip(),
            tool_calls=calls,
            usage=Usage(billed_input, completion_tokens, billed_input + completion_tokens),
            finish_reason=message.stop_reason or "stop",
            latency_ms=latency_ms,
            model=message.model or target,
            raw_blocks=raw_blocks,
            raw={"cache_read_tokens": cache_read, "cache_write_tokens": cache_write},
        )

    async def list_models(self) -> list[str]:
        page = await self._client.models.list(limit=100)
        return sorted(m.id for m in page.data)

    async def health(self) -> dict[str, Any]:
        if not self.options.get("api_key"):
            return {
                "ok": False,
                "provider": self.name,
                "model": self.model,
                "needs_api_key": True,
                "error": "Falta la clave de API. Introducela en la pestana Modelo.",
            }
        try:
            models = await self.list_models()
            return {
                "ok": True,
                "provider": self.name,
                "base_url": "api.anthropic.com",
                "model": self.model,
                "model_available": self.model in models,
                "models": models,
            }
        except Exception as exc:  # noqa: BLE001 - health nunca debe romper
            return {
                "ok": False,
                "provider": self.name,
                "model": self.model,
                "needs_api_key": True,
                "error": _readable(exc),
            }

    def pull_model(self, model: str) -> AsyncIterator[dict[str, Any]]:
        raise NotImplementedError("Los modelos de Anthropic se sirven en la nube, no se descargan.")

    async def aclose(self) -> None:
        await self._client.close()


def _readable(exc: Exception) -> str:
    """Mensaje de error util para la UI, sin volcar la traza entera."""
    message = getattr(exc, "message", None) or str(exc)
    status = getattr(exc, "status_code", None)
    if status == 401:
        return "Clave de API invalida o revocada (401)."
    if status == 403:
        return "La clave no tiene permiso para este recurso (403)."
    if status == 429:
        return "Limite de peticiones alcanzado (429). Reintenta en unos segundos."
    return f"{type(exc).__name__}: {message}"


# El catalogo define que modelos ofrece la UI para este proveedor.
assert "anthropic" in CATALOG
