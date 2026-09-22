"""Proveedor generico para cualquier servidor con API compatible OpenAI.

Sirve para llama.cpp (`llama-server`), LM Studio, vLLM, TGI, LocalAI, etc.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from .base import (
    LLMProvider,
    LLMResponse,
    ToolCall,
    ToolSpec,
    Usage,
    coerce_arguments,
    split_thinking,
)

logger = logging.getLogger(__name__)


class LLMRequestError(RuntimeError):
    """Rechazo del proveedor con el motivo real, no solo el codigo HTTP.

    Guarda tambien las claves del payload enviado: cuando un proveedor se
    queja de un parametro, saber que se le mando ahorra media depuracion.
    """

    def __init__(self, status: int, detail: str, payload: dict[str, Any] | None = None) -> None:
        self.status = status
        self.detail = detail
        self.payload_keys = sorted(payload) if payload else []
        # Sin payload el error viene de listar modelos, no de una peticion a un modelo.
        where = f" de '{payload.get('model', '?')}'" if payload else ""
        hint = ""
        if status == 404:
            hint = (
                " — ese identificador no existe en tu cuenta. Pulsa 'Validar' en la "
                "pestana Modelo para ver los que si estan disponibles."
            )
        super().__init__(f"HTTP {status}{where}: {detail}{hint}")


def _error_text(response: httpx.Response) -> str:
    """Extrae el mensaje de error del cuerpo, venga como venga."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:300]
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or error)[:300]
    # Formato de la API de Cloudflare: {"errors": [{"code": 4006, "message": "..."}]}.
    errors = body.get("errors") if isinstance(body, dict) else None
    if not error and isinstance(errors, list) and errors:
        parts = [
            f"{e.get('message', e)} (codigo {e['code']})" if isinstance(e, dict) and e.get("code")
            else str(e.get("message", e) if isinstance(e, dict) else e)
            for e in errors
        ]
        return "; ".join(parts)[:300]
    # Formato "problem details" (NVIDIA): {"status": 403, "title": "Forbidden", "detail": "..."}.
    if not error and isinstance(body, dict) and isinstance(body.get("detail"), str):
        title = body.get("title")
        return (f"{title}: {body['detail']}" if title else body["detail"])[:300]
    return str(error or body)[:300]


class OpenAICompatProvider(LLMProvider):
    name = "openai_compat"
    # La mayoria de servidores locales se anuncian como http://host:puerto y
    # cuelgan la API de /v1. Los que ya traen su ruta completa (Gemini) ponen
    # esto a False.
    appends_v1 = True
    # Que poner en `content` de un turno del asistente que solo trae
    # tool_calls. Los servidores locales suelen exigir `null`; Gemini lo
    # rechaza y quiere que la clave no aparezca. "null" | "omit" | "empty".
    empty_assistant_content = "null"
    # Gemini tampoco admite el campo `name` en los mensajes de rol `tool`.
    tool_message_includes_name = True
    # Clave de `extra_content` que este proveedor emite y espera de vuelta en
    # cada tool_call (Gemini: "google", con su `thought_signature` dentro).
    # None = el proveedor no usa esa extension y no se le manda nunca: OpenAI y
    # los motores locales devuelven 400 ante un campo que no conocen.
    extra_content_key: str | None = None

    def __init__(self, base_url: str, model: str, **options: Any) -> None:
        super().__init__(base_url, model, **options)
        root = self.base_url
        if self.appends_v1 and not root.endswith("/v1"):
            root = f"{root}/v1"
        headers = {"Content-Type": "application/json"}
        api_key = options.get("api_key") or "not-needed"
        headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=root,
            timeout=float(options.get("request_timeout", 600.0)),
            headers=headers,
        )
        # None = todavia no se sabe si el motor acepta `chat_template_kwargs`
        # (el parametro con el que se activa o desactiva el razonamiento).
        self._accepts_template_kwargs: bool | None = None

    def _serialize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            role = msg["role"]
            if role == "tool":
                block: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                }
                if self.tool_message_includes_name:
                    block["name"] = msg.get("name", "")
                out.append(block)
                continue
            item: dict[str, Any] = {"role": role, "content": msg.get("content", "") or ""}
            calls = msg.get("tool_calls") or []
            if calls:
                item["tool_calls"] = [self._serialize_tool_call(c) for c in calls]
                if not item["content"]:
                    # Cada servidor tiene su manía con el content vacío.
                    if self.empty_assistant_content == "null":
                        item["content"] = None
                    elif self.empty_assistant_content == "omit":
                        item.pop("content")
            out.append(item)
        return out

    def _serialize_tool_call(self, call: dict[str, Any]) -> dict[str, Any]:
        """Reconstruye un tool_call del asistente para devolverselo al modelo.

        `extra_content` se reenvia tal cual llego. Es lo que necesita Gemini 3:
        firma cada functionCall con un `thought_signature` y, si el turno vuelve
        sin el, responde 400 ("Function call is missing a thought_signature").
        Solo se manda al proveedor que lo emitio, porque para el resto es un
        campo desconocido.
        """
        block: dict[str, Any] = {
            "id": call["id"],
            "type": "function",
            "function": {
                "name": call["name"],
                "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
            },
        }
        if self.extra_content_key:
            extra_content = (call.get("extra") or {}).get("extra_content")
            if isinstance(extra_content, dict) and extra_content.get(self.extra_content_key):
                block["extra_content"] = {
                    self.extra_content_key: extra_content[self.extra_content_key]
                }
        return block

    def _tool_call_extra(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Guarda los campos del proveedor que habra que devolverle luego."""
        if not self.extra_content_key:
            return {}
        extra_content = raw.get("extra_content")
        if isinstance(extra_content, dict) and extra_content.get(self.extra_content_key):
            return {
                "extra_content": {self.extra_content_key: extra_content[self.extra_content_key]}
            }
        return {}

    def _prepare_tools(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        """Traduce las herramientas MCP al formato del proveedor."""
        return [t.to_openai() for t in tools]

    def _adapt(self, payload: dict[str, Any], thinking: bool | None) -> dict[str, Any]:
        """Traslada el interruptor de razonamiento al dialecto del motor local.

        llama.cpp, vLLM, LM Studio y SGLang no tienen un parametro propio: le
        pasan variables a la plantilla de chat del modelo, y la que usan los
        modelos hibridos (qwen3 y derivados) es `enable_thinking`. Si el motor
        la rechaza, `_post` la retira y reintenta una sola vez.
        """
        if thinking is not None and self._accepts_template_kwargs is not False:
            kwargs = dict(payload.get("chat_template_kwargs") or {})
            kwargs["enable_thinking"] = bool(thinking)
            payload["chat_template_kwargs"] = kwargs
        return payload

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        """Envia la peticion y reintenta sin `chat_template_kwargs` si molesta."""
        response = await self._client.post("/chat/completions", json=payload)
        if response.status_code != 400 or "chat_template_kwargs" not in payload:
            if response.status_code < 400 and "chat_template_kwargs" in payload:
                self._accepts_template_kwargs = True
            return response

        detail = response.text.lower()
        if "chat_template_kwargs" not in detail and "enable_thinking" not in detail:
            return response

        self._accepts_template_kwargs = False
        payload.pop("chat_template_kwargs", None)
        logger.info(
            "%s: el motor no acepta chat_template_kwargs; se sigue sin controlar el razonamiento",
            self.name,
        )
        return await self._client.post("/chat/completions", json=payload)

    async def model_features(self, model: str | None = None) -> dict[str, Any]:
        """El motor no publica capacidades: se deja el control al usuario."""
        return {
            "thinking": "toggle",
            "thinking_supported": True,
            "can_disable_thinking": True,
            "tools": True,
            "capabilities": [],
            "source": "desconocido",
        }

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
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": self._serialize_messages(messages),
            "temperature": temperature if temperature is not None else self.options.get("temperature", 0.6),
            "top_p": self.options.get("top_p", 0.95),
            "max_tokens": max_tokens or self.options.get("max_tokens", 4096),
            "stream": False,
        }
        if tools:
            payload["tools"] = self._prepare_tools(tools)
            payload["tool_choice"] = "auto"

        # Puntos de extension para las subclases de nube, que ajustan los
        # parametros al dialecto de cada proveedor y reintentan si hace falta.
        payload = self._adapt(payload, thinking)

        started = time.perf_counter()
        resp = await self._post(payload)
        if resp.status_code >= 400:
            # `raise_for_status` solo deja el codigo, y con un 400 el codigo no
            # dice nada: el motivo ("Unknown name X", "model not found") va en
            # el cuerpo. Sin esto, depurar un rechazo del proveedor es adivinar.
            logger.warning(
                "%s rechazo la peticion (%s) | modelo=%s roles=%s herramientas=%s params=%s",
                self.name,
                resp.status_code,
                payload.get("model"),
                [m["role"] for m in payload.get("messages", [])],
                [t["function"]["name"] for t in payload.get("tools", [])],
                sorted(k for k in payload if k not in {"messages", "tools"}),
            )
            raise LLMRequestError(
                resp.status_code, self._explain_error(resp.status_code, _error_text(resp)), payload
            )
        data = resp.json()
        latency_ms = (time.perf_counter() - started) * 1000

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        # `reasoning_content` / `reasoning`: convencion de vLLM, SGLang y llama.cpp.
        thoughts = message.get("reasoning_content") or message.get("reasoning") or ""
        if not thoughts:
            content, thoughts = split_thinking(content)

        calls = [
            ToolCall(
                id=c.get("id") or f"call_{idx}",
                name=(c.get("function") or {}).get("name", ""),
                arguments=coerce_arguments((c.get("function") or {}).get("arguments")),
                extra=self._tool_call_extra(c),
            )
            for idx, c in enumerate(message.get("tool_calls") or [])
        ]

        usage_raw = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
            completion_tokens=int(usage_raw.get("completion_tokens") or 0),
            total_tokens=int(usage_raw.get("total_tokens") or 0),
        )
        if not usage.total_tokens:
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

        return LLMResponse(
            content=content.strip(),
            thinking=thoughts.strip(),
            tool_calls=calls,
            usage=usage,
            finish_reason=choice.get("finish_reason") or "stop",
            latency_ms=latency_ms,
            model=data.get("model") or payload["model"],
            raw={},
        )

    def _explain_error(self, status: int, detail: str) -> str:
        """Gancho para traducir un rechazo del proveedor a algo accionable."""
        return detail

    def _filter_models(self, ids: list[str]) -> list[str]:
        """Gancho para quedarse solo con los modelos utiles del proveedor."""
        return sorted(ids)

    async def list_models(self) -> list[str]:
        resp = await self._client.get("/models", timeout=15.0)
        if resp.status_code >= 400:
            # El motivo real ("Please pass a valid API key") va en el cuerpo;
            # `raise_for_status` solo dejaria el codigo, que no dice nada.
            raise LLMRequestError(resp.status_code, _error_text(resp))
        raw = [m.get("id", "") for m in resp.json().get("data", [])]
        return self._filter_models([m for m in raw if m])

    async def health(self) -> dict[str, Any]:
        try:
            models = await self.list_models()
            return {
                "ok": True,
                "provider": self.name,
                "base_url": self.base_url,
                "model": self.model,
                "model_available": self.model in models,
                "models": models,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "provider": self.name,
                "base_url": self.base_url,
                "model": self.model,
                "error": f"{type(exc).__name__}: {exc}",
            }

    async def aclose(self) -> None:
        await self._client.aclose()
