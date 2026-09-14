"""Proveedor para Ollama (motor por defecto en local)."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
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


class OllamaProvider(LLMProvider):
    name = "ollama"
    supports_pull = True

    def __init__(self, base_url: str, model: str, **options: Any) -> None:
        super().__init__(base_url, model, **options)
        self._timeout = float(options.get("request_timeout", 600.0))
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout)
        # Mantener el modelo en RAM entre mensajes: en CPU la carga cuesta segundos.
        self._keep_alive = options.get("keep_alive", "30m")
        # Soporte de razonamiento por modelo. Se resuelve con `/api/show` y, si
        # ese endpoint no lo dice, con lo que responda `/api/chat`.
        #   True  -> acepta `think`
        #   False -> lo rechaza; hay que omitir el parametro
        #   ausente/None -> todavia no se sabe
        self._think_support: dict[str, bool] = {}
        self._show_cache: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ utils
    def _serialize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            role = msg["role"]
            if role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "content": msg.get("content", ""),
                        "tool_name": msg.get("name", ""),
                    }
                )
                continue
            item: dict[str, Any] = {"role": role, "content": msg.get("content", "") or ""}
            calls = msg.get("tool_calls") or []
            if calls:
                item["tool_calls"] = [
                    {"function": {"name": c["name"], "arguments": c.get("arguments", {})}}
                    for c in calls
                ]
            out.append(item)
        return out

    async def _show(self, model: str) -> dict[str, Any]:
        """`/api/show`: metadatos del modelo instalado (incluye capabilities)."""
        if model in self._show_cache:
            return self._show_cache[model]
        try:
            resp = await self._client.post("/api/show", json={"model": model}, timeout=20.0)
            resp.raise_for_status()
            data = resp.json()
        except Exception:  # noqa: BLE001 - un motor viejo o un modelo ausente no rompe nada
            data = {}
        self._show_cache[model] = data
        return data

    async def _thinking_supported(self, model: str) -> bool | None:
        """True/False si se sabe; None si el motor no lo declara.

        Ollama lista las capacidades del modelo en `/api/show` desde la 0.9;
        `thinking` aparece en los modelos hibridos (qwen3, deepseek-r1,
        gpt-oss, magistral...). Preguntarlo es lo que hace que el interruptor
        de razonamiento de la UI funcione con cualquier modelo descargado, no
        solo con los del catalogo.
        """
        if model in self._think_support:
            return self._think_support[model]
        capabilities = (await self._show(model)).get("capabilities")
        if not isinstance(capabilities, list):
            return None
        supported = "thinking" in capabilities
        self._think_support[model] = supported
        return supported

    async def model_features(self, model: str | None = None) -> dict[str, Any]:
        target = model or self.model
        data = await self._show(target)
        capabilities = data.get("capabilities")
        if not isinstance(capabilities, list):
            # Motor antiguo sin `capabilities`: se cae al catalogo estatico.
            return await super().model_features(target)
        thinks = "thinking" in capabilities
        return {
            "thinking": "toggle" if thinks else "none",
            "thinking_supported": thinks,
            "can_disable_thinking": thinks,
            "tools": "tools" in capabilities,
            "capabilities": capabilities,
            "source": "engine",
        }

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        thinking: bool | None,
        think_supported: bool | None = None,
    ) -> dict[str, Any]:
        opts = {
            "temperature": temperature if temperature is not None else self.options.get("temperature", 0.6),
            "top_p": self.options.get("top_p", 0.95),
            "num_ctx": self.options.get("num_ctx", 16384),
            "num_predict": max_tokens or self.options.get("max_tokens", 4096),
        }
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": self._serialize_messages(messages),
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": opts,
        }
        if tools:
            payload["tools"] = [t.to_openai() for t in tools]

        # `think` se manda SIEMPRE que el modelo lo acepte, tambien cuando vale
        # False. Omitirlo no desactiva el razonamiento: modelos como qwen3
        # razonan por defecto, asi que sin el parametro explicito el
        # interruptor de la UI solo funcionaria en un sentido.
        want_think = self.options.get("thinking", True) if thinking is None else thinking
        if think_supported is not False:
            payload["think"] = bool(want_think)
        return payload

    # ------------------------------------------------------------------- api
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
        target = model or self.model
        supported = await self._thinking_supported(target)
        payload = self._build_payload(
            messages, tools, model, temperature, max_tokens, thinking, supported
        )
        started = time.perf_counter()
        resp = await self._client.post("/api/chat", json=payload)

        # Red de seguridad para motores que no declaran capacidades: si el
        # modelo rechaza `think`, se recuerda y se reintenta sin el.
        if resp.status_code == 400 and "think" in payload:
            body = resp.text.lower()
            if "think" in body:
                self._think_support[target] = False
                payload.pop("think", None)
                resp = await self._client.post("/api/chat", json=payload)
        elif resp.status_code == 200 and "think" in payload:
            self._think_support.setdefault(target, True)

        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.perf_counter() - started) * 1000

        message = data.get("message") or {}
        content = message.get("content") or ""
        thoughts = message.get("thinking") or ""
        if not thoughts:
            content, thoughts = split_thinking(content)

        calls = [
            ToolCall(
                name=(c.get("function") or {}).get("name", ""),
                arguments=coerce_arguments((c.get("function") or {}).get("arguments")),
            )
            for c in (message.get("tool_calls") or [])
        ]

        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        completion_tokens = int(data.get("eval_count") or 0)

        return LLMResponse(
            content=content.strip(),
            thinking=thoughts.strip(),
            tool_calls=calls,
            usage=Usage(prompt_tokens, completion_tokens, prompt_tokens + completion_tokens),
            finish_reason=data.get("done_reason") or "stop",
            latency_ms=latency_ms,
            model=data.get("model") or payload["model"],
            raw={
                "total_duration_ms": (data.get("total_duration") or 0) / 1e6,
                "load_duration_ms": (data.get("load_duration") or 0) / 1e6,
                "eval_duration_ms": (data.get("eval_duration") or 0) / 1e6,
            },
        )

    async def list_models(self) -> list[str]:
        resp = await self._client.get("/api/tags", timeout=15.0)
        resp.raise_for_status()
        return sorted(m.get("name", "") for m in resp.json().get("models", []))

    async def pull_model(self, model: str) -> AsyncIterator[dict[str, Any]]:
        """Descarga un modelo del registro de Ollama.

        `/api/pull` responde con NDJSON: una linea por cambio de estado, y
        durante la descarga una linea por capa con `completed` / `total`.
        Sin timeout: un modelo de varios GB puede tardar mucho.
        """
        payload = {"model": model, "stream": True}
        async with self._client.stream(
            "POST", "/api/pull", json=payload, timeout=httpx.Timeout(None)
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                detail = response.text.strip() or f"HTTP {response.status_code}"
                raise httpx.HTTPStatusError(detail, request=response.request, response=response)
            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    async def health(self) -> dict[str, Any]:
        try:
            version = (await self._client.get("/api/version", timeout=5.0)).json()
            models = await self.list_models()
            return {
                "ok": True,
                "provider": self.name,
                "base_url": self.base_url,
                "version": version.get("version"),
                "model": self.model,
                "model_available": self.model in models,
                "models": models,
            }
        except Exception as exc:  # noqa: BLE001 - health nunca debe romper
            return {
                "ok": False,
                "provider": self.name,
                "base_url": self.base_url,
                "model": self.model,
                "error": f"{type(exc).__name__}: {exc}",
            }

    async def aclose(self) -> None:
        await self._client.aclose()
