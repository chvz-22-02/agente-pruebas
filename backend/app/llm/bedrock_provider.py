"""Amazon Bedrock a traves de su API nativa Converse.

Bedrock publica tambien un endpoint compatible con OpenAI
(`/openai/v1/chat/completions`), que habria encajado sin escribir nada: es lo
que se hace con Gemini. No sirve aqui. Su matriz de compatibilidad deja fuera
a Claude, a Nova y a Llama -justo las familias por las que se usa Bedrock- y
solo admite OpenAI, DeepSeek, Qwen, Mistral Large 3 y companhia. Converse es la
unica superficie con una forma unica de tool calling para todos los modelos de
chat del servicio, y sin herramientas no se puede probar un MCP.

Autenticacion: clave de API de Bedrock (`AWS_BEARER_TOKEN_BEDROCK`) como
`Authorization: Bearer`, sin firma SigV4 ni boto3. La region va dentro del
host, asi que viaja en la URL base con el mismo marcador que la cuenta de
Cloudflare.

Notas de la API que condicionan este codigo:

* El `system` va en un campo propio, no como un mensaje mas.
* Los `toolResult` van dentro de un mensaje de usuario, y todos los de un
  mismo turno en el mismo mensaje: repartirlos le ensena al modelo a dejar de
  pedir herramientas en paralelo (el mismo motivo que en Claude).
* Muchos modelos rechazan el identificador base y exigen el del perfil de
  inferencia (`us.anthropic...`). El 400 lo explica, pero en jerga de AWS, asi
  que se traduce en `_explain_error`.
* El razonamiento no tiene un campo comun: va en `additionalModelRequestFields`
  y cada familia lo deletrea a su manera (ver `_reasoning`). Una clave que el
  esquema del modelo no conoce es un 400, asi que solo se manda donde esta
  documentado y, si aun asi lo rechazan, se retira y se recuerda.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any
from urllib.parse import quote

import httpx

from .base import LLMProvider, LLMResponse, ToolCall, ToolSpec, Usage
from .catalog import model_info

logger = logging.getLogger(__name__)

# La region ocupa en la URL el mismo hueco que la cuenta de Cloudflare, para
# reutilizar el campo que la UI ya sabe pedir y rellenar.
REGION_PLACEHOLDER = "{account_id}"
REGION_ENV = "AWS_REGION"

# Prefijos de los perfiles de inferencia entre regiones. No son universales:
# hay modelos que solo se invocan por su identificador pelado y solo en su
# region, por eso el desplegable se rellena con lo que diga la cuenta.
GEO_PREFIXES = ("us.", "eu.", "apac.", "us-gov.", "global.")

# Paginacion del plano de control al listar perfiles de inferencia.
CONTROL_PAGE_SIZE = 100
CONTROL_MAX_PAGES = 20


def _family(model: str) -> str:
    """Identificador sin el prefijo de region: `us.anthropic.x` -> `anthropic.x`."""
    for prefix in GEO_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


class MissingRegion(ValueError):
    """La URL de Bedrock necesita la region y no ha llegado ninguna."""

    def __init__(self) -> None:
        super().__init__(
            "AWS necesita la region de Bedrock (p.ej. us-east-1). Escribela en la pestana "
            f"Modelo o define {REGION_ENV} en backend/.env. Tiene que ser la misma region en "
            "la que generaste la clave."
        )


class BedrockRequestError(RuntimeError):
    """Rechazo de Bedrock con el motivo real, no solo el codigo HTTP."""

    def __init__(self, status: int, detail: str, model: str = "") -> None:
        self.status = status
        self.detail = detail
        where = f" de '{model}'" if model else ""
        super().__init__(f"HTTP {status}{where}: {detail}")


def _error_text(response: httpx.Response) -> str:
    """El cuerpo de error de AWS es {"message": "..."}; el tipo va en cabecera."""
    kind = response.headers.get("x-amzn-errortype", "").split(":")[0]
    try:
        body = response.json()
    except ValueError:
        return (kind or response.text[:300]).strip()
    message = body.get("message") or body.get("Message") if isinstance(body, dict) else None
    detail = str(message or body)[:300]
    return f"{kind}: {detail}" if kind and kind not in detail else detail


class BedrockProvider(LLMProvider):
    name = "aws"

    def __init__(self, base_url: str, model: str, **options: Any) -> None:
        region = (options.get("account_id") or os.environ.get(REGION_ENV, "")).strip()
        if REGION_PLACEHOLDER in base_url and region:
            base_url = base_url.replace(REGION_PLACEHOLDER, region)
        super().__init__(base_url, model, **options)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=float(options.get("request_timeout", 600.0)),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {options.get('api_key') or ''}",
            },
        )
        # Ajustes aprendidos de los rechazos de la API, como en los demas
        # proveedores de nube: mas vale perder el interruptor que la peticion.
        self._drop_reasoning = False
        self._drop_sampling = False

    @property
    def has_region(self) -> bool:
        return REGION_PLACEHOLDER not in self.base_url

    def _require_region(self) -> None:
        if not self.has_region:
            raise MissingRegion()

    @property
    def region(self) -> str:
        """Region deducida del host, para componer la URL del plano de control."""
        host = self.base_url.split("//", 1)[-1]
        parts = host.split(".")
        return parts[1] if len(parts) > 2 else ""

    # ------------------------------------------------------------- mensajes -
    @staticmethod
    def _split_system(messages: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
        system = [m.get("content", "") for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]
        return [{"text": text} for text in system if text], rest

    def _serialize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []

        for msg in messages:
            role = msg["role"]

            if role == "tool":
                block = {
                    "toolResult": {
                        "toolUseId": msg.get("tool_call_id", ""),
                        "content": [{"text": msg.get("content", "") or "(sin contenido)"}],
                    }
                }
                # Se acumulan en el ultimo mensaje de usuario: todos los
                # resultados de un turno tienen que ir juntos.
                if out and out[-1]["role"] == "user":
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
                continue

            if role == "assistant":
                # Si tenemos los bloques tal y como los devolvio el modelo
                # (incluido el razonamiento firmado), se devuelven intactos.
                raw = msg.get("_provider_blocks")
                if raw:
                    out.append({"role": "assistant", "content": raw})
                    continue

                blocks: list[dict[str, Any]] = []
                if msg.get("content"):
                    blocks.append({"text": msg["content"]})
                for call in msg.get("tool_calls") or []:
                    blocks.append(
                        {
                            "toolUse": {
                                "toolUseId": call["id"],
                                "name": call["name"],
                                "input": call.get("arguments") or {},
                            }
                        }
                    )
                # Converse rechaza un mensaje con la lista de contenido vacia.
                out.append({"role": "assistant", "content": blocks or [{"text": "..."}]})
                continue

            text = msg.get("content", "") or ""
            if out and out[-1]["role"] == "user":
                out[-1]["content"].append({"text": text})
            else:
                out.append({"role": "user", "content": [{"text": text}]})

        return out

    @staticmethod
    def _prepare_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        """Las herramientas MCP en el formato de Converse."""
        return [
            {
                "toolSpec": {
                    "name": tool.name,
                    "description": (tool.description or tool.name)[:1024],
                    "inputSchema": {"json": tool.input_schema or {"type": "object", "properties": {}}},
                }
            }
            for tool in tools
        ]

    # --------------------------------------------------------- razonamiento -
    def _reasoning(self, model: str, thinking: bool | None) -> dict[str, Any]:
        """Traduce el interruptor de la UI al dialecto de cada familia.

        Converse no tiene un campo comun para esto: viaja en
        `additionalModelRequestFields` y el esquema lo valida contra el modelo,
        asi que una clave inventada es un 400. Solo se manda donde AWS lo
        documenta:

        * Anthropic: `thinking` con `adaptive` / `disabled` (los Claude
          anteriores a 4.6 pedian `enabled` + `budget_tokens`, que los nuevos
          ya rechazan; si protestan, `_post` retira el campo).
        * Amazon Nova: `reasoningConfig`, apagado por defecto.
        * DeepSeek razona siempre y no lo expone; gpt-oss y Qwen no lo
          documentan. A esos no se les manda nada.
        """
        if self._drop_reasoning:
            return {}
        info = model_info(self.name, model)
        if info is not None and info.thinking == "none":
            return {}

        want = self.options.get("thinking", True) if thinking is None else thinking
        family = _family(model)
        if family.startswith("anthropic."):
            return {"thinking": {"type": "adaptive" if want else "disabled"}}
        if family.startswith("amazon.nova"):
            return {"reasoningConfig": {"type": "enabled" if want else "disabled"}}
        return {}

    # ------------------------------------------------------------------ api -
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
        self._require_region()
        target = model or self.model
        system, conversation = self._split_system(messages)

        inference: dict[str, Any] = {
            "maxTokens": max_tokens or self.options.get("max_tokens") or 4096,
        }
        if not self._drop_sampling:
            temp = temperature if temperature is not None else self.options.get("temperature")
            if temp is not None:
                inference["temperature"] = temp
            top_p = self.options.get("top_p")
            if top_p is not None:
                inference["topP"] = top_p

        body: dict[str, Any] = {
            "messages": self._serialize_messages(conversation),
            "inferenceConfig": inference,
        }
        if system:
            body["system"] = system
        if tools:
            body["toolConfig"] = {"tools": self._prepare_tools(tools), "toolChoice": {"auto": {}}}
        reasoning = self._reasoning(target, thinking)
        if reasoning:
            body["additionalModelRequestFields"] = reasoning

        started = time.perf_counter()
        response = await self._post(target, body)
        latency_ms = (time.perf_counter() - started) * 1000

        if response.status_code >= 400:
            logger.warning(
                "aws rechazo la peticion (%s) | modelo=%s roles=%s herramientas=%s",
                response.status_code,
                target,
                [m["role"] for m in body["messages"]],
                [t["toolSpec"]["name"] for t in body.get("toolConfig", {}).get("tools", [])],
            )
            raise BedrockRequestError(
                response.status_code,
                self._explain_error(response.status_code, _error_text(response)),
                target,
            )

        data = response.json()
        message = (data.get("output") or {}).get("message") or {}
        blocks = message.get("content") or []

        text_parts: list[str] = []
        thoughts: list[str] = []
        calls: list[ToolCall] = []
        for block in blocks:
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block:
                use = block["toolUse"]
                calls.append(
                    ToolCall(
                        id=use.get("toolUseId", ""),
                        name=use.get("name", ""),
                        arguments=dict(use.get("input") or {}),
                    )
                )
            elif "reasoningContent" in block:
                reasoning = (block["reasoningContent"] or {}).get("reasoningText") or {}
                if reasoning.get("text"):
                    thoughts.append(reasoning["text"])

        raw_usage = data.get("usage") or {}
        prompt_tokens = int(raw_usage.get("inputTokens") or 0)
        completion_tokens = int(raw_usage.get("outputTokens") or 0)
        total = int(raw_usage.get("totalTokens") or 0) or prompt_tokens + completion_tokens

        return LLMResponse(
            content="\n".join(text_parts).strip(),
            thinking="\n".join(thoughts).strip(),
            tool_calls=calls,
            usage=Usage(prompt_tokens, completion_tokens, total),
            finish_reason=data.get("stopReason") or "stop",
            latency_ms=latency_ms,
            model=target,
            # Converse devuelve el razonamiento firmado; hay que devolverselo
            # intacto junto con los resultados de las herramientas.
            raw_blocks=blocks,
        )

    async def _post(self, model: str, body: dict[str, Any]) -> httpx.Response:
        """Envia la peticion y reintenta una vez si rechazan un parametro.

        Es la misma red de seguridad que con OpenAI y Gemini: en vez de
        mantener una tabla de que modelo acepta que -que envejece mal- se
        manda lo documentado y, si la API se queja de un campo concreto, se
        retira y se recuerda para no pagar el 400 en cada peticion.
        """
        # Los identificadores llevan version tras dos puntos
        # (`openai.gpt-oss-120b-1:0`), que en la ruta va escapado.
        path = f"/model/{quote(model, safe='')}/converse"
        response = await self._client.post(path, json=body)
        if response.status_code != 400:
            return response

        detail = response.text.lower()
        retried = False
        if "additionalModelRequestFields" in body and any(
            token in detail for token in ("thinking", "reasoning", "additionalmodelrequestfields")
        ):
            body.pop("additionalModelRequestFields")
            self._drop_reasoning = True
            retried = True
        if any(token in detail for token in ("temperature", "topp", "top_p")):
            body["inferenceConfig"].pop("temperature", None)
            body["inferenceConfig"].pop("topP", None)
            self._drop_sampling = True
            retried = True

        if not retried:
            return response
        logger.info("aws: reintento tras parametro rechazado por %s", model)
        return await self._client.post(path, json=body)

    # -------------------------------------------------------------- modelos -
    def _control_url(self, path: str) -> str:
        """El listado vive en el plano de control, otro host que la inferencia."""
        return self.base_url.replace("//bedrock-runtime.", "//bedrock.", 1) + path

    async def _control_get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        response = await self._client.get(self._control_url(path), params=params, timeout=20.0)
        if response.status_code >= 400:
            raise BedrockRequestError(
                response.status_code,
                self._explain_listing_error(response.status_code, _error_text(response)),
            )
        return response.json()

    async def list_models(self) -> list[str]:
        """Perfiles de inferencia y modelos bajo demanda de la cuenta.

        Hacen falta los dos: el prefijo de region (`us.`) no es universal, y
        hay modelos que solo se invocan por su identificador pelado.
        """
        self._require_region()
        profiles: list[str] = []
        token: str | None = None
        for _ in range(CONTROL_MAX_PAGES):
            params: dict[str, Any] = {"maxResults": CONTROL_PAGE_SIZE, "type": "SYSTEM_DEFINED"}
            if token:
                params["nextToken"] = token
            page = await self._control_get("/inference-profiles", params)
            for row in page.get("inferenceProfileSummaries") or []:
                if row.get("inferenceProfileId") and row.get("status", "ACTIVE") == "ACTIVE":
                    profiles.append(row["inferenceProfileId"])
            token = page.get("nextToken")
            if not token:
                break

        # `ListFoundationModels` devuelve tambien identificadores retirados que
        # siguen ahi por compatibilidad: se filtran por ciclo de vida.
        catalogue = await self._control_get(
            "/foundation-models", {"byOutputModality": "TEXT", "byInferenceType": "ON_DEMAND"}
        )
        on_demand = [
            row["modelId"]
            for row in catalogue.get("modelSummaries") or []
            if row.get("modelId")
            and (row.get("modelLifecycle") or {}).get("status", "ACTIVE") == "ACTIVE"
        ]
        return sorted({*profiles, *on_demand})

    async def health(self) -> dict[str, Any]:
        if not self.options.get("api_key"):
            return {
                "ok": False,
                "provider": self.name,
                "base_url": self.base_url,
                "model": self.model,
                "needs_api_key": True,
                "error": "Falta la clave de API. Introducela en la pestana Modelo.",
            }
        if not self.has_region:
            return {
                "ok": False,
                "provider": self.name,
                "base_url": self.base_url,
                "model": self.model,
                "needs_api_key": True,
                "error": str(MissingRegion()),
            }
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
        except Exception as exc:  # noqa: BLE001 - health nunca debe romper
            return {
                "ok": False,
                "provider": self.name,
                "base_url": self.base_url,
                "model": self.model,
                "needs_api_key": True,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _explain_listing_error(self, status: int, detail: str) -> str:
        if status == 403:
            return (
                f"{detail} — o la clave no vale para esta region, o le falta permiso para listar "
                "(bedrock:ListInferenceProfiles y bedrock:ListFoundationModels). Si es lo segundo, "
                "la inferencia funciona igual: escribe el identificador del modelo a mano."
            )
        return detail

    def _explain_error(self, status: int, detail: str) -> str:
        lowered = detail.lower()
        if "on-demand throughput isn" in lowered or "inference profile" in lowered:
            return (
                f"{detail} — ese modelo no se invoca por su identificador base: necesita el del "
                "perfil de inferencia, con prefijo de region (us., eu., apac.). Pulsa 'Validar' "
                "para ver los que acepta tu cuenta."
            )
        if status in {401, 403}:
            return (
                f"{detail} — revisa la clave de Bedrock y que sea de esta misma region. Si el "
                "modelo es de Anthropic, la cuenta necesita ademas haber aceptado una vez su "
                "formulario de acceso en la consola de Bedrock."
            )
        if status == 404:
            return (
                f"{detail} — ese identificador no existe en esta region. Pulsa 'Validar' en la "
                "pestana Modelo para ver los disponibles."
            )
        return detail

    async def aclose(self) -> None:
        await self._client.aclose()
