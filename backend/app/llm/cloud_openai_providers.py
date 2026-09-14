"""Proveedores de nube que hablan el dialecto OpenAI: OpenAI y Google Gemini.

Gemini publica un endpoint compatible con OpenAI
(`generativelanguage.googleapis.com/v1beta/openai`), asi que los dos comparten
el mismo transporte y solo cambian en la URL base y en como llaman a los
parametros de razonamiento.

La parte delicada son los modelos de razonamiento de OpenAI: rechazan
`max_tokens` (piden `max_completion_tokens`) y `temperature` distinta de 1.
En vez de mantener una lista de que modelo acepta que -que envejece mal- se
envia lo que dice el catalogo y, si la API se queja de un parametro concreto,
se reintenta una vez sin el. El ajuste se recuerda por instancia.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import ToolSpec
from .catalog import model_info
from .openai_compat_provider import OpenAICompatProvider
from .schema_utils import sanitize_tool_schema

logger = logging.getLogger(__name__)

# Familias de Gemini que interesan para un agente conversacional.
GEMINI_FAMILIES = ("flash-lite", "flash", "pro")
# Variantes que no sirven como modelo de agente.
EXCLUDED_MODEL_MARKERS = (
    "embedding", "aqa", "imagen", "veo", "tts", "image-generation",
    "native-audio", "live", "vision-latest", "learnlm",
)


def _gemini_sort_key(model: str) -> tuple:
    """Ordena por version descendente y las familias de mas a menos capaces."""
    version = 0.0
    for part in model.split("-"):
        try:
            version = float(part)
            break
        except ValueError:
            continue
    # flash-lite contiene "flash": se comprueba primero el mas especifico.
    if "flash-lite" in model:
        family = 2
    elif "flash" in model:
        family = 1
    else:
        family = 0
    return (-version, family, "preview" in model, model)


class _CloudOpenAIProvider(OpenAICompatProvider):
    """Base comun: adapta parametros segun lo que acepte cada modelo."""

    catalog_name = "openai"
    # OpenAI exige `max_completion_tokens` en sus modelos de razonamiento;
    # la capa de compatibilidad de Gemini se queda con `max_tokens`.
    renames_max_tokens = True
    # Valores de `reasoning_effort` con el razonamiento activado y desactivado.
    # "minimal" solo existe en OpenAI: mandarselo a Gemini es un 400.
    effort_on = "medium"
    effort_off = "minimal"

    def __init__(self, base_url: str, model: str, **options: Any) -> None:
        super().__init__(base_url, model, **options)
        # Ajustes aprendidos en caliente a partir de los errores de la API.
        self._use_completion_tokens: bool | None = None
        self._drop_sampling: bool | None = None
        # El modelo rechazo `reasoning_effort`. Se recuerda para no pagar un
        # 400 + reintento en cada peticion.
        self._drop_effort: bool = False

    def _accepts_sampling(self, model: str) -> bool:
        if self._drop_sampling is not None:
            return not self._drop_sampling
        info = model_info(self.catalog_name, model)
        return info.sampling if info else True

    def _reasoning_model(self, model: str) -> bool:
        info = model_info(self.catalog_name, model)
        return bool(info and info.thinking == "effort")

    def _adapt(self, payload: dict[str, Any], thinking: bool | None) -> dict[str, Any]:
        model = payload["model"]

        if not self._accepts_sampling(model):
            payload.pop("temperature", None)
            payload.pop("top_p", None)

        use_completion = (
            self._use_completion_tokens
            if self._use_completion_tokens is not None
            else (self.renames_max_tokens and self._reasoning_model(model))
        )
        if use_completion and "max_tokens" in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")

        if self._reasoning_model(model) and not self._drop_effort:
            want = self.options.get("thinking", True) if thinking is None else thinking
            payload["reasoning_effort"] = self.effort_on if want else self.effort_off

        return payload

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        """Envia la peticion y reintenta una vez si rechazan un parametro."""
        response = await self._client.post("/chat/completions", json=payload)
        if response.status_code != 400:
            return response

        detail = response.text.lower()
        retried = False

        if "max_tokens" in detail and "max_completion_tokens" in detail and "max_tokens" in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")
            self._use_completion_tokens = True
            retried = True
        if "temperature" in detail and "temperature" in payload:
            payload.pop("temperature", None)
            payload.pop("top_p", None)
            self._drop_sampling = True
            retried = True
        if "reasoning_effort" in detail and "reasoning_effort" in payload:
            payload.pop("reasoning_effort")
            self._drop_effort = True
            retried = True

        if not retried:
            return response
        logger.info("%s: reintento tras parametro rechazado por %s", self.name, payload["model"])
        return await self._client.post("/chat/completions", json=payload)

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
        result = await super().health()
        if not result.get("ok"):
            result["needs_api_key"] = True
            error = str(result.get("error", ""))
            lowered = error.lower()
            # Gemini responde 400 "Please pass a valid API key" donde OpenAI
            # devuelve 401, asi que se mira tambien el texto.
            if "401" in error or "unauthorized" in lowered or "valid api key" in lowered:
                result["error"] = "Clave de API invalida o revocada."
            elif "429" in error:
                result["error"] = "Limite de peticiones alcanzado (429)."
        return result


class OpenAIProvider(_CloudOpenAIProvider):
    name = "openai"
    catalog_name = "openai"


class GoogleProvider(_CloudOpenAIProvider):
    """Gemini a traves de su capa de compatibilidad con OpenAI.

    Es la que mas se aparta del dialecto de OpenAI, y cada desviacion se
    manifiesta como un 400 sin explicacion en la UI:

    * rechaza `"content": null` en el turno del asistente que solo trae
      tool_calls -> fallaba justo despues de la primera herramienta;
    * rechaza el campo `name` en los mensajes de rol `tool`;
    * no conoce `reasoning_effort: "minimal"` (eso es de OpenAI);
    * solo acepta un subconjunto de JSON Schema en las herramientas, asi que
      los esquemas de los servidores MCP hay que traducirlos;
    * desde Gemini 3 firma cada llamada a herramienta con un
      `thought_signature` y exige que vuelva en el siguiente turno.
    """

    name = "google"
    catalog_name = "google"
    renames_max_tokens = False
    # La URL de Gemini ya termina en /v1beta/openai: no lleva /v1 detras.
    appends_v1 = False
    empty_assistant_content = "omit"
    tool_message_includes_name = False
    effort_off = "low"
    # Gemini 3 devuelve la firma de su razonamiento en
    # `tool_calls[].extra_content.google.thought_signature` y rechaza el turno
    # siguiente si no se la devuelves: "Function call is missing a
    # thought_signature in functionCall parts".
    extra_content_key = "google"

    def _prepare_tools(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        prepared = []
        for tool in tools:
            spec = tool.to_openai()
            schema = sanitize_tool_schema(tool.input_schema)
            if schema.get("properties"):
                spec["function"]["parameters"] = schema
            else:
                # Una herramienta sin argumentos: Gemini rechaza un esquema con
                # `properties` vacio, hay que omitir `parameters` del todo.
                spec["function"].pop("parameters", None)
            prepared.append(spec)
        return prepared

    def _filter_models(self, ids: list[str]) -> list[str]:
        """Deja solo los Gemini conversacionales de las familias utiles.

        La cuenta devuelve tambien embeddings, imagen, veo, tts y aqa, que no
        sirven para un agente y solo ensucian el desplegable.
        """
        keep = []
        for raw in ids:
            model = raw.removeprefix("models/")
            if not model.startswith("gemini-"):
                continue
            if any(bad in model for bad in EXCLUDED_MODEL_MARKERS):
                continue
            if not any(family in model for family in GEMINI_FAMILIES):
                continue
            keep.append(model)
        # Mas nuevo primero: ordena por version descendente y deja los preview
        # justo detras de su equivalente estable.
        return sorted(set(keep), key=_gemini_sort_key)
