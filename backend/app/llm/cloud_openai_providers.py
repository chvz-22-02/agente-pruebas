"""Proveedores de nube que hablan el dialecto OpenAI: OpenAI, Gemini, Cloudflare y NVIDIA.

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

import asyncio
import logging
import os
from typing import Any

import httpx

from .base import LLMResponse, ToolSpec
from .catalog import model_info
from .openai_compat_provider import LLMRequestError, OpenAICompatProvider, _error_text
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


class MissingAccountId(ValueError):
    """La URL de Cloudflare necesita el identificador de cuenta y no ha llegado."""

    def __init__(self) -> None:
        super().__init__(
            "Cloudflare necesita tu Account ID. Escribelo en la pestana Modelo o define "
            f"{CLOUDFLARE_ACCOUNT_ENV} en backend/.env (lo ves en dash.cloudflare.com, "
            "barra lateral de Workers AI)."
        )


CLOUDFLARE_ACCOUNT_ENV = "CLOUDFLARE_ACCOUNT_ID"
ACCOUNT_PLACEHOLDER = "{account_id}"
# El listado de modelos no esta en la capa compatible con OpenAI (`/v1/models`
# no existe): se usa la busqueda de la API propia de Workers AI.
CLOUDFLARE_MODELS_PER_PAGE = 100
CLOUDFLARE_MAX_PAGES = 10


def _property(model: dict[str, Any], name: str) -> Any:
    """Las propiedades de un modelo llegan como lista de {property_id, value}."""
    props = model.get("properties")
    if isinstance(props, dict):
        return props.get(name)
    for prop in props or []:
        if isinstance(prop, dict) and prop.get("property_id") == name:
            return prop.get("value")
    return None


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


class CloudflareProvider(_CloudOpenAIProvider):
    """Workers AI a traves de su endpoint compatible con OpenAI.

    Diferencias con el resto de nubes que conviene tener presentes:

    * la URL lleva el identificador de cuenta
      (`/client/v4/accounts/<id>/ai/v1`); llega desde la UI o de
      CLOUDFLARE_ACCOUNT_ID;
    * no hay `/v1/models`: la lista real sale de `/ai/models/search`, y se
      filtra a generacion de texto con function calling, porque un modelo sin
      herramientas no sirve para probar un MCP;
    * los errores vienen en `errors: [{code, message}]`, y el que importa en
      el free tier es el de las 10.000 neuronas diarias agotadas.
    """

    name = "cloudflare"
    catalog_name = "cloudflare"
    renames_max_tokens = False
    # La URL ya termina en /ai/v1.
    appends_v1 = False
    # El esquema de mensajes de Workers AI declara `content` como texto.
    empty_assistant_content = "empty"
    effort_off = "low"

    def __init__(self, base_url: str, model: str, **options: Any) -> None:
        account = (options.get("account_id") or os.environ.get(CLOUDFLARE_ACCOUNT_ENV, "")).strip()
        if ACCOUNT_PLACEHOLDER in base_url and account:
            base_url = base_url.replace(ACCOUNT_PLACEHOLDER, account)
        super().__init__(base_url, model, **options)

    @property
    def has_account(self) -> bool:
        return ACCOUNT_PLACEHOLDER not in self.base_url

    def _require_account(self) -> None:
        if not self.has_account:
            raise MissingAccountId()

    async def chat(self, messages: list[dict[str, Any]], tools: list[ToolSpec] | None = None,
                   **kwargs: Any) -> LLMResponse:
        self._require_account()
        return await super().chat(messages, tools, **kwargs)

    def _explain_error(self, status: int, detail: str) -> str:
        lowered = detail.lower()
        if status == 429 or "4006" in detail or "neuron" in lowered:
            return (
                f"{detail} — agotaste el free tier de Workers AI (10.000 neuronas al dia, se "
                "reinicia a las 00:00 UTC) o superaste el limite de peticiones. Un modelo mas "
                "barato por token, como GLM-4.7 Flash, rinde mas conversaciones."
            )
        if status in {401, 403} or "10000" in detail or "authentication" in lowered:
            return (
                f"{detail} — revisa que el token tenga el permiso 'Workers AI' y que el "
                "Account ID sea el de la cuenta del token."
            )
        return detail

    def _search_url(self) -> str:
        return self.base_url.removesuffix("/v1") + "/models/search"

    async def list_models(self) -> list[str]:
        self._require_account()
        rows: list[dict[str, Any]] = []
        for page in range(1, CLOUDFLARE_MAX_PAGES + 1):
            resp = await self._client.get(
                self._search_url(),
                params={"task": "Text Generation", "per_page": CLOUDFLARE_MODELS_PER_PAGE, "page": page},
                timeout=20.0,
            )
            if resp.status_code >= 400:
                raise LLMRequestError(
                    resp.status_code, self._explain_error(resp.status_code, _error_text(resp))
                )
            batch = (resp.json() or {}).get("result") or []
            rows.extend(r for r in batch if isinstance(r, dict))
            if len(batch) < CLOUDFLARE_MODELS_PER_PAGE:
                break
        return self._select_models(rows)

    @staticmethod
    def _select_models(rows: list[dict[str, Any]]) -> list[str]:
        """Generacion de texto con function calling, en orden alfabetico.

        Si la API no informara de la propiedad en ningun modelo (cambio de
        formato), se devuelven todos los de texto antes que una lista vacia.
        """
        text = [
            r for r in rows
            if r.get("name") and (r.get("task") or {}).get("name", "Text Generation") == "Text Generation"
        ]
        informed = [r for r in text if _property(r, "function_calling") is not None]
        chosen = [r for r in text if _truthy(_property(r, "function_calling"))] if informed else text
        return sorted({r["name"] for r in chosen})

    async def health(self) -> dict[str, Any]:
        if not self.has_account:
            return {
                "ok": False,
                "provider": self.name,
                "base_url": self.base_url,
                "model": self.model,
                "needs_api_key": True,
                "error": str(MissingAccountId()),
            }
        return await super().health()


# Modelos del catalogo de NVIDIA que no son conversacionales: embeddings,
# rerankers, guardarrailes, parsers, vision pura, traduccion, recompensa...
NVIDIA_EXCLUDED_MARKERS = (
    "embed", "rerank", "retriever", "guard", "safety", "reward", "parse", "clip",
    "vila", "neva", "kosmos", "fuyu", "deplot", "translate", "synthetic-video",
    "cosmos", "calibration", "diffusion", "topic-control",
)
# Reintentos: el free tier son 40 peticiones por minuto, y una evaluacion
# (agente + simulador + evaluador) las encadena muy rapido. Ademas los
# endpoints gratuitos fallan de forma intermitente: medido el 2026-09-17, la
# misma peticion devolvia 500 "Internal server error" en 0,3 s y a la siguiente
# funcionaba, con cualquier variante de payload; y 503 "Service temporarily
# overloaded" cuando van cargados. Todo eso es transitorio y se reintenta.
NVIDIA_RETRY_STATUSES = {429, 500, 502, 503, 504}
NVIDIA_MAX_RETRIES = 5
NVIDIA_MAX_WAIT_S = 30.0
NVIDIA_BASE_WAIT_S = 2.0


class NvidiaProvider(_CloudOpenAIProvider):
    """Endpoints alojados de build.nvidia.com (NIM), compatibles con OpenAI.

    * El razonamiento no va por `reasoning_effort`: cada modelo lo lee de su
      plantilla de chat. Nemotron y Qwen usan `enable_thinking`; Kimi, GLM y
      DeepSeek, `thinking`. Se mandan las dos claves (una plantilla ignora las
      variables que no usa) y, si el endpoint rechaza el parametro, se retira y
      se recuerda, igual que con los motores locales.
    * El free tier limita a 40 peticiones por minuto. Un 429 no se le pasa al
      agente: se espera lo que diga `Retry-After` (o un backoff) y se reintenta.
    """

    name = "nvidia"
    catalog_name = "nvidia"
    renames_max_tokens = False
    # La URL por defecto ya termina en /v1.
    appends_v1 = False

    def _adapt(self, payload: dict[str, Any], thinking: bool | None) -> dict[str, Any]:
        model = payload["model"]
        if not self._accepts_sampling(model):
            payload.pop("temperature", None)
            payload.pop("top_p", None)
        info = model_info(self.catalog_name, model)
        # "always" y "none" no se controlan; fuera del catalogo se intenta.
        controllable = info is None or info.thinking == "toggle"
        want = self.options.get("thinking", True) if thinking is None else thinking
        if controllable and self._accepts_template_kwargs is not False:
            kwargs = dict(payload.get("chat_template_kwargs") or {})
            kwargs["enable_thinking"] = bool(want)
            kwargs["thinking"] = bool(want)
            payload["chat_template_kwargs"] = kwargs
        return payload

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        for attempt in range(NVIDIA_MAX_RETRIES + 1):
            response = await OpenAICompatProvider._post(self, payload)
            if response.status_code not in NVIDIA_RETRY_STATUSES or attempt == NVIDIA_MAX_RETRIES:
                return response
            wait = self._retry_wait(response, attempt)
            logger.info(
                "nvidia: %s en %s, reintento %d/%d en %.1f s",
                response.status_code, payload.get("model"), attempt + 1, NVIDIA_MAX_RETRIES, wait,
            )
            await asyncio.sleep(wait)
        return response

    @staticmethod
    def _retry_wait(response: httpx.Response, attempt: int) -> float:
        header = response.headers.get("retry-after", "")
        try:
            wait = float(header)
        except ValueError:
            wait = NVIDIA_BASE_WAIT_S * (2**attempt)
        return max(0.0, min(wait, NVIDIA_MAX_WAIT_S))

    def _explain_error(self, status: int, detail: str) -> str:
        if status == 429:
            return (
                f"{detail} — limite del free tier de NVIDIA (40 peticiones por minuto) tras "
                f"{NVIDIA_MAX_RETRIES} reintentos. Espera un minuto o baja el ritmo "
                "(menos repeticiones, o simulador y evaluador en otro proveedor)."
            )
        if status in {401, 403}:
            return f"{detail} — revisa la clave nvapi- en build.nvidia.com/settings/api-keys."
        if status in {500, 502, 503, 504}:
            return (
                f"{detail} — el endpoint gratuito de NVIDIA esta saturado y siguio asi tras "
                f"{NVIDIA_MAX_RETRIES} reintentos. Prueba en unos minutos o con otro modelo."
            )
        if status == 404:
            return (
                f"{detail} — el modelo aparece en el catalogo pero no esta disponible para tu "
                "cuenta en el free tier. Elige otro."
            )
        return detail

    def _filter_models(self, ids: list[str]) -> list[str]:
        return sorted(
            m for m in set(ids) if not any(marker in m.lower() for marker in NVIDIA_EXCLUDED_MARKERS)
        )
