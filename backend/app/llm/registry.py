"""Registro de proveedores: cambiar de modelo = una entrada en el diccionario.

Anadir un motor nuevo se reduce a implementar `LLMProvider` y registrarlo aqui.

Sobre las claves de API: llegan por peticion desde la UI y se usan para
construir el cliente, pero nunca se guardan en disco, ni se registran en
MLflow, ni aparecen en logs. La cache de proveedores se indexa por un hash de
la clave, no por la clave, para que dos claves distintas no compartan cliente
sin tener que conservar el secreto como parte de la llave del diccionario.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from typing import Any

from ..config import settings
from .anthropic_provider import AnthropicProvider
from .base import LLMProvider
from .catalog import CATALOG
from .cloud_openai_providers import (
    CloudflareProvider,
    GoogleProvider,
    NvidiaProvider,
    OpenAIProvider,
)
from .ollama_provider import OllamaProvider
from .openai_compat_provider import OpenAICompatProvider

PROVIDERS: dict[str, type[LLMProvider]] = {
    "ollama": OllamaProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "google": GoogleProvider,
    "cloudflare": CloudflareProvider,
    "nvidia": NvidiaProvider,
    "openai_compat": OpenAICompatProvider,
}

_cache: dict[tuple[str, str, str, str], LLMProvider] = {}
_lock = asyncio.Lock()


def _fingerprint(api_key: str) -> str:
    """Huella corta de la clave, para indexar la cache sin conservar el secreto."""
    if not api_key:
        return "-"
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def resolve_api_key(provider: str, api_key: str | None) -> str:
    """La clave de la UI manda; si no viene, se cae al entorno del backend.

    Asi se puede trabajar sin escribir la clave en el navegador (poniendola en
    `backend/.env`) o sin tocar el backend (escribiendola en la UI).
    """
    if api_key:
        return api_key.strip()
    info = CATALOG.get(provider)
    if info and info.key_env:
        return os.environ.get(info.key_env, "").strip()
    return settings.llm_api_key


def default_base_url(provider: str) -> str:
    """URL base del proveedor cuando la peticion no trae una.

    Ojo con el caso que parece inofensivo: si un proveedor de nube cayera a
    `settings.llm_base_url` heredaria la URL del motor local, y como Ollama
    tambien sirve `/v1/models`, el cliente de nube se pondria a hablar con
    Ollama devolviendo respuestas que parecen validas. Por eso un proveedor
    del catalogo se queda con lo suyo, aunque este vacio.
    """
    info = CATALOG.get(provider)
    if info is not None:
        return info.default_base_url
    return settings.llm_base_url


def default_options(provider: str, api_key: str) -> dict[str, Any]:
    return {
        "temperature": settings.llm_temperature,
        "top_p": settings.llm_top_p,
        "num_ctx": settings.llm_num_ctx,
        "max_tokens": settings.llm_max_tokens,
        "request_timeout": settings.llm_request_timeout,
        "thinking": settings.llm_thinking,
        "api_key": api_key,
    }


async def get_provider(
    provider: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> LLMProvider:
    name = (provider or settings.llm_provider).lower()
    if name not in PROVIDERS:
        raise ValueError(f"Proveedor LLM desconocido: {name}. Disponibles: {sorted(PROVIDERS)}")

    key = resolve_api_key(name, api_key)
    info = CATALOG.get(name)
    if info and info.needs_api_key and not key:
        raise MissingAPIKey(name)

    url = (base_url or default_base_url(name)).rstrip("/")
    mdl = model or settings.llm_model
    cache_key = (name, url, mdl, _fingerprint(key))

    async with _lock:
        if cache_key not in _cache:
            _cache[cache_key] = PROVIDERS[name](url, mdl, **default_options(name, key))
        return _cache[cache_key]


class MissingAPIKey(RuntimeError):
    """El proveedor necesita una clave y no ha llegado ninguna."""

    def __init__(self, provider: str) -> None:
        info = CATALOG.get(provider)
        env = info.key_env if info else ""
        super().__init__(
            f"El proveedor '{provider}' necesita una clave de API. "
            f"Introducela en la pestana Modelo de la UI"
            + (f" o exporta {env} en el backend." if env else ".")
        )
        self.provider = provider


async def close_all() -> None:
    async with _lock:
        for provider in _cache.values():
            await provider.aclose()
        _cache.clear()
