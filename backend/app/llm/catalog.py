"""Catalogo de proveedores y modelos que la UI ofrece para elegir.

Es solo una lista de sugerencias: el campo de modelo de la UI acepta cualquier
identificador, y cuando hay clave de API el backend consulta la lista real al
proveedor (`list_models`). El catalogo sirve para arrancar sin tener que
recordar los identificadores de memoria.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# Como se le pide razonamiento a cada familia de modelos:
#   "adaptive" -> parametro thinking={"type": "adaptive"} (Claude 4.6+)
#   "effort"   -> se regula con output_config.effort
#   "toggle"   -> el motor acepta un booleano (Ollama `think`)
#   "always"   -> siempre razona, no se configura (Claude Fable)
#   "none"     -> el modelo no expone razonamiento
ThinkingMode = Literal["adaptive", "effort", "toggle", "always", "none"]


@dataclass(frozen=True)
class ModelInfo:
    id: str
    label: str
    context: str = ""
    notes: str = ""
    # Si es False, enviar temperature/top_p devuelve un 400 (Claude 4.7+).
    sampling: bool = True
    thinking: ThinkingMode = "none"
    recommended: bool = False


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    label: str
    kind: Literal["local", "cloud"]
    needs_api_key: bool
    default_base_url: str = ""
    key_env: str = ""
    key_hint: str = ""
    console_url: str = ""
    supports_pull: bool = False
    models: list[ModelInfo] = field(default_factory=list)


# Los precios y ventanas de contexto cambian a menudo; aqui solo hay lo
# necesario para elegir con criterio, no una tabla de tarifas.
CATALOG: dict[str, ProviderInfo] = {
    "ollama": ProviderInfo(
        name="ollama",
        label="Ollama (local)",
        kind="local",
        needs_api_key=False,
        default_base_url="http://127.0.0.1:11434",
        supports_pull=True,
        models=[
            ModelInfo("qwen3:8b", "Qwen3 8B", "32K", "5,2 GB · equilibrio en CPU",
                      thinking="toggle", recommended=True),
            ModelInfo("qwen3:4b", "Qwen3 4B", "32K", "2,6 GB · el doble de rapido",
                      thinking="toggle"),
            ModelInfo("qwen3:14b", "Qwen3 14B", "32K", "9,3 GB · mas preciso, mas lento",
                      thinking="toggle"),
            ModelInfo("granite3.3:8b", "Granite 3.3 8B", "128K", "4,9 GB · buen tool calling"),
        ],
    ),
    "anthropic": ProviderInfo(
        name="anthropic",
        label="Anthropic (Claude)",
        kind="cloud",
        needs_api_key=True,
        default_base_url="https://api.anthropic.com",
        key_env="ANTHROPIC_API_KEY",
        key_hint="sk-ant-...",
        console_url="https://console.anthropic.com/settings/keys",
        models=[
            ModelInfo("claude-opus-5", "Claude Opus 5", "1M",
                      "razonamiento adaptativo por defecto", sampling=False,
                      thinking="adaptive", recommended=True),
            ModelInfo("claude-sonnet-5", "Claude Sonnet 5", "1M",
                      "mas barato que Opus, muy capaz", sampling=False, thinking="adaptive"),
            ModelInfo("claude-opus-4-8", "Claude Opus 4.8", "1M",
                      "generacion anterior de Opus", sampling=False, thinking="adaptive"),
            ModelInfo("claude-sonnet-4-6", "Claude Sonnet 4.6", "1M",
                      "acepta temperature", sampling=True, thinking="adaptive"),
            ModelInfo("claude-haiku-4-5", "Claude Haiku 4.5", "200K",
                      "el mas rapido y barato", sampling=True),
            ModelInfo("claude-fable-5-1", "Claude Fable 5.1", "1M",
                      "el mas capaz; razona siempre", sampling=False, thinking="always"),
        ],
    ),
    "openai": ProviderInfo(
        name="openai",
        label="OpenAI",
        kind="cloud",
        needs_api_key=True,
        default_base_url="https://api.openai.com/v1",
        key_env="OPENAI_API_KEY",
        key_hint="sk-...",
        console_url="https://platform.openai.com/api-keys",
        models=[
            ModelInfo("gpt-5", "GPT-5", "400K", "modelo de razonamiento",
                      sampling=False, thinking="effort", recommended=True),
            ModelInfo("gpt-5-mini", "GPT-5 mini", "400K", "mas rapido y barato",
                      sampling=False, thinking="effort"),
            ModelInfo("gpt-5-nano", "GPT-5 nano", "400K", "el mas barato",
                      sampling=False, thinking="effort"),
            ModelInfo("gpt-4.1", "GPT-4.1", "1M", "sin razonamiento, acepta temperature"),
            ModelInfo("gpt-4.1-mini", "GPT-4.1 mini", "1M", "economico"),
            ModelInfo("o4-mini", "o4-mini", "200K", "razonamiento economico",
                      sampling=False, thinking="effort"),
        ],
    ),
    "google": ProviderInfo(
        name="google",
        label="Google (Gemini)",
        kind="cloud",
        needs_api_key=True,
        default_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        key_env="GOOGLE_API_KEY",
        key_hint="AIza...",
        console_url="https://aistudio.google.com/apikey",
        # Solo las familias pro / flash / flash-lite. Los identificadores con
        # version envejecen y devuelven 404, asi que esta lista es solo el
        # punto de partida: al validar la clave, el desplegable se rellena con
        # los modelos reales de la cuenta (incluidos los -preview), ya
        # filtrados a estas mismas familias.
        models=[
            # Gemini 3.x razona por defecto; el interruptor se traduce a
            # `reasoning_effort`. Si el modelo lo rechazara, el proveedor lo
            # recuerda y sigue sin el (ver `_drop_effort`).
            ModelInfo("gemini-3.5-flash", "Gemini 3.5 Flash", "",
                      "generacion 3.x", thinking="effort", recommended=True),
            ModelInfo("gemini-2.5-pro", "Gemini 2.5 Pro", "1M",
                      "el mas capaz", thinking="effort"),
            ModelInfo("gemini-2.5-flash", "Gemini 2.5 Flash", "1M",
                      "rapido y economico", thinking="effort"),
            ModelInfo("gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite", "1M",
                      "el mas barato", thinking="effort"),
        ],
    ),
    "openai_compat": ProviderInfo(
        name="openai_compat",
        label="Compatible OpenAI (llama.cpp, LM Studio, vLLM...)",
        kind="local",
        needs_api_key=False,
        default_base_url="http://127.0.0.1:8080",
        key_hint="opcional",
        models=[],
    ),
}


def model_info(provider: str, model: str) -> ModelInfo | None:
    """Busca un modelo en el catalogo. Devuelve None si es uno escrito a mano."""
    info = CATALOG.get(provider)
    if info is None:
        return None
    return next((m for m in info.models if m.id == model), None)


def as_json() -> list[dict[str, Any]]:
    """Serializa el catalogo para la UI."""
    return [
        {**asdict(p), "models": [asdict(m) for m in p.models]}
        for p in CATALOG.values()
    ]
