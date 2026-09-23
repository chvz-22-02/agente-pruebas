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
    # Variable de entorno con el dato que va dentro de la URL, para proveedores
    # que lo llevan ahi: la cuenta en Cloudflare, la region en AWS. La URL por
    # defecto trae el marcador `{account_id}` en su sitio.
    account_env: str = ""
    # Como se llama ese dato en la UI ("Account ID", "Region"...).
    account_label: str = ""
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
            ModelInfo("qwen3:8b", "Qwen3 8B", "32K", "5,2 GB Â· equilibrio en CPU",
                      thinking="toggle", recommended=True),
            ModelInfo("qwen3:4b", "Qwen3 4B", "32K", "2,6 GB Â· el doble de rapido",
                      thinking="toggle"),
            ModelInfo("qwen3:14b", "Qwen3 14B", "32K", "9,3 GB Â· mas preciso, mas lento",
                      thinking="toggle"),
            ModelInfo("granite3.3:8b", "Granite 3.3 8B", "128K", "4,9 GB Â· buen tool calling"),
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
    "cloudflare": ProviderInfo(
        name="cloudflare",
        label="Cloudflare Workers AI",
        kind="cloud",
        needs_api_key=True,
        # El endpoint compatible con OpenAI cuelga de la cuenta. `{account_id}`
        # se rellena con lo que escribas en la UI o con CLOUDFLARE_ACCOUNT_ID.
        default_base_url="https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        key_env="CLOUDFLARE_API_TOKEN",
        key_hint="token con permiso Workers AI",
        console_url="https://dash.cloudflare.com/profile/api-tokens",
        account_env="CLOUDFLARE_ACCOUNT_ID",
        account_label="Account ID",
        # Solo modelos con function calling: sin herramientas no sirven para
        # probar un MCP. El free tier son 10.000 neuronas al dia (se reinicia a
        # las 00:00 UTC); los mas baratos por token rinden mas conversaciones.
        # Al validar el token, el desplegable se rellena con el catalogo real.
        models=[
            ModelInfo("@cf/zai-org/glm-4.7-flash", "GLM-4.7 Flash", "131K",
                      "muy barato: el que mas rinde en el free tier", thinking="effort",
                      recommended=True),
            ModelInfo("@cf/openai/gpt-oss-120b", "gpt-oss 120B", "128K",
                      "mas capaz, gasta el free tier antes", thinking="effort"),
            ModelInfo("@cf/openai/gpt-oss-20b", "gpt-oss 20B", "128K",
                      "rapido y economico", thinking="effort"),
            ModelInfo("@cf/qwen/qwen3-30b-a3b-fp8", "Qwen3 30B A3B", "32K",
                      "ventana corta para MCP con muchas herramientas", thinking="effort"),
            ModelInfo("@cf/mistralai/mistral-small-3.1-24b-instruct", "Mistral Small 3.1 24B",
                      "128K", "sin razonamiento"),
            ModelInfo("@cf/meta/llama-4-scout-17b-16e-instruct", "Llama 4 Scout 17B", "131K",
                      "sin razonamiento"),
            ModelInfo("@cf/meta/llama-3.3-70b-instruct-fp8-fast", "Llama 3.3 70B fast", "24K",
                      "ventana corta"),
        ],
    ),
    "nvidia": ProviderInfo(
        name="nvidia",
        label="NVIDIA (build.nvidia.com)",
        kind="cloud",
        needs_api_key=True,
        default_base_url="https://integrate.api.nvidia.com/v1",
        key_env="NVIDIA_API_KEY",
        key_hint="nvapi-...",
        console_url="https://build.nvidia.com/settings/api-keys",
        # Endpoints gratuitos del NVIDIA Developer Program: sin cupo de tokens,
        # pero con 40 peticiones por minuto por cuenta. Solo modelos con tool
        # calling comprobado en el free tier (2026-09-17). Ojo: `/v1/models`
        # lista tambien modelos que la cuenta gratuita no puede usar (404
        # "Function not found for account", p.ej. kimi-k2.6, nemotron-nano-3).
        # "toggle": el razonamiento viaja en `chat_template_kwargs`.
        models=[
            ModelInfo("nvidia/nemotron-3-super-120b-a12b", "Nemotron 3 Super 120B", "1M",
                      "el mas rapido en el free tier (~1 s)", thinking="toggle", recommended=True),
            ModelInfo("z-ai/glm-5.3", "GLM-5.3", "", "~15 s por respuesta", thinking="toggle"),
            ModelInfo("z-ai/glm-5.3-flash", "GLM-5.3 Flash", "",
                      "cola larga en el free tier (minutos)", thinking="toggle"),
            ModelInfo("deepseek-ai/deepseek-v4-flash-0731", "DeepSeek V4 Flash", "",
                      "cola larga en el free tier (minutos)", thinking="toggle"),
            ModelInfo("openai/gpt-oss-20b", "gpt-oss 20B", "128K", "razona siempre; ~1 min",
                      thinking="always"),
            ModelInfo("mistralai/mistral-nemotron", "Mistral Nemotron", "",
                      "sin razonamiento; dio 500 en la prueba, sin verificar"),
        ],
    ),
    "aws": ProviderInfo(
        name="aws",
        label="AWS (Amazon Bedrock)",
        kind="cloud",
        needs_api_key=True,
        # La region va dentro del host. `{account_id}` se rellena con lo que
        # escribas en la UI o con AWS_REGION.
        default_base_url="https://bedrock-runtime.{account_id}.amazonaws.com",
        key_env="AWS_BEARER_TOKEN_BEDROCK",
        key_hint="clave de API de Bedrock",
        console_url="https://console.aws.amazon.com/bedrock/home#/api-keys",
        account_env="AWS_REGION",
        account_label="Region",
        # Punto de partida, no catalogo: los identificadores de Bedrock llevan
        # version y caducan, el prefijo de region (`us.`) no es universal y
        # varios de estos modelos solo existen en algunas regiones. Al pulsar
        # 'Validar' el desplegable se rellena con los perfiles de inferencia y
        # los modelos bajo demanda que tenga de verdad tu cuenta y tu region.
        models=[
            ModelInfo("us.anthropic.claude-sonnet-4-6", "Claude Sonnet 4.6 (US)", "1M",
                      "razonamiento adaptativo", thinking="adaptive", recommended=True),
            ModelInfo("us.anthropic.claude-haiku-4-5-20251001-v1:0", "Claude Haiku 4.5 (US)",
                      "200K", "el mas rapido; su razonamiento usa el dialecto antiguo y no se "
                      "controla desde aqui"),
            ModelInfo("us.amazon.nova-2-lite-v1:0", "Amazon Nova 2 Lite (US)", "1M",
                      "razonamiento apagado por defecto", thinking="toggle"),
            ModelInfo("openai.gpt-oss-120b-1:0", "gpt-oss 120B", "128K",
                      "solo en us-east-1, us-east-2 y us-west-2"),
            ModelInfo("deepseek.v3.2", "DeepSeek V3.2", "164K",
                      "razona siempre, no se puede apagar; solo en regiones us-*",
                      thinking="always"),
            ModelInfo("qwen.qwen3-coder-next", "Qwen3 Coder Next", "256K",
                      "solo en us-east-1, eu-west-2 y ap-southeast-2"),
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
