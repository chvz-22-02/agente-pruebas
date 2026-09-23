"""Pruebas de la traduccion al dialecto de cada proveedor de nube.

No hacen ninguna llamada de red ni necesitan claves: comprueban como se
construye el payload, que es donde estan las diferencias que rompen.

    python tests\\test_cloud_providers.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm.anthropic_provider import AnthropicProvider  # noqa: E402
from app.llm.base import ToolSpec  # noqa: E402
from app.llm.bedrock_provider import (  # noqa: E402
    BedrockProvider,
    MissingRegion,
    _error_text,
)
from app.llm.catalog import CATALOG, model_info  # noqa: E402
from app.llm.cloud_openai_providers import (  # noqa: E402
    CloudflareProvider,
    GoogleProvider,
    MissingAccountId,
    NvidiaProvider,
    OpenAIProvider,
)
from app.llm.schema_utils import sanitize_tool_schema  # noqa: E402

CONVERSATION = [
    {"role": "system", "content": "Eres un agente de pruebas."},
    {"role": "user", "content": "Consulta el stock del SKU-002"},
    {
        "role": "assistant",
        "content": "Voy a mirarlo.",
        "tool_calls": [
            {"id": "call_a", "name": "consultar_inventario", "arguments": {"sku": "SKU-002"}},
            {"id": "call_b", "name": "hora_actual", "arguments": {}},
        ],
    },
    {"role": "tool", "content": '{"stock": 7}', "tool_call_id": "call_a", "name": "consultar_inventario"},
    {"role": "tool", "content": "2026-09-07", "tool_call_id": "call_b", "name": "hora_actual"},
]

TOOLS = [ToolSpec("consultar_inventario", "Consulta el inventario", {"type": "object", "properties": {}})]


def test_anthropic_messages() -> None:
    provider = AnthropicProvider("", "claude-opus-5", api_key="dummy")

    system, rest = provider._split_system(CONVERSATION)
    assert system == "Eres un agente de pruebas.", system
    assert all(m["role"] != "system" for m in rest)

    serialized = provider._serialize_messages(rest)
    roles = [m["role"] for m in serialized]
    assert roles == ["user", "assistant", "user"], roles

    # Los dos tool_result deben ir juntos en UN solo mensaje de usuario:
    # repartirlos le ensena al modelo a dejar de pedir herramientas en paralelo.
    results = serialized[2]["content"]
    assert len(results) == 2, results
    assert all(b["type"] == "tool_result" for b in results)
    assert [b["tool_use_id"] for b in results] == ["call_a", "call_b"]

    # El turno del asistente lleva texto + un bloque tool_use por llamada.
    blocks = serialized[1]["content"]
    assert blocks[0] == {"type": "text", "text": "Voy a mirarlo."}
    assert [b["name"] for b in blocks[1:]] == ["consultar_inventario", "hora_actual"]
    assert blocks[1]["input"] == {"sku": "SKU-002"}
    print("  anthropic: system separado, tool_result agrupados, tool_use correctos")


def test_anthropic_preserves_raw_blocks() -> None:
    """Los bloques originales (razonamiento firmado incluido) se devuelven intactos."""
    provider = AnthropicProvider("", "claude-opus-5", api_key="dummy")
    original = [
        {"type": "thinking", "thinking": "...", "signature": "abc123"},
        {"type": "tool_use", "id": "call_a", "name": "x", "input": {}},
    ]
    messages = [
        {"role": "user", "content": "hola"},
        {"role": "assistant", "content": "", "tool_calls": [], "_provider_blocks": original},
        {"role": "tool", "content": "ok", "tool_call_id": "call_a", "name": "x"},
    ]
    serialized = provider._serialize_messages(messages)
    assert serialized[1]["content"] is original, "se perdieron los bloques originales"
    print("  anthropic: bloques de razonamiento firmados preservados")


def test_anthropic_thinking_mapping() -> None:
    provider = AnthropicProvider("", "claude-opus-5", api_key="dummy")

    think, effort = provider._thinking_and_effort("claude-opus-5", True)
    assert think == {"type": "adaptive", "display": "summarized"} and effort is None

    # Desactivar razonamiento en Opus 5 hace que a veces escriba la llamada a
    # la herramienta como texto: se baja el esfuerzo en vez de desactivarlo.
    think, effort = provider._thinking_and_effort("claude-opus-5", False)
    assert think["type"] == "adaptive" and effort == "low", (think, effort)

    # Fable razona siempre y rechaza el parametro.
    assert provider._thinking_and_effort("claude-fable-5-1", True) == (None, None)
    assert provider._thinking_and_effort("claude-haiku-4-5", True) == (None, None)
    print("  anthropic: thinking/effort segun el modelo")


def test_anthropic_payload() -> None:
    provider = AnthropicProvider("", "claude-opus-5", api_key="dummy")
    kwargs = provider._build_kwargs(CONVERSATION, TOOLS, "claude-opus-5", 8000, True)

    assert "temperature" not in kwargs and "top_p" not in kwargs, "Claude 4.7+ devuelve 400 con sampling"
    assert kwargs["system"] == "Eres un agente de pruebas."
    assert kwargs["max_tokens"] == 8000
    # El esquema de herramienta de Claude es plano, no anidado bajo "function".
    tool = kwargs["tools"][0]
    assert set(tool) == {"name", "description", "input_schema"}, tool
    print("  anthropic: payload sin sampling y con esquema plano de herramientas")


def test_openai_reasoning_params() -> None:
    provider = OpenAIProvider("https://api.openai.com/v1", "gpt-5", api_key="dummy")
    payload = {"model": "gpt-5", "max_tokens": 4096, "temperature": 0.6, "top_p": 0.95}
    adapted = provider._adapt(dict(payload), thinking=True)

    assert "max_tokens" not in adapted and adapted["max_completion_tokens"] == 4096
    assert "temperature" not in adapted and "top_p" not in adapted
    assert adapted["reasoning_effort"] == "medium"

    off = provider._adapt(dict(payload), thinking=False)
    assert off["reasoning_effort"] == "minimal"

    # Un modelo sin razonamiento conserva los parametros clasicos.
    classic = provider._adapt({"model": "gpt-4.1", "max_tokens": 4096, "temperature": 0.6}, None)
    assert classic["max_tokens"] == 4096 and classic["temperature"] == 0.6
    assert "reasoning_effort" not in classic
    print("  openai: max_completion_tokens y reasoning_effort solo donde toca")


def test_google_keeps_max_tokens() -> None:
    provider = GoogleProvider(
        "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-pro", api_key="dummy"
    )
    adapted = provider._adapt(
        {"model": "gemini-2.5-pro", "max_tokens": 4096, "temperature": 0.6}, thinking=True
    )
    # La capa de compatibilidad de Gemini se queda con max_tokens y temperature.
    assert adapted["max_tokens"] == 4096, adapted
    assert adapted["temperature"] == 0.6
    assert adapted["reasoning_effort"] == "medium"
    print("  google: conserva max_tokens y temperature, anade reasoning_effort")


def test_cloud_never_inherits_local_url() -> None:
    """Regresion: un proveedor de nube no debe heredar la URL del motor local.

    Cuando Anthropic no tenia URL por defecto, caia en `settings.llm_base_url`
    (Ollama). Como Ollama tambien sirve `/v1/models`, el cliente de Claude se
    ponia a hablar con Ollama y devolvia una lista de modelos que parecia
    valida: un fallo silencioso, de los peores.
    """
    from app.llm.registry import default_base_url

    for name in ("anthropic", "openai", "google", "cloudflare", "nvidia", "aws"):
        url = default_base_url(name)
        assert url.startswith("https://"), f"{name} deberia salir a internet, no a {url}"
        assert "127.0.0.1" not in url and "localhost" not in url

    # Y aunque le fuercen una URL local, Anthropic la ignora.
    provider = AnthropicProvider("http://127.0.0.1:11434", "claude-opus-5", api_key="dummy")
    assert "anthropic.com" in str(provider._client.base_url), provider._client.base_url
    print("  nube: ninguna ruta cae al motor local")


# Esquema tipico de un servidor MCP hecho con pydantic/FastMCP: trae casi todo
# lo que Gemini rechaza.
MCP_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "consultar_inventarioArguments",
    "type": "object",
    "additionalProperties": False,
    "$defs": {
        "Filtro": {
            "type": "object",
            "title": "Filtro",
            "properties": {"desde": {"type": "string", "format": "date-time"}},
            "additionalProperties": False,
        }
    },
    "properties": {
        "sku": {"type": "string", "title": "Sku", "format": "uuid", "minLength": 3},
        "unidades": {"type": "integer", "exclusiveMinimum": 0, "default": 1},
        "modo": {"const": "rapido"},
        "nota": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "filtro": {"$ref": "#/$defs/Filtro"},
        "etiquetas": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "origen": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
    },
    "required": ["sku", "inexistente"],
}


def test_gemini_schema_sanitizer() -> None:
    clean = sanitize_tool_schema(MCP_SCHEMA)

    # Nada de claves que Gemini no conoce, a ningun nivel de anidamiento.
    prohibidas = {"$schema", "$defs", "definitions", "additionalProperties",
                  "title", "exclusiveMinimum", "uniqueItems", "const", "oneOf", "allOf", "$ref"}

    def walk(node, path="raiz"):
        if isinstance(node, dict):
            for key, value in node.items():
                assert key not in prohibidas, f"{path}: quedo la clave {key}"
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]")

    walk(clean)

    props = clean["properties"]
    # `format: uuid` no existe para Gemini; `date-time` si.
    assert "format" not in props["sku"], props["sku"]
    assert props["filtro"]["properties"]["desde"]["format"] == "date-time"
    # const -> enum de un elemento
    assert props["modo"]["enum"] == ["rapido"]
    # anyOf con null -> nullable
    assert props["nota"]["type"] == "string" and props["nota"]["nullable"] is True
    # $ref resuelto en linea
    assert props["filtro"]["type"] == "object"
    # oneOf -> anyOf
    assert [b["type"] for b in props["origen"]["anyOf"]] == ["string", "integer"]
    # un required que apunta a una propiedad inexistente rompe la validacion
    assert clean["required"] == ["sku"], clean["required"]
    print("  gemini: esquema MCP traducido al subconjunto que acepta")


def test_gemini_message_shape() -> None:
    """Regresion del fallo tras la primera herramienta."""
    provider = GoogleProvider(
        "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash", api_key="dummy"
    )
    messages = [
        {"role": "user", "content": "hola"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_a", "name": "x", "arguments": {}}]},
        {"role": "tool", "content": "ok", "tool_call_id": "call_a", "name": "x"},
    ]
    serialized = provider._serialize_messages(messages)

    assistant = serialized[1]
    # Gemini rechaza content:null junto a tool_calls -> la clave no debe ir.
    assert "content" not in assistant, assistant
    assert assistant["tool_calls"][0]["id"] == "call_a"

    tool = serialized[2]
    # Y tampoco admite `name` en el mensaje de rol tool.
    assert "name" not in tool, tool
    assert tool["tool_call_id"] == "call_a"

    # El proveedor generico conserva el comportamiento anterior.
    from app.llm.openai_compat_provider import OpenAICompatProvider

    generic = OpenAICompatProvider("http://127.0.0.1:8080", "x")
    gen = generic._serialize_messages(messages)
    assert gen[1]["content"] is None and gen[2]["name"] == "x"
    print("  gemini: sin content:null ni name en tool; el generico no cambia")


def test_gemini_effort_value() -> None:
    """`minimal` es de OpenAI: mandarselo a Gemini es un 400."""
    google = GoogleProvider(
        "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-pro", api_key="dummy"
    )
    off = google._adapt({"model": "gemini-2.5-pro", "max_tokens": 100}, thinking=False)
    assert off["reasoning_effort"] == "low", off

    openai = OpenAIProvider("https://api.openai.com/v1", "gpt-5", api_key="dummy")
    off_oai = openai._adapt({"model": "gpt-5", "max_tokens": 100}, thinking=False)
    assert off_oai["reasoning_effort"] == "minimal", off_oai
    print("  gemini: reasoning_effort low en vez del minimal de OpenAI")


def test_gemini_model_filter() -> None:
    provider = GoogleProvider(
        "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-pro", api_key="dummy"
    )
    crudo = [
        "models/gemini-2.5-pro", "models/gemini-2.5-flash",
        "models/gemini-2.5-flash-lite", "models/gemini-3-pro-preview",
        "models/gemini-2.5-flash-preview-09-2025",
        "models/text-embedding-004", "models/gemini-embedding-001",
        "models/imagen-4.0-generate-001", "models/veo-3.0-generate-001",
        "models/gemini-2.5-flash-native-audio", "models/aqa",
        "models/gemini-2.0-flash-live-001", "models/gemini-2.5-pro-tts",
    ]
    filtrado = provider._filter_models(crudo)

    assert "gemini-3-pro-preview" in filtrado, "los preview deben incluirse"
    assert "gemini-2.5-flash-preview-09-2025" in filtrado
    for basura in ("embedding", "imagen", "veo", "aqa", "native-audio", "live", "tts"):
        assert not any(basura in m for m in filtrado), f"se colo {basura}: {filtrado}"
    # Sin prefijo "models/" y lo mas nuevo primero.
    assert all(not m.startswith("models/") for m in filtrado)
    assert filtrado[0] == "gemini-3-pro-preview", filtrado
    print(f"  gemini: {len(filtrado)} modelos utiles de {len(crudo)} -> {filtrado}")


def test_gemini_tool_without_arguments() -> None:
    """Muchas herramientas MCP no llevan argumentos (`hora_actual`, `listar_*`).

    Gemini rechaza una declaracion de funcion con `properties` vacio, asi que
    en ese caso `parameters` no debe ir. Es un fallo desde el primer mensaje,
    porque las herramientas viajan ya en la primera peticion.
    """
    provider = GoogleProvider(
        "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash", api_key="dummy"
    )
    sin_args = ToolSpec("hora_actual", "Devuelve la hora", {"type": "object", "properties": {}})
    con_args = ToolSpec("consultar", "Consulta", MCP_SCHEMA)

    prepared = provider._prepare_tools([sin_args, con_args])
    assert "parameters" not in prepared[0]["function"], prepared[0]
    assert prepared[1]["function"]["parameters"]["properties"], prepared[1]

    # El proveedor generico y OpenAI si aceptan el objeto vacio.
    openai = OpenAIProvider("https://api.openai.com/v1", "gpt-4.1", api_key="dummy")
    assert "parameters" in openai._prepare_tools([sin_args])[0]["function"]
    print("  gemini: herramienta sin argumentos se declara sin `parameters`")


# Respuesta real de Gemini 3 por la capa OpenAI: cada tool_call viene firmada.
GEMINI_TOOL_CALLS = [
    {
        "id": "function-call-5873527561210830497",
        "type": "function",
        "function": {"name": "list_data_sources", "arguments": '{"scope":"all"}'},
        "extra_content": {"google": {"thought_signature": "CvcQAdHN2OekY10ClPFkYA=="}},
    },
    {
        "id": "function-call-1111111111111111111",
        "type": "function",
        "function": {"name": "hora_actual", "arguments": "{}"},
        "extra_content": {"google": {"thought_signature": "Zm9vYmFyc2lnbmF0dXJl"}},
    },
]


def test_gemini_thought_signature_roundtrip() -> None:
    """Gemini 3 exige que la firma de cada llamada vuelva en el turno siguiente.

    Sin esto responde 400: "Function call is missing a thought_signature in
    functionCall parts".
    """
    google = GoogleProvider("https://generativelanguage.googleapis.com/v1beta/openai",
                            "gemini-3.5-flash", api_key="dummy")

    # 1) se captura al leer la respuesta
    extras = [google._tool_call_extra(c) for c in GEMINI_TOOL_CALLS]
    assert extras[0] == {"extra_content": {"google": {"thought_signature": "CvcQAdHN2OekY10ClPFkYA=="}}}
    assert extras[1]["extra_content"]["google"]["thought_signature"] == "Zm9vYmFyc2lnbmF0dXJl"

    # 2) y se devuelve intacta al serializar el turno del asistente
    historial = [
        {"role": "user", "content": "lista las fuentes"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": c["id"], "name": c["function"]["name"], "arguments": {}, "extra": e}
                for c, e in zip(GEMINI_TOOL_CALLS, extras, strict=True)
            ],
        },
    ]
    enviado = google._serialize_messages(historial)[1]["tool_calls"]
    assert len(enviado) == 2, enviado
    for original, saliente in zip(GEMINI_TOOL_CALLS, enviado, strict=True):
        assert saliente["extra_content"] == original["extra_content"], saliente
    print("  gemini: las 2 firmas vuelven en extra_content.google.thought_signature")


def test_thought_signature_never_leaks_to_other_providers() -> None:
    """Solo Gemini recibe `extra_content`: para OpenAI es un campo desconocido."""
    openai = OpenAIProvider("https://api.openai.com/v1", "gpt-5", api_key="dummy")
    assert openai.extra_content_key is None

    contaminado = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_a",
                    "name": "hora_actual",
                    "arguments": {},
                    "extra": {"extra_content": {"google": {"thought_signature": "xxx"}}},
                }
            ],
        }
    ]
    # Caso real: se empieza una conversacion con Gemini y se cambia de modelo.
    enviado = openai._serialize_messages(contaminado)[0]["tool_calls"][0]
    assert "extra_content" not in enviado, enviado
    assert set(enviado) == {"id", "type", "function"}, enviado
    print("  openai: no se le cuela extra_content al cambiar de proveedor")


def test_gemini3_effort_and_rejection_memory() -> None:
    """Gemini 3.x recibe `reasoning_effort`; si lo rechaza, no se reintenta siempre."""
    google = GoogleProvider("https://generativelanguage.googleapis.com/v1beta/openai",
                            "gemini-3.5-flash", api_key="dummy")

    encendido = google._adapt({"model": "gemini-3.5-flash", "max_tokens": 100}, True)
    apagado = google._adapt({"model": "gemini-3.5-flash", "max_tokens": 100}, False)
    assert encendido["reasoning_effort"] == "medium", encendido
    assert apagado["reasoning_effort"] == "low", apagado

    # Tras un 400 por el parametro, deja de enviarse: si no, cada peticion
    # pagaria un rechazo y un reintento.
    google._drop_effort = True
    assert "reasoning_effort" not in google._adapt({"model": "gemini-3.5-flash"}, True)
    print("  gemini 3.x: reasoning_effort medium/low y se recuerda el rechazo")


CF_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
CF_MODEL = "@cf/zai-org/glm-4.7-flash"


def test_cloudflare_account_id_resolution() -> None:
    """El Account ID de la UI manda; si no llega, se usa CLOUDFLARE_ACCOUNT_ID."""
    import os

    previous = os.environ.pop("CLOUDFLARE_ACCOUNT_ID", None)
    try:
        explicit = CloudflareProvider(CF_URL, CF_MODEL, api_key="dummy", account_id="abc123")
        assert explicit.base_url.endswith("/accounts/abc123/ai/v1"), explicit.base_url
        assert str(explicit._client.base_url).rstrip("/").endswith("/accounts/abc123/ai/v1")

        # URL ya resuelta desde la UI: se respeta tal cual.
        typed = CloudflareProvider(CF_URL.replace("{account_id}", "uiacct"), CF_MODEL, api_key="dummy")
        assert "/accounts/uiacct/" in typed.base_url

        os.environ["CLOUDFLARE_ACCOUNT_ID"] = "envacct"
        from_env = CloudflareProvider(CF_URL, CF_MODEL, api_key="dummy")
        assert "/accounts/envacct/" in from_env.base_url
        del os.environ["CLOUDFLARE_ACCOUNT_ID"]

        missing = CloudflareProvider(CF_URL, CF_MODEL, api_key="dummy")
        assert not missing.has_account
        health = asyncio.run(missing.health())
        assert not health["ok"] and "Account ID" in health["error"], health
        try:
            asyncio.run(missing.chat([{"role": "user", "content": "hola"}]))
        except MissingAccountId:
            pass
        else:
            raise AssertionError("sin Account ID no deberia llegar a hacer la peticion")
    finally:
        if previous is not None:
            os.environ["CLOUDFLARE_ACCOUNT_ID"] = previous
    print("  cloudflare: Account ID desde la UI, la URL o el entorno; sin el, error claro")


def test_cloudflare_payload() -> None:
    provider = CloudflareProvider(CF_URL, CF_MODEL, api_key="dummy", account_id="acct")
    payload = {"model": CF_MODEL, "max_tokens": 4096, "temperature": 0.6, "top_p": 0.95}

    on = provider._adapt(dict(payload), thinking=True)
    assert on["max_tokens"] == 4096 and "max_completion_tokens" not in on
    assert on["temperature"] == 0.6 and on["reasoning_effort"] == "medium", on
    assert provider._adapt(dict(payload), thinking=False)["reasoning_effort"] == "low"

    # Un modelo sin razonamiento no recibe reasoning_effort.
    plain = provider._adapt({"model": "@cf/meta/llama-4-scout-17b-16e-instruct", "max_tokens": 10}, True)
    assert "reasoning_effort" not in plain

    # El turno del asistente que solo trae tool_calls lleva content "" (texto),
    # no null: el esquema de Workers AI declara content como string.
    serialized = provider._serialize_messages(
        [{"role": "assistant", "content": "", "tool_calls": CONVERSATION[2]["tool_calls"]}]
    )
    assert serialized[0]["content"] == "" and len(serialized[0]["tool_calls"]) == 2
    assert "extra_content" not in serialized[0]["tool_calls"][0]
    print("  cloudflare: max_tokens, temperature y reasoning_effort segun el catalogo")


def test_cloudflare_lists_models_with_function_calling() -> None:
    import httpx

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        assert request.headers["authorization"] == "Bearer dummy"
        return httpx.Response(200, json={"success": True, "result": [
            {"name": "@cf/zai-org/glm-4.7-flash", "task": {"name": "Text Generation"},
             "properties": [{"property_id": "function_calling", "value": "true"}]},
            {"name": "@cf/meta/llama-3.1-8b-instruct", "task": {"name": "Text Generation"},
             "properties": [{"property_id": "context_window", "value": "8000"}]},
            {"name": "@cf/openai/gpt-oss-20b", "task": {"name": "Text Generation"},
             "properties": [{"property_id": "function_calling", "value": True}]},
        ]})

    provider = CloudflareProvider(CF_URL, CF_MODEL, api_key="dummy", account_id="acct")
    provider._client = httpx.AsyncClient(
        base_url=provider.base_url, headers=provider._client.headers, transport=httpx.MockTransport(handler)
    )
    models = asyncio.run(provider.list_models())
    assert models == ["@cf/openai/gpt-oss-20b", "@cf/zai-org/glm-4.7-flash"], models
    assert "/accounts/acct/ai/models/search" in seen[0] and "task=Text+Generation" in seen[0], seen

    # Sin informacion de function calling en ningun modelo: no se vacia la lista.
    rows = [{"name": "@cf/a", "task": {"name": "Text Generation"}}, {"name": "@cf/emb", "task": {"name": "Text Embeddings"}}]
    assert CloudflareProvider._select_models(rows) == ["@cf/a"]
    print("  cloudflare: modelos reales via /ai/models/search, solo con function calling")


def test_cloudflare_errors_are_actionable() -> None:
    import httpx

    from app.llm.openai_compat_provider import _error_text

    body = {"success": False, "errors": [{"code": 4006, "message": "you have used up your daily free allocation of 10,000 neurons"}]}
    detail = _error_text(httpx.Response(429, json=body))
    assert "codigo 4006" in detail and "neurons" in detail, detail

    provider = CloudflareProvider(CF_URL, CF_MODEL, api_key="dummy", account_id="acct")
    assert "00:00 UTC" in provider._explain_error(429, detail)
    assert "Workers AI" in provider._explain_error(401, "Authentication error (codigo 10000)")
    # El resto de proveedores no cambia sus mensajes.
    assert OpenAIProvider("https://api.openai.com/v1", "gpt-5", api_key="d")._explain_error(429, "x") == "x"
    print("  cloudflare: errores del free tier y de permisos con indicaciones")


NV_URL = "https://integrate.api.nvidia.com/v1"


def test_nvidia_thinking_via_template_kwargs() -> None:
    provider = NvidiaProvider(NV_URL, "nvidia/nemotron-3-super-120b-a12b", api_key="nvapi-dummy")
    base = {"model": "nvidia/nemotron-3-super-120b-a12b", "max_tokens": 4096, "temperature": 0.6}

    on = provider._adapt(dict(base), thinking=True)
    # Nemotron/Qwen leen `enable_thinking`; Kimi/GLM/DeepSeek, `thinking`.
    assert on["chat_template_kwargs"] == {"enable_thinking": True, "thinking": True}, on
    assert on["max_tokens"] == 4096 and "reasoning_effort" not in on and on["temperature"] == 0.6
    off = provider._adapt(dict(base), thinking=False)
    assert off["chat_template_kwargs"] == {"enable_thinking": False, "thinking": False}

    # Los que razonan siempre o nunca no reciben el parametro.
    assert "chat_template_kwargs" not in provider._adapt({"model": "openai/gpt-oss-20b"}, True)
    assert "chat_template_kwargs" not in provider._adapt({"model": "mistralai/mistral-nemotron"}, True)
    # Fuera del catalogo se intenta; si el endpoint lo rechazo, no se vuelve a mandar.
    assert "chat_template_kwargs" in provider._adapt({"model": "qwen/qwen3-next"}, True)
    provider._accepts_template_kwargs = False
    assert "chat_template_kwargs" not in provider._adapt(dict(base), True)
    print("  nvidia: razonamiento en chat_template_kwargs solo donde es conmutable")


def test_nvidia_model_filter() -> None:
    ids = [
        "nvidia/nemotron-3-super-120b-a12b", "nvidia/nv-embedqa-mistral-7b-v2", "moonshotai/kimi-k2.6",
        "nvidia/llama-3.1-nemoguard-8b-content-safety", "nvidia/nemotron-4-340b-reward",
        "nvidia/nemotron-parse", "snowflake/arctic-embed-l", "nvidia/riva-translate-4b-instruct",
        "z-ai/glm-5.3-flash",
    ]
    assert NvidiaProvider(NV_URL, "x", api_key="d")._filter_models(ids) == [
        "moonshotai/kimi-k2.6", "nvidia/nemotron-3-super-120b-a12b", "z-ai/glm-5.3-flash",
    ]
    print("  nvidia: fuera embeddings, guardarrailes, parsers, reward y traduccion")


def test_nvidia_retries_on_rate_limit() -> None:
    import httpx

    from app.llm import cloud_openai_providers as mod

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"detail": "Too Many Requests"})
        if calls["n"] == 2:
            return httpx.Response(500, json={"detail": "Internal server error"})
        if calls["n"] == 3:
            # El free tier tambien responde 503 cuando va saturado: transitorio.
            return httpx.Response(503, headers={"Retry-After": "0"}, json={"detail": "Service temporarily overloaded"})
        return httpx.Response(200, json={
            "model": "moonshotai/kimi-k2.6",
            "choices": [{"message": {"content": "hola"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        })

    provider = NvidiaProvider(NV_URL, "moonshotai/kimi-k2.6", api_key="nvapi-dummy")
    provider._client = httpx.AsyncClient(base_url=NV_URL, transport=httpx.MockTransport(handler))
    mod.NVIDIA_BASE_WAIT_S, saved_wait = 0.0, mod.NVIDIA_BASE_WAIT_S  # el 500 no trae Retry-After
    response = asyncio.run(provider.chat([{"role": "user", "content": "hola"}], thinking=False))
    assert response.content == "hola" and calls["n"] == 4, (response, calls)

    mod.NVIDIA_BASE_WAIT_S = saved_wait
    # Sin Retry-After: backoff exponencial acotado.
    no_header = httpx.Response(429)
    assert NvidiaProvider._retry_wait(no_header, 0) == mod.NVIDIA_BASE_WAIT_S
    assert NvidiaProvider._retry_wait(no_header, 10) == mod.NVIDIA_MAX_WAIT_S
    assert "40 peticiones" in provider._explain_error(429, "Too Many Requests")

    from app.llm.openai_compat_provider import _error_text

    forbidden = httpx.Response(403, json={"status": 403, "title": "Forbidden", "detail": "Authorization failed"})
    assert _error_text(forbidden) == "Forbidden: Authorization failed"
    assert "saturado" in provider._explain_error(503, "Service temporarily overloaded")
    print("  nvidia: 429, 500 y 503 del free tier se reintentan respetando Retry-After")


AWS_URL_TEMPLATE = "https://bedrock-runtime.{account_id}.amazonaws.com"
AWS_URL = "https://bedrock-runtime.us-east-1.amazonaws.com"
AWS_MODEL = "us.anthropic.claude-sonnet-4-6"


def _bedrock(handler=None, model: str = AWS_MODEL, url: str = AWS_URL) -> BedrockProvider:
    import httpx

    provider = BedrockProvider(url, model, api_key="dummy", temperature=0.6, top_p=0.95)
    if handler is not None:
        provider._client = httpx.AsyncClient(
            base_url=provider.base_url,
            headers=provider._client.headers,
            transport=httpx.MockTransport(handler),
        )
    return provider


def test_bedrock_region_resolution() -> None:
    """La region de la UI manda; si no llega, se usa AWS_REGION."""
    import os

    previous = os.environ.pop("AWS_REGION", None)
    try:
        typed = BedrockProvider(AWS_URL, AWS_MODEL, api_key="dummy")
        assert typed.has_region and typed.region == "us-east-1", typed.base_url

        os.environ["AWS_REGION"] = "eu-west-1"
        from_env = BedrockProvider(AWS_URL_TEMPLATE, AWS_MODEL, api_key="dummy")
        assert from_env.base_url == "https://bedrock-runtime.eu-west-1.amazonaws.com"
        del os.environ["AWS_REGION"]

        missing = BedrockProvider(AWS_URL_TEMPLATE, AWS_MODEL, api_key="dummy")
        assert not missing.has_region
        health = asyncio.run(missing.health())
        assert not health["ok"] and "region" in health["error"].lower(), health
        try:
            asyncio.run(missing.chat([{"role": "user", "content": "hola"}]))
        except MissingRegion:
            pass
        else:
            raise AssertionError("sin region no deberia llegar a hacer la peticion")
    finally:
        if previous is not None:
            os.environ["AWS_REGION"] = previous
    print("  aws: region desde la URL o el entorno; sin ella, error claro")


def test_bedrock_message_shape() -> None:
    """Converse: system aparte, toolResult agrupados dentro de un mensaje de usuario."""
    provider = _bedrock()

    system, rest = provider._split_system(CONVERSATION)
    assert system == [{"text": "Eres un agente de pruebas."}], system
    assert all(m["role"] != "system" for m in rest)

    serialized = provider._serialize_messages(rest)
    assert [m["role"] for m in serialized] == ["user", "assistant", "user"]

    # Los dos resultados van juntos, como en Claude: repartirlos le ensena al
    # modelo a dejar de pedir herramientas en paralelo.
    results = serialized[2]["content"]
    assert len(results) == 2 and all("toolResult" in b for b in results), results
    assert [b["toolResult"]["toolUseId"] for b in results] == ["call_a", "call_b"]
    assert results[0]["toolResult"]["content"] == [{"text": '{"stock": 7}'}]

    blocks = serialized[1]["content"]
    assert blocks[0] == {"text": "Voy a mirarlo."}
    assert [b["toolUse"]["name"] for b in blocks[1:]] == ["consultar_inventario", "hora_actual"]
    assert blocks[1]["toolUse"]["input"] == {"sku": "SKU-002"}
    print("  aws: system aparte, toolResult agrupados y toolUse correctos")


def test_bedrock_preserves_reasoning_blocks() -> None:
    """El razonamiento viene firmado y hay que devolverlo intacto."""
    provider = _bedrock()
    original = [
        {"reasoningContent": {"reasoningText": {"text": "...", "signature": "abc123"}}},
        {"toolUse": {"toolUseId": "call_a", "name": "x", "input": {}}},
    ]
    serialized = provider._serialize_messages(
        [
            {"role": "user", "content": "hola"},
            {"role": "assistant", "content": "", "tool_calls": [], "_provider_blocks": original},
            {"role": "tool", "content": "ok", "tool_call_id": "call_a", "name": "x"},
        ]
    )
    assert serialized[1]["content"] is original, "se perdieron los bloques firmados"
    print("  aws: bloques reasoningContent firmados preservados")


def test_bedrock_reasoning_by_family() -> None:
    """Converse no tiene campo comun: cada familia lo deletrea a su manera."""
    provider = _bedrock()

    assert provider._reasoning(AWS_MODEL, True) == {"thinking": {"type": "adaptive"}}
    assert provider._reasoning(AWS_MODEL, False) == {"thinking": {"type": "disabled"}}
    # El prefijo de region no cambia la familia.
    assert provider._reasoning("anthropic.claude-sonnet-4-6", True) == {"thinking": {"type": "adaptive"}}

    nova = provider._reasoning("us.amazon.nova-2-lite-v1:0", True)
    assert nova == {"reasoningConfig": {"type": "enabled"}}, nova

    # DeepSeek razona siempre y no lo expone; Qwen y gpt-oss no lo documentan.
    # Mandar una clave que el esquema del modelo no conoce es un 400 seguro.
    for model in ("deepseek.v3.2", "qwen.qwen3-coder-next", "openai.gpt-oss-120b-1:0"):
        assert provider._reasoning(model, True) == {}, model

    # El catalogo puede vetarlo para un modelo concreto (Haiku 4.5 usa el
    # dialecto antiguo, que los nuevos ya rechazan).
    assert provider._reasoning("us.anthropic.claude-haiku-4-5-20251001-v1:0", True) == {}

    provider._drop_reasoning = True
    assert provider._reasoning(AWS_MODEL, True) == {}
    print("  aws: thinking en Claude, reasoningConfig en Nova, nada en el resto")


def test_bedrock_converse_roundtrip() -> None:
    """Payload de ida y parseo de vuelta contra una respuesta real de Converse."""
    import httpx

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.headers["authorization"] == "Bearer dummy"
        return httpx.Response(200, json={
            "output": {"message": {"role": "assistant", "content": [
                {"reasoningContent": {"reasoningText": {"text": "lo pienso", "signature": "s1"}}},
                {"text": "Voy a mirarlo."},
                {"toolUse": {"toolUseId": "tu_1", "name": "consultar_inventario",
                             "input": {"sku": "SKU-002"}}},
            ]}},
            "stopReason": "tool_use",
            "usage": {"inputTokens": 120, "outputTokens": 30, "totalTokens": 150},
        })

    provider = _bedrock(handler)
    response = asyncio.run(provider.chat(CONVERSATION, TOOLS, thinking=True))

    body = json.loads(seen[0].content)
    assert seen[0].url.path == f"/model/{AWS_MODEL}/converse", seen[0].url
    assert body["system"] == [{"text": "Eres un agente de pruebas."}]
    assert body["inferenceConfig"]["temperature"] == 0.6 and body["inferenceConfig"]["topP"] == 0.95
    assert body["additionalModelRequestFields"] == {"thinking": {"type": "adaptive"}}
    # Las herramientas MCP van con el esquema bajo inputSchema.json.
    tool = body["toolConfig"]["tools"][0]["toolSpec"]
    assert set(tool) == {"name", "description", "inputSchema"}, tool
    assert tool["inputSchema"]["json"] == {"type": "object", "properties": {}}
    assert body["toolConfig"]["toolChoice"] == {"auto": {}}

    assert response.content == "Voy a mirarlo."
    assert response.thinking == "lo pienso"
    assert response.finish_reason == "tool_use"
    assert [c.name for c in response.tool_calls] == ["consultar_inventario"]
    assert response.tool_calls[0].id == "tu_1"
    assert response.tool_calls[0].arguments == {"sku": "SKU-002"}
    assert response.usage.total_tokens == 150 and response.usage.prompt_tokens == 120
    # Los bloques se conservan para devolverlos firmados en el turno siguiente.
    assert response.raw_blocks[0]["reasoningContent"]["reasoningText"]["signature"] == "s1"
    print("  aws: toolConfig de ida, toolUse/reasoningContent de vuelta")


def test_bedrock_model_id_is_escaped() -> None:
    """Los identificadores con version (`...-1:0`) llevan los dos puntos escapados."""
    import httpx

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.raw_path.decode())
        return httpx.Response(200, json={"output": {"message": {"content": [{"text": "ok"}]}},
                                         "stopReason": "end_turn", "usage": {}})

    provider = _bedrock(handler, model="openai.gpt-oss-120b-1:0")
    asyncio.run(provider.chat([{"role": "user", "content": "hola"}]))
    assert seen[0] == "/model/openai.gpt-oss-120b-1%3A0/converse", seen
    print("  aws: el identificador del modelo va escapado en la ruta")


def test_bedrock_retries_without_rejected_fields() -> None:
    """Si el modelo rechaza el campo de razonamiento, se retira y se recuerda."""
    import httpx

    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "additionalModelRequestFields" in body:
            return httpx.Response(400, json={
                "message": "Malformed input request: #/thinking: subject must not be valid"})
        return httpx.Response(200, json={"output": {"message": {"content": [{"text": "ok"}]}},
                                         "stopReason": "end_turn", "usage": {}})

    provider = _bedrock(handler)
    response = asyncio.run(provider.chat([{"role": "user", "content": "hola"}], thinking=True))
    assert response.content == "ok" and len(bodies) == 2, bodies
    assert "additionalModelRequestFields" not in bodies[1]

    # Recordado: la siguiente peticion ya no lo manda ni paga el rechazo.
    asyncio.run(provider.chat([{"role": "user", "content": "otra"}], thinking=True))
    assert len(bodies) == 3 and "additionalModelRequestFields" not in bodies[2]
    print("  aws: el campo de razonamiento rechazado se retira y no vuelve")


def test_bedrock_lists_profiles_and_on_demand_models() -> None:
    """El desplegable necesita las dos listas: el prefijo `us.` no es universal."""
    import httpx

    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        if request.url.path == "/inference-profiles":
            if request.url.params.get("nextToken"):
                return httpx.Response(200, json={"inferenceProfileSummaries": [
                    {"inferenceProfileId": "us.amazon.nova-2-lite-v1:0", "status": "ACTIVE"}]})
            return httpx.Response(200, json={
                "inferenceProfileSummaries": [
                    {"inferenceProfileId": "us.anthropic.claude-sonnet-4-6", "status": "ACTIVE"},
                    {"inferenceProfileId": "us.anthropic.viejo", "status": "INACTIVE"},
                ],
                "nextToken": "pagina2",
            })
        return httpx.Response(200, json={"modelSummaries": [
            {"modelId": "deepseek.v3.2", "modelLifecycle": {"status": "ACTIVE"}},
            {"modelId": "amazon.titan-retirado", "modelLifecycle": {"status": "LEGACY"}},
        ]})

    provider = _bedrock(handler)
    models = asyncio.run(provider.list_models())
    assert models == [
        "deepseek.v3.2", "us.amazon.nova-2-lite-v1:0", "us.anthropic.claude-sonnet-4-6",
    ], models

    # El listado vive en el plano de control, otro host que la inferencia.
    assert all(u.host == "bedrock.us-east-1.amazonaws.com" for u in seen), seen
    assert seen[0].params["type"] == "SYSTEM_DEFINED"
    assert seen[-1].params["byOutputModality"] == "TEXT"
    print("  aws: perfiles paginados + modelos bajo demanda, sin los retirados")


def test_bedrock_errors_are_actionable() -> None:
    import httpx

    provider = _bedrock()
    profile = provider._explain_error(
        400, "Invocation of model ID anthropic.claude-haiku-4-5 with on-demand throughput isn't supported"
    )
    assert "perfil de inferencia" in profile, profile
    assert "formulario de acceso" in provider._explain_error(403, "AccessDeniedException")
    assert "bedrock:ListInferenceProfiles" in provider._explain_listing_error(403, "denied")

    # El cuerpo de error de AWS es {"message": ...} y el tipo va en cabecera.
    response = httpx.Response(
        400, headers={"x-amzn-errortype": "ValidationException:http://internal"},
        json={"message": "Malformed input request"},
    )
    assert _error_text(response) == "ValidationException: Malformed input request"
    print("  aws: perfiles de inferencia, permisos de listado y errores de AWS traducidos")


def test_catalog_consistency() -> None:
    for name, info in CATALOG.items():
        assert info.name == name
        if info.needs_api_key:
            assert info.key_env, f"{name} sin variable de entorno para la clave"
        for m in info.models:
            assert model_info(name, m.id) is m
    recommended = [m.id for m in CATALOG["anthropic"].models if m.recommended]
    assert recommended == ["claude-opus-5"], recommended
    print(f"  catalogo: {sum(len(p.models) for p in CATALOG.values())} modelos en {len(CATALOG)} proveedores")


if __name__ == "__main__":
    for test in [
        test_anthropic_messages,
        test_anthropic_preserves_raw_blocks,
        test_anthropic_thinking_mapping,
        test_anthropic_payload,
        test_openai_reasoning_params,
        test_google_keeps_max_tokens,
        test_cloud_never_inherits_local_url,
        test_gemini_schema_sanitizer,
        test_gemini_message_shape,
        test_gemini_effort_value,
        test_gemini_model_filter,
        test_gemini_tool_without_arguments,
        test_gemini_thought_signature_roundtrip,
        test_gemini3_effort_and_rejection_memory,
        test_thought_signature_never_leaks_to_other_providers,
        test_cloudflare_account_id_resolution,
        test_cloudflare_payload,
        test_cloudflare_lists_models_with_function_calling,
        test_cloudflare_errors_are_actionable,
        test_nvidia_thinking_via_template_kwargs,
        test_nvidia_model_filter,
        test_nvidia_retries_on_rate_limit,
        test_bedrock_region_resolution,
        test_bedrock_message_shape,
        test_bedrock_preserves_reasoning_blocks,
        test_bedrock_reasoning_by_family,
        test_bedrock_converse_roundtrip,
        test_bedrock_model_id_is_escaped,
        test_bedrock_retries_without_rejected_fields,
        test_bedrock_lists_profiles_and_on_demand_models,
        test_bedrock_errors_are_actionable,
        test_catalog_consistency,
    ]:
        print(f"{test.__name__}:")
        test()
    print("\nTODO OK")
