"""Pruebas de la traduccion al dialecto de cada proveedor de nube.

No hacen ninguna llamada de red ni necesitan claves: comprueban como se
construye el payload, que es donde estan las diferencias que rompen.

    python tests\\test_cloud_providers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm.anthropic_provider import AnthropicProvider  # noqa: E402
from app.llm.base import ToolSpec  # noqa: E402
from app.llm.catalog import CATALOG, model_info  # noqa: E402
from app.llm.cloud_openai_providers import GoogleProvider, OpenAIProvider  # noqa: E402
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

    for name in ("anthropic", "openai", "google"):
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
        test_catalog_consistency,
    ]:
        print(f"{test.__name__}:")
        test()
    print("\nTODO OK")
