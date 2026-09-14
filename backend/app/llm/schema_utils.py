"""Adaptacion de esquemas JSON Schema al subconjunto que acepta Gemini.

Los servidores MCP declaran sus herramientas con JSON Schema completo: los
generados con pydantic traen `$schema`, `$defs`/`$ref`, `additionalProperties`,
`anyOf` con `const`, `exclusiveMinimum`, formatos como `uri` o `uuid`...

Gemini solo admite un subconjunto de OpenAPI 3.0 y responde 400 ante cualquier
clave que no conozca, lo que hace fallar la conversacion desde el primer
mensaje. Aqui se traduce el esquema a ese subconjunto:

* se resuelven los `$ref` contra `$defs` / `definitions`,
* `allOf` se fusiona y `oneOf` pasa a `anyOf`,
* `const` se convierte en un `enum` de un elemento,
* se descarta todo lo que no este en la lista de claves permitidas.

Se usa una lista de permitidos y no de prohibidos a proposito: ante un esquema
raro es preferible perder una restriccion (el modelo sigue funcionando) que
enviar una clave desconocida (la peticion entera falla).
"""

from __future__ import annotations

from typing import Any

# Claves del subconjunto OpenAPI 3.0 que acepta Gemini.
ALLOWED_KEYS = frozenset(
    {
        "type", "format", "description", "nullable", "enum", "items",
        "properties", "required", "anyOf", "default", "example",
        "minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems",
        "pattern", "propertyOrdering",
    }
)

# `format` solo admite estos valores; el resto ("uri", "uuid", "email"...)
# provoca un rechazo aunque el tipo sea correcto.
ALLOWED_FORMATS = {
    "string": {"date-time", "enum"},
    "integer": {"int32", "int64"},
    "number": {"float", "double"},
}

MAX_DEPTH = 12


def _merge(target: dict[str, Any], extra: dict[str, Any]) -> None:
    """Fusiona un subesquema de `allOf` sin pisar lo ya definido."""
    for key, value in extra.items():
        if key == "properties" and isinstance(value, dict):
            target.setdefault("properties", {}).update(value)
        elif key == "required" and isinstance(value, list):
            merged = list(dict.fromkeys([*target.get("required", []), *value]))
            target["required"] = merged
        else:
            target.setdefault(key, value)


def _resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any] | None:
    """Resuelve una referencia local del tipo `#/$defs/Nombre`."""
    if not ref.startswith("#/"):
        return None
    parts = [p for p in ref.lstrip("#/").split("/") if p]
    # El nombre util es el ultimo segmento; los intermedios son $defs/definitions.
    return defs.get(parts[-1]) if parts else None


def sanitize(schema: Any, defs: dict[str, Any] | None = None, depth: int = 0) -> dict[str, Any]:
    """Devuelve una copia del esquema con solo lo que Gemini entiende."""
    if not isinstance(schema, dict):
        return {"type": "string"}
    if depth > MAX_DEPTH:
        # Esquema recursivo: se corta con algo valido en lugar de desbordar.
        return {"type": "object", "description": "estructura anidada"}

    defs = {**(defs or {}), **(schema.get("$defs") or {}), **(schema.get("definitions") or {})}

    node = dict(schema)

    if ref := node.pop("$ref", None):
        resolved = _resolve_ref(ref, defs)
        if resolved is None:
            return {"type": "object", "description": f"referencia no resuelta: {ref}"}
        merged = dict(resolved)
        merged.update({k: v for k, v in node.items() if k != "$ref"})
        return sanitize(merged, defs, depth + 1)

    for combinator in ("allOf", "oneOf"):
        parts = node.pop(combinator, None)
        if not isinstance(parts, list):
            continue
        if combinator == "allOf":
            for part in parts:
                if isinstance(part, dict):
                    _merge(node, sanitize(part, defs, depth + 1))
        else:
            # `oneOf` es exclusivo y `anyOf` no, pero Gemini solo conoce anyOf
            # y para guiar al modelo la diferencia es irrelevante.
            node.setdefault("anyOf", parts)

    if "const" in node:
        node["enum"] = [node.pop("const")]

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key not in ALLOWED_KEYS:
            continue  # $schema, additionalProperties, title, $id, exclusive*...

        if key == "properties" and isinstance(value, dict):
            out["properties"] = {k: sanitize(v, defs, depth + 1) for k, v in value.items()}
        elif key == "items":
            out["items"] = sanitize(value, defs, depth + 1)
        elif key == "anyOf" and isinstance(value, list):
            # `anyOf: [X, {"type": "null"}]` es como pydantic marca opcional:
            # Gemini lo expresa con `nullable`, no con una rama de tipo null.
            branches = [b for b in value if isinstance(b, dict)]
            non_null = [b for b in branches if b.get("type") != "null"]
            if len(non_null) < len(branches):
                out["nullable"] = True
            if len(non_null) == 1:
                out.update(sanitize(non_null[0], defs, depth + 1))
            elif non_null:
                out["anyOf"] = [sanitize(b, defs, depth + 1) for b in non_null]
        elif key == "required" and isinstance(value, list):
            out["required"] = [r for r in value if isinstance(r, str)]
        else:
            out[key] = value

    if "type" in out and isinstance(out["type"], list):
        # `type: ["string", "null"]` -> tipo simple + nullable.
        types = [t for t in out["type"] if t != "null"]
        out["nullable"] = out.get("nullable", len(types) < len(out["type"]))
        out["type"] = types[0] if types else "string"

    fmt = out.get("format")
    if fmt is not None and fmt not in ALLOWED_FORMATS.get(out.get("type", ""), set()):
        out.pop("format")

    if "properties" in out or "required" in out:
        out.setdefault("type", "object")
        # Una propiedad requerida que no existe hace fallar la validacion.
        if "required" in out:
            known = set(out.get("properties", {}))
            out["required"] = [r for r in out["required"] if r in known]
            if not out["required"]:
                out.pop("required")
    elif "items" in out:
        out.setdefault("type", "array")
    elif "type" not in out and "anyOf" not in out:
        out["type"] = "string"

    return out


def sanitize_tool_schema(schema: Any) -> dict[str, Any]:
    """Punto de entrada para el `input_schema` de una herramienta MCP."""
    result = sanitize(schema or {})
    result.setdefault("type", "object")
    if result["type"] == "object":
        result.setdefault("properties", {})
    return result
