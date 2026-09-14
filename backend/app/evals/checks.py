"""Comprobaciones deterministas, agregacion de la nota y parseo del evaluador.

El reparto de responsabilidades es deliberado: el evaluador LLM solo decide,
criterio a criterio, si se cumple o no (y por que). La nota y el aprobado los
calcula este modulo con los pesos del YAML, de forma reproducible. Asi un
cambio de modelo evaluador cambia los juicios, pero nunca la aritmetica.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .spec import Case

NUMBER_TOKEN = re.compile(r"-?\d[\d.,]*\d|-?\d")
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
CODE_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


# ------------------------------------------------------------------ numeros --
def _interpretations(token: str) -> set[float]:
    """Lecturas posibles de una cifra escrita a la espanola o a la inglesa.

    ``24,9`` -> 24.9 · ``1.234,5`` -> 1234.5 · ``1,234.5`` -> 1234.5. Ante la
    duda (``1.234``) se aceptan ambas: la comprobacion es "se menciona la
    cifra", y ser generoso aqui es preferible a un falso negativo.
    """
    values: set[float] = set()
    candidates = {
        token.replace(",", ""),  # coma como separador de miles
        token.replace(".", "").replace(",", "."),  # punto como separador de miles
        token.replace(",", "."),  # coma decimal sin miles
    }
    for candidate in candidates:
        try:
            values.add(float(candidate))
        except ValueError:
            continue
    return values


def numbers_in(text: str) -> set[float]:
    found: set[float] = set()
    for token in NUMBER_TOKEN.findall(text or ""):
        found |= _interpretations(token.strip(".,"))
    return found


def value_mentioned(text: str, expected: float, tolerance: float) -> tuple[bool, float | None]:
    """Devuelve si alguna cifra del texto cae dentro de la tolerancia y la mas cercana."""
    closest: float | None = None
    for value in numbers_in(text):
        if closest is None or abs(value - expected) < abs(closest - expected):
            closest = value
    ok = closest is not None and abs(closest - expected) <= max(tolerance, 1e-9)
    return ok, closest


# ------------------------------------------------------------ herramientas --
def _tool_matches(event: dict[str, Any], wanted: str) -> bool:
    """Coincide con el nombre expuesto, el real o el prefijado por servidor."""
    target = wanted.strip().lower()
    names = {
        (event.get("tool_name") or "").lower(),
        (event.get("real_tool_name") or "").lower(),
    }
    return target in names or any(n.endswith(f"__{target}") for n in names)


def _check(
    check_id: str, description: str, ok: bool, detail: str, weight: float = 1.0, mandatory: bool = False
) -> dict[str, Any]:
    return {
        "id": check_id,
        "origen": "determinista",
        "descripcion": description,
        "cumple": bool(ok),
        "detalle": detail,
        "peso": weight,
        "obligatorio": mandatory,
    }


def deterministic_checks(
    case: Case, agent_texts: list[str], tool_events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Verificaciones que no necesitan a ningun LLM."""
    checks: list[dict[str, Any]] = []
    expected = case.resultado_esperado
    tools = case.herramientas
    answers = "\n".join(agent_texts)
    lowered = answers.lower()
    used = sorted({e.get("tool_name", "") for e in tool_events})

    for name in tools.debe_usar:
        hits = [e for e in tool_events if _tool_matches(e, name)]
        checks.append(
            _check(
                f"usa:{name}",
                f"Usa la herramienta '{name}'",
                bool(hits),
                f"{len(hits)} llamada(s)" if hits else f"no se llamo; usadas: {', '.join(used) or 'ninguna'}",
            )
        )
    for name in tools.no_debe_usar:
        hits = [e for e in tool_events if _tool_matches(e, name)]
        checks.append(
            _check(
                f"no_usa:{name}",
                f"No usa la herramienta '{name}'",
                not hits,
                f"se llamo {len(hits)} vez/veces" if hits else "no se llamo",
            )
        )
    if tools.max_llamadas is not None:
        checks.append(
            _check(
                "max_llamadas",
                f"Como mucho {tools.max_llamadas} llamadas al MCP",
                len(tool_events) <= tools.max_llamadas,
                f"{len(tool_events)} llamadas",
            )
        )
    if tools.sin_errores:
        failed = [e for e in tool_events if not e.get("ok", True)]
        checks.append(
            _check(
                "sin_errores_mcp",
                "Ninguna llamada al MCP devuelve error",
                not failed,
                "; ".join(f"{e.get('tool_name')}: {(e.get('error') or '')[:120]}" for e in failed[:3])
                or "sin errores",
            )
        )

    if expected.tipo == "valor" and isinstance(expected.valor, float):
        ok, closest = value_mentioned(answers, expected.valor, expected.tolerancia)
        checks.append(
            _check(
                "valor_mencionado",
                f"La respuesta menciona {expected.valor:g}{expected.unidad and ' ' + expected.unidad}"
                + (f" (±{expected.tolerancia:g})" if expected.tolerancia else ""),
                ok,
                f"cifra mas cercana en las respuestas: {closest:g}" if closest is not None
                else "no hay cifras en las respuestas",
            )
        )
    if expected.tipo == "texto" and isinstance(expected.valor, str) and expected.valor.strip():
        ok = expected.valor.strip().lower() in lowered
        checks.append(
            _check("texto_mencionado", f"La respuesta contiene '{expected.valor}'", ok,
                   "encontrado" if ok else "no encontrado")
        )
    for fragment in expected.debe_contener:
        ok = fragment.lower() in lowered
        checks.append(
            _check(f"contiene:{fragment}", f"La respuesta contiene '{fragment}'", ok,
                   "encontrado" if ok else "no encontrado")
        )
    for fragment in expected.no_debe_contener:
        ok = fragment.lower() not in lowered
        checks.append(
            _check(f"no_contiene:{fragment}", f"La respuesta no contiene '{fragment}'", ok,
                   "ausente" if ok else "aparece en la respuesta")
        )
    return checks


# --------------------------------------------------------------- agregacion --
def judged_items(case: Case, verdict: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Cruza la rubrica del YAML con lo que dijo el evaluador.

    Un criterio sobre el que el evaluador no se pronuncia cuenta como no
    cumplido: es la lectura conservadora, y se marca para que se vea.
    """
    verdict = verdict or {}
    raw_criteria = verdict.get("criterios") if isinstance(verdict.get("criterios"), list) else []
    by_id = {
        str(item.get("id")): item for item in raw_criteria if isinstance(item, dict) and item.get("id")
    }
    expected = case.resultado_esperado
    items: list[dict[str, Any]] = []

    judged = verdict.get("resultado_esperado") if isinstance(verdict.get("resultado_esperado"), dict) else None
    items.append(
        {
            "id": "resultado_esperado",
            "origen": "evaluador",
            "descripcion": expected.descripcion,
            "cumple": _truthy(judged.get("cumple")) if judged else False,
            "detalle": (judged or {}).get("justificacion") or "El evaluador no se pronuncio",
            "valor_reportado": (judged or {}).get("valor_reportado"),
            "peso": expected.peso,
            "obligatorio": expected.obligatorio,
            "sin_juicio": judged is None,
        }
    )
    for criterion in case.criterios:
        item = by_id.get(criterion.id)
        items.append(
            {
                "id": criterion.id,
                "origen": "evaluador",
                "descripcion": criterion.descripcion,
                "cumple": _truthy(item.get("cumple")) if item else False,
                "detalle": (item or {}).get("justificacion") or "El evaluador no se pronuncio",
                "peso": criterion.peso,
                "obligatorio": criterion.obligatorio,
                "sin_juicio": item is None,
            }
        )
    return items


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "si", "sí", "yes", "1", "cumple", "aprobado"}
    return False


def aggregate(items: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    total = sum(max(float(i.get("peso", 1.0)), 0.0) for i in items)
    earned = sum(max(float(i.get("peso", 1.0)), 0.0) for i in items if i.get("cumple"))
    score = earned / total if total > 0 else 0.0
    failed_mandatory = [i["id"] for i in items if i.get("obligatorio") and not i.get("cumple")]
    return {
        "score": round(score, 4),
        "passed": score >= threshold and not failed_mandatory,
        "threshold": threshold,
        "failed_mandatory": failed_mandatory,
        "items_total": len(items),
        "items_passed": sum(1 for i in items if i.get("cumple")),
    }


# ------------------------------------------------------------------- JSON ----
def extract_json(text: str) -> dict[str, Any] | None:
    """Saca el objeto JSON de la respuesta del evaluador.

    Los modelos locales envuelven el JSON en bloques de codigo, lo preceden de
    una frase o dejan restos de razonamiento: se prueban esas variantes antes
    de darlo por invalido.
    """
    if not text:
        return None
    cleaned = THINK_BLOCK.sub("", text)
    # Razonamiento sin etiqueta de apertura (modelos que razonan siempre).
    cleaned = cleaned.rsplit("</think>", 1)[-1].strip()
    candidates = [cleaned, *CODE_FENCE.findall(cleaned)]
    for candidate in candidates:
        try:
            parsed = json.loads(candidate.strip())
            if isinstance(parsed, dict):
                return parsed
        except (TypeError, ValueError):
            pass
    for block in _balanced_objects(cleaned):
        try:
            parsed = json.loads(block)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _balanced_objects(text: str) -> list[str]:
    """Objetos ``{...}`` de primer nivel, del mas largo al mas corto."""
    found: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                found.append(text[start : index + 1])
    return sorted(found, key=len, reverse=True)
