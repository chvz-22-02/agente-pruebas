"""Comprobaciones deterministas, agregacion de la nota y parseo del evaluador.

El reparto de responsabilidades es deliberado: el evaluador LLM solo decide,
criterio a criterio, si se cumple o no (y por que). La nota y el aprobado los
calcula este modulo con los pesos de la rubrica, de forma reproducible. Asi un
cambio de modelo evaluador cambia los juicios, pero nunca la aritmetica.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .spec import Case, criterios_de

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


def expected_numbers(text: str) -> list[float]:
    """Cifras de un ``valor_esperado`` escrito en texto libre.

    Lo que hay en el banco de pruebas va del dato suelto (``30646``) a la serie
    (``863, 933, 4907``), el rango (``30.0% - 33.8%``) o ninguna cifra en
    absoluto (``empleo o mercado laboral``, ``No aplica``). Aqui se lee el
    fichero, no la respuesta del agente, asi que se toma la lectura literal:
    punto decimal y nada de separadores de miles.
    """
    values: list[float] = []
    for match in NUMBER_TOKEN.finditer(text or ""):
        token = match.group().strip(".,")
        # En "2020-2024" el guion separa un rango, no marca un negativo.
        if token.startswith("-") and match.start() and text[match.start() - 1].isdigit():
            token = token[1:]
        try:
            value = float(token)
        except ValueError:
            continue
        if value not in values:
            values.append(value)
    return values


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


def deterministic_checks(case: Case, agent_texts: list[str]) -> list[dict[str, Any]]:
    """Verificaciones que no necesitan a ningun LLM.

    Solo hay una, y solo cuando el ``valor_esperado`` de la consulta trae
    cifras: que esas cifras aparezcan en lo que respondio el agente. El resto
    del ``valor_esperado`` (un ambito tematico, un "No aplica") no se puede
    comprobar con una expresion regular y lo juzga el evaluador.
    """
    expected = expected_numbers(case.valor_esperado)
    if not expected:
        return []
    answers = "\n".join(agent_texts)
    missing = [value for value in expected if not value_mentioned(answers, value, 0.0)[0]]
    listed = ", ".join(f"{value:g}" for value in expected)
    return [
        _check(
            "valor_mencionado",
            f"Las respuestas mencionan {listed}",
            not missing,
            f"{len(expected) - len(missing)} de {len(expected)}"
            + (f"; faltan {', '.join(f'{v:g}' for v in missing)}" if missing else ""),
        )
    ]


# --------------------------------------------------------------- agregacion --
def judged_items(case: Case, verdict: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Cruza la rubrica de la consulta con lo que dijo el evaluador.

    Un criterio sobre el que el evaluador no se pronuncia cuenta como no
    cumplido: es la lectura conservadora, y se marca para que se vea.
    """
    verdict = verdict or {}
    raw_criteria = verdict.get("criterios") if isinstance(verdict.get("criterios"), list) else []
    by_id = {
        str(item.get("id")): item for item in raw_criteria if isinstance(item, dict) and item.get("id")
    }
    items: list[dict[str, Any]] = []
    for criterion in criterios_de(case):
        item = by_id.get(criterion.id)
        items.append(
            {
                "id": criterion.id,
                "origen": "evaluador",
                "descripcion": criterion.descripcion,
                "cumple": _truthy(item.get("cumple")) if item else False,
                "detalle": (item or {}).get("justificacion") or "El evaluador no se pronuncio",
                "valor_reportado": (item or {}).get("valor_reportado"),
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
