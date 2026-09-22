"""Estructura de los ficheros JSON de evaluacion y su validacion.

Hay dos ficheros independientes, para poder combinar el mismo juego de
personas con distintos bancos de consultas (y al reves):

* ``personas.json``:  los perfiles que interpreta el simulador, indexados por
  su identificador.
* ``consultas.json``: el banco de pruebas. Cada consulta nombra a la persona
  que la formula, su objetivo, el primer mensaje y lo que deberia salir.

Los modelos rechazan claves desconocidas (``extra="forbid"``) a proposito: en
un fichero escrito a mano, ``valor-esperado`` en lugar de ``valor_esperado`` no
debe pasar en silencio y dejar la consulta sin dato que comprobar.

La rubrica no viaja en los ficheros: es la misma para todo el banco y vive en
``CRITERIOS`` / ``criterios_de``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

ID_PATTERN = r"^[A-Za-z0-9_.-]+$"
MAX_TURNS_LIMIT = 30
DEFAULT_MAX_TURNS = 6
DEFAULT_THRESHOLD = 0.7


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------ personas -
class Persona(_Strict):
    """Un perfil de ``personas.json``; el ``id`` es la clave del objeto."""

    id: str = Field(pattern=ID_PATTERN)
    nombre: str
    # Quien es, que sabe y como conversa: va tal cual al prompt del simulador.
    descripcion: str = ""


# ----------------------------------------------------------------- consultas -
Ambiguity = Literal["alta", "media", "baja", ""]


class Case(_Strict):
    """Una consulta del banco de pruebas."""

    id: str = Field(pattern=ID_PATTERN)
    # Quien la formula; tiene que existir en personas.json.
    persona: str
    # Lo que la persona quiere conseguir. Lo recibe el simulador, no el agente.
    goal: str
    # Primer mensaje literal. Vacio => lo redacta el simulador.
    consulta_inicial: str = ""
    # Cuanto concreta la persona al preguntar; modula el prompt del simulador.
    ambiguedad: Ambiguity = ""
    # Que deberia ocurrir, en texto libre ("Existe un valor y se devuelve...").
    resultado_esperado: str = ""
    # El dato esperado, en texto libre: una cifra, una serie, un rango, un
    # ambito tematico, o "No aplica" cuando no deberia existir.
    valor_esperado: str = ""

    @field_validator("ambiguedad", mode="before")
    @classmethod
    def normalize_ambiguity(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @property
    def label(self) -> str:
        return self.id


# ------------------------------------------------------------------ rubrica --
class Criterion(BaseModel):
    id: str
    descripcion: str
    peso: float = 1.0
    # Si no se cumple, el caso suspende aunque la nota llegue al umbral.
    obligatorio: bool = False


# Los criterios que no dependen de la consulta. La nota es la suma de los pesos
# cumplidos entre la suma de los pesos, asi que el peso es lo que gradua cuanto
# cuenta cada cosa.
CRITERIOS = [
    Criterion(
        id="sin_invenciones",
        descripcion=(
            "Toda cifra o dato que el agente afirma esta respaldado por el resultado de una "
            "herramienta del MCP: no rellena huecos con conocimiento propio."
        ),
        peso=2.0,
        obligatorio=True,
    ),
    Criterion(
        id="respuesta_util",
        descripcion=(
            "La respuesta contesta a lo que pide la persona y esta redactada a su nivel, "
            "sin dejarla con el trabajo a medias."
        ),
        peso=1.0,
    ),
]


def criterios_de(case: Case) -> list[Criterion]:
    """Rubrica de una consulta: lo suyo primero, y despues los criterios fijos."""
    rubric = [
        Criterion(
            id="resultado_esperado",
            descripcion=case.resultado_esperado.strip() or "El agente consigue lo que la persona buscaba.",
            peso=3.0,
            obligatorio=True,
        )
    ]
    if case.valor_esperado.strip():
        rubric.append(
            Criterion(
                id="valor_esperado",
                descripcion=f"El dato que da el agente se corresponde con: {case.valor_esperado.strip()}",
                peso=3.0,
            )
        )
    return rubric + CRITERIOS


# --------------------------------------------------------------- resultado ---
@dataclass
class SuiteSpec:
    """Resultado de validar la pareja de ficheros."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    personas: list[Persona] = field(default_factory=list)
    cases: list[Case] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "personas": [
                {"id": p.id, "nombre": p.nombre, "descripcion": p.descripcion} for p in self.personas
            ],
            "cases": [
                {
                    "id": c.id,
                    "persona": c.persona,
                    "goal": c.goal,
                    "consulta_inicial": c.consulta_inicial,
                    "ambiguedad": c.ambiguedad,
                    "resultado_esperado": c.resultado_esperado,
                    "valor_esperado": c.valor_esperado,
                }
                for c in self.cases
            ],
            "matrix": len(build_matrix(self)) if self.ok else 0,
        }


@dataclass
class EvalItemSpec:
    """Una ejecucion concreta: una consulta, con su persona, en una repeticion."""

    case: Case
    persona: Persona
    repetition: int
    max_turns: int
    threshold: float


# ------------------------------------------------------------------ errores --
_MESSAGES = {
    "missing": "campo obligatorio",
    "extra_forbidden": "clave no reconocida (¿error de escritura?)",
    "string_pattern_mismatch": "solo se admiten letras, numeros y los simbolos _ . -",
    "string_type": "debe ser texto",
    "list_type": "debe ser una lista",
    "int_parsing": "debe ser un numero entero",
    "float_parsing": "debe ser un numero",
    "bool_parsing": "debe ser true o false",
}


def _describe(exc: ValidationError, source: str) -> list[str]:
    out = []
    for err in exc.errors():
        field_path = ".".join(str(part) for part in err["loc"])
        message = _MESSAGES.get(err["type"])
        if err["type"] == "literal_error":
            expected = err.get("ctx", {}).get("expected", "")
            message = f"valor no permitido; usa uno de: {expected}"
        out.append(f"{source}{'.' + field_path if field_path else ''}: {message or err['msg']}")
    return out


def _load_json(text: str, source: str, errors: list[str]) -> Any:
    if not text or not text.strip():
        errors.append(f"{source}: el fichero esta vacio")
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        errors.append(f"{source}: JSON invalido (linea {exc.lineno}, columna {exc.colno}): {exc.msg}")
        return None


def _parse_personas(raw: Any, errors: list[str]) -> list[Persona]:
    if not isinstance(raw, dict):
        errors.append('personas.json: la raiz debe ser un objeto {"id_de_persona": {...}}')
        return []
    if not raw:
        errors.append("personas.json: no hay ninguna persona")
        return []
    personas: list[Persona] = []
    for key, value in raw.items():
        source = f"personas.json · {key}"
        if not isinstance(value, dict):
            errors.append(f"{source}: debe ser un objeto con 'nombre' y 'descripcion'")
            continue
        try:
            # La clave manda sobre un 'id' escrito dentro por despiste.
            personas.append(Persona.model_validate({**value, "id": key}))
        except ValidationError as exc:
            errors.extend(_describe(exc, source))
    return personas


def _parse_cases(raw: Any, errors: list[str]) -> list[Case]:
    if not isinstance(raw, list):
        errors.append("consultas.json: la raiz debe ser una lista de consultas")
        return []
    if not raw:
        errors.append("consultas.json: no hay ninguna consulta")
        return []
    cases: list[Case] = []
    for index, value in enumerate(raw):
        if not isinstance(value, dict):
            errors.append(f"consultas.json · [{index}]: debe ser un objeto")
            continue
        source = f"consultas.json · [{index}]"
        if value.get("id"):
            source += f" ({value['id']})"
        try:
            cases.append(Case.model_validate(value))
        except ValidationError as exc:
            errors.extend(_describe(exc, source))
    return cases


def _duplicates(ids: list[str]) -> list[str]:
    seen, dup = set(), []
    for item in ids:
        if item in seen and item not in dup:
            dup.append(item)
        seen.add(item)
    return dup


# -------------------------------------------------------------- validacion --
def parse_suite(personas_json: str, consultas_json: str) -> SuiteSpec:
    errors: list[str] = []
    warnings: list[str] = []

    raw_personas = _load_json(personas_json, "personas.json", errors)
    raw_cases = _load_json(consultas_json, "consultas.json", errors)

    personas = _parse_personas(raw_personas, errors) if raw_personas is not None else []
    cases = _parse_cases(raw_cases, errors) if raw_cases is not None else []

    for dup in _duplicates([c.id for c in cases]):
        errors.append(f"consultas.json: el id de consulta '{dup}' esta repetido")

    known = {p.id for p in personas}
    for case in cases:
        if personas and case.persona not in known:
            errors.append(
                f"consultas.json · {case.id}: la persona '{case.persona}' no existe en personas.json"
            )
        if not case.resultado_esperado.strip():
            warnings.append(
                f"consultas.json · {case.id}: sin 'resultado_esperado'; el evaluador solo tendra el objetivo"
            )
        if not case.valor_esperado.strip():
            warnings.append(
                f"consultas.json · {case.id}: sin 'valor_esperado'; no se comprobara ninguna cifra"
            )

    return SuiteSpec(
        ok=not errors and bool(personas) and bool(cases),
        errors=errors,
        warnings=warnings,
        personas=personas,
        cases=cases,
    )


def build_matrix(
    spec: SuiteSpec,
    case_ids: list[str] | None = None,
    persona_ids: list[str] | None = None,
    repetitions: int = 1,
    max_turns: int = DEFAULT_MAX_TURNS,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[EvalItemSpec]:
    """Expande las consultas x repeticiones en ejecuciones concretas.

    Cada consulta trae su persona, asi que no hay producto cartesiano: filtrar
    por persona deja fuera las consultas que no son suyas.
    """
    if not spec.ok:
        return []
    by_id = {p.id: p for p in spec.personas}
    items: list[EvalItemSpec] = []
    for case in spec.cases:
        if case_ids and case.id not in case_ids:
            continue
        if persona_ids and case.persona not in persona_ids:
            continue
        for rep in range(1, max(repetitions, 1) + 1):
            items.append(EvalItemSpec(case, by_id[case.persona], rep, max_turns, threshold))
    return items


def text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]
