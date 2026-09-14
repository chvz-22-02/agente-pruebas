"""Estructura de los ficheros YAML de evaluacion y su validacion.

Hay dos ficheros independientes, para poder combinar el mismo juego de
personas con distintas baterias de casos (y al reves):

* ``personas.yaml``: los usuarios que el simulador debe interpretar.
* ``casos.yaml``:    las casuisticas a evaluar, con su resultado esperado, la
  rubrica para el evaluador LLM y las comprobaciones deterministas.

Los modelos rechazan claves desconocidas (``extra="forbid"``) a proposito: en
un YAML escrito a mano, ``criterio:`` en lugar de ``criterios:`` no debe pasar
en silencio y dejar el caso sin rubrica.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

ID_PATTERN = r"^[A-Za-z0-9_.-]+$"
MAX_TURNS_LIMIT = 30


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _to_number(value: Any) -> Any:
    """Acepta ``24,9`` ademas de ``24.9``: el YAML lo escriben personas."""
    if isinstance(value, str):
        text = value.strip().replace(" ", "")
        if re.fullmatch(r"-?\d+(,\d+)?", text):
            return float(text.replace(",", "."))
        if re.fullmatch(r"-?\d+(\.\d+)?", text):
            return float(text)
    return value


# ------------------------------------------------------------------ personas -
class PersonaStyle(_Strict):
    tono: str = "neutral"
    conocimiento: str = "intermedio"
    paciencia: str = "media"
    idioma: str = "es"


class Persona(_Strict):
    id: str = Field(pattern=ID_PATTERN)
    nombre: str
    descripcion: str = ""
    estilo: PersonaStyle = Field(default_factory=PersonaStyle)
    # Instrucciones libres para el simulador, una por linea.
    comportamiento: list[str] = Field(default_factory=list)
    # Sobrescribe el limite de turnos del caso para esta persona.
    max_turnos: int | None = Field(default=None, ge=1, le=MAX_TURNS_LIMIT)


class PersonasFile(_Strict):
    version: int = 1
    personas: list[Persona] = Field(min_length=1)


# --------------------------------------------------------------------- casos -
ExpectedType = Literal["valor", "no_existe", "error_controlado", "texto", "libre"]


class ExpectedResult(_Strict):
    """Lo que el agente deberia acabar respondiendo.

    * ``valor``:            una cifra concreta (``valor`` + ``tolerancia``).
    * ``no_existe``:        el dato no existe para lo pedido; no debe inventarlo.
    * ``error_controlado``: la peticion es invalida y debe explicarlo.
    * ``texto``:            una respuesta textual concreta (``valor`` como texto).
    * ``libre``:            solo la ``descripcion``; el evaluador juzga.
    """

    tipo: ExpectedType = "libre"
    descripcion: str
    valor: float | str | None = None
    unidad: str = ""
    # Diferencia absoluta admitida al buscar la cifra en la respuesta.
    tolerancia: float = Field(default=0.0, ge=0)
    # Comprobaciones literales (sin distinguir mayusculas) sobre las respuestas.
    debe_contener: list[str] = Field(default_factory=list)
    no_debe_contener: list[str] = Field(default_factory=list)
    peso: float = Field(default=3.0, ge=0)
    # Si no se cumple, el caso falla aunque la puntuacion supere el umbral.
    obligatorio: bool = True

    @field_validator("valor", mode="before")
    @classmethod
    def coerce_valor(cls, value: Any) -> Any:
        return _to_number(value)


class Criterion(_Strict):
    id: str = Field(pattern=ID_PATTERN)
    descripcion: str
    peso: float = Field(default=1.0, ge=0)
    obligatorio: bool = False


class ToolExpectations(_Strict):
    debe_usar: list[str] = Field(default_factory=list)
    no_debe_usar: list[str] = Field(default_factory=list)
    max_llamadas: int | None = Field(default=None, ge=0)
    # Falla si alguna llamada al MCP devolvio error.
    sin_errores: bool = False


class Case(_Strict):
    id: str = Field(pattern=ID_PATTERN)
    titulo: str = ""
    # Lo que la persona quiere conseguir. Se le da al simulador, no al agente.
    objetivo: str
    # Primer mensaje literal. Vacio => lo redacta el simulador.
    mensaje_inicial: str = ""
    # Datos que la persona conoce y usa solo si el agente se los pide.
    contexto_persona: str = ""
    # Personas que ejecutan el caso. Vacio => las de `defaults.personas`.
    personas: list[str] = Field(default_factory=list)
    max_turnos: int | None = Field(default=None, ge=1, le=MAX_TURNS_LIMIT)
    resultado_esperado: ExpectedResult
    criterios: list[Criterion] = Field(default_factory=list)
    herramientas: ToolExpectations = Field(default_factory=ToolExpectations)
    umbral: float | None = Field(default=None, ge=0, le=1)
    etiquetas: list[str] = Field(default_factory=list)

    @property
    def label(self) -> str:
        return self.titulo or self.id


class SuiteInfo(_Strict):
    nombre: str = "suite sin nombre"
    descripcion: str = ""


class CaseDefaults(_Strict):
    max_turnos: int = Field(default=6, ge=1, le=MAX_TURNS_LIMIT)
    # Vacio => todas las personas del fichero de personas.
    personas: list[str] = Field(default_factory=list)
    umbral_aprobacion: float = Field(default=0.7, ge=0, le=1)


class CasesFile(_Strict):
    version: int = 1
    suite: SuiteInfo = Field(default_factory=SuiteInfo)
    defaults: CaseDefaults = Field(default_factory=CaseDefaults)
    casos: list[Case] = Field(min_length=1)


# --------------------------------------------------------------- resultado ---
@dataclass
class SuiteSpec:
    """Resultado de validar la pareja de ficheros."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    personas: PersonasFile | None = None
    cases: CasesFile | None = None

    def summary(self) -> dict[str, Any]:
        personas = self.personas.personas if self.personas else []
        cases = self.cases.casos if self.cases else []
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "suite": self.cases.suite.model_dump() if self.cases else None,
            "defaults": self.cases.defaults.model_dump() if self.cases else None,
            "personas": [
                {"id": p.id, "nombre": p.nombre, "descripcion": p.descripcion} for p in personas
            ],
            "cases": [
                {
                    "id": c.id,
                    "titulo": c.label,
                    "objetivo": c.objetivo,
                    "tipo": c.resultado_esperado.tipo,
                    "personas": personas_for_case(self, c),
                    "criterios": len(c.criterios),
                    "etiquetas": c.etiquetas,
                }
                for c in cases
            ],
            "matrix": len(build_matrix(self)) if self.ok else 0,
        }


@dataclass
class EvalItemSpec:
    """Una ejecucion concreta: un caso, con una persona, en una repeticion."""

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
    "too_short": "no puede estar vacio",
    "string_type": "debe ser texto",
    "list_type": "debe ser una lista",
    "int_parsing": "debe ser un numero entero",
    "float_parsing": "debe ser un numero",
    "bool_parsing": "debe ser true o false",
    "greater_than_equal": "valor demasiado pequeño",
    "less_than_equal": "valor demasiado grande",
}


def _loc(loc: tuple[Any, ...], raw: Any) -> str:
    """``('casos', 0, 'tipo')`` -> ``casos[0] (id_del_caso).tipo``."""
    parts: list[str] = []
    node = raw
    for item in loc:
        if isinstance(item, int):
            label = f"[{item}]"
            try:
                node = node[item]
                if isinstance(node, dict) and node.get("id"):
                    label += f" ({node['id']})"
            except (IndexError, KeyError, TypeError):
                node = None
            parts.append(label)
        else:
            parts.append(("." if parts else "") + str(item))
            node = node.get(item) if isinstance(node, dict) else None
    return "".join(parts) or "(raiz)"


def _format_validation(exc: ValidationError, raw: Any, source: str) -> list[str]:
    out = []
    for err in exc.errors():
        message = _MESSAGES.get(err["type"])
        if err["type"] == "literal_error":
            expected = err.get("ctx", {}).get("expected", "")
            message = f"valor no permitido; usa uno de: {expected}"
        out.append(f"{source} · {_loc(err['loc'], raw)}: {message or err['msg']}")
    return out


def _load_yaml(text: str, source: str, errors: list[str]) -> Any:
    if not text or not text.strip():
        errors.append(f"{source}: el fichero esta vacio")
        return None
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" (linea {mark.line + 1}, columna {mark.column + 1})" if mark else ""
        problem = getattr(exc, "problem", None) or str(exc)
        errors.append(f"{source}: YAML invalido{where}: {problem}")
        return None


def _duplicates(ids: list[str]) -> list[str]:
    seen, dup = set(), []
    for item in ids:
        if item in seen and item not in dup:
            dup.append(item)
        seen.add(item)
    return dup


# -------------------------------------------------------------- validacion --
def parse_suite(personas_yaml: str, cases_yaml: str) -> SuiteSpec:
    errors: list[str] = []
    warnings: list[str] = []

    raw_personas = _load_yaml(personas_yaml, "personas.yaml", errors)
    raw_cases = _load_yaml(cases_yaml, "casos.yaml", errors)

    personas: PersonasFile | None = None
    cases: CasesFile | None = None

    if raw_personas is not None:
        if not isinstance(raw_personas, dict):
            errors.append("personas.yaml: la raiz debe ser un mapa con la clave 'personas'")
        else:
            try:
                personas = PersonasFile.model_validate(raw_personas)
            except ValidationError as exc:
                errors.extend(_format_validation(exc, raw_personas, "personas.yaml"))

    if raw_cases is not None:
        if not isinstance(raw_cases, dict):
            errors.append("casos.yaml: la raiz debe ser un mapa con la clave 'casos'")
        else:
            try:
                cases = CasesFile.model_validate(raw_cases)
            except ValidationError as exc:
                errors.extend(_format_validation(exc, raw_cases, "casos.yaml"))

    if personas:
        for dup in _duplicates([p.id for p in personas.personas]):
            errors.append(f"personas.yaml: el id de persona '{dup}' esta repetido")

    if cases:
        for dup in _duplicates([c.id for c in cases.casos]):
            errors.append(f"casos.yaml: el id de caso '{dup}' esta repetido")
        for case in cases.casos:
            for dup in _duplicates([c.id for c in case.criterios]):
                errors.append(f"casos.yaml · {case.id}: el criterio '{dup}' esta repetido")
            if "resultado_esperado" in {c.id for c in case.criterios}:
                errors.append(
                    f"casos.yaml · {case.id}: 'resultado_esperado' es un id reservado para criterios"
                )
            expected = case.resultado_esperado
            if expected.tipo in {"valor", "texto"} and expected.valor is None:
                warnings.append(
                    f"casos.yaml · {case.id}: tipo '{expected.tipo}' sin 'valor'; "
                    "solo se juzgara con la descripcion"
                )
            if expected.tolerancia and not isinstance(expected.valor, float):
                warnings.append(
                    f"casos.yaml · {case.id}: 'tolerancia' solo aplica a valores numericos"
                )
            if not case.criterios:
                warnings.append(
                    f"casos.yaml · {case.id}: sin 'criterios'; se evaluara solo el resultado esperado"
                )

    if personas and cases:
        known = {p.id for p in personas.personas}
        for pid in cases.defaults.personas:
            if pid not in known:
                errors.append(f"casos.yaml · defaults.personas: la persona '{pid}' no existe")
        for case in cases.casos:
            for pid in case.personas:
                if pid not in known:
                    errors.append(f"casos.yaml · {case.id}: la persona '{pid}' no existe")

    return SuiteSpec(
        ok=not errors and personas is not None and cases is not None,
        errors=errors,
        warnings=warnings,
        personas=personas,
        cases=cases,
    )


def personas_for_case(spec: SuiteSpec, case: Case) -> list[str]:
    if spec.personas is None or spec.cases is None:
        return []
    chosen = case.personas or spec.cases.defaults.personas
    return list(chosen) if chosen else [p.id for p in spec.personas.personas]


def build_matrix(
    spec: SuiteSpec,
    case_ids: list[str] | None = None,
    persona_ids: list[str] | None = None,
    repetitions: int = 1,
    max_turns_override: int | None = None,
) -> list[EvalItemSpec]:
    """Expande casos x personas x repeticiones en ejecuciones concretas.

    Los filtros de la UI restringen, nunca amplian: una persona que el caso no
    declara no se le aplica aunque este marcada.
    """
    if not spec.ok or spec.personas is None or spec.cases is None:
        return []
    by_id = {p.id: p for p in spec.personas.personas}
    items: list[EvalItemSpec] = []
    for case in spec.cases.casos:
        if case_ids and case.id not in case_ids:
            continue
        for pid in personas_for_case(spec, case):
            if persona_ids and pid not in persona_ids:
                continue
            persona = by_id[pid]
            max_turns = (
                max_turns_override
                or persona.max_turnos
                or case.max_turnos
                or spec.cases.defaults.max_turnos
            )
            threshold = (
                case.umbral if case.umbral is not None else spec.cases.defaults.umbral_aprobacion
            )
            for rep in range(1, max(repetitions, 1) + 1):
                items.append(EvalItemSpec(case, persona, rep, max_turns, threshold))
    return items


def text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]
