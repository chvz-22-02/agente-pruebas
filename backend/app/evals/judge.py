"""Evaluador LLM: juzga la conversacion terminada contra la rubrica de la consulta.

Recibe la transcripcion completa, incluidas las llamadas al MCP con sus
resultados (recortados para caber en la ventana de un modelo local), y las
verificaciones deterministas ya calculadas. Devuelve un JSON con un juicio
por criterio; la nota la calcula `checks.aggregate`, no el modelo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import LLMProvider, Usage
from .checks import extract_json
from .spec import Case, Persona, criterios_de

# Presupuesto de caracteres del material que se le pasa al evaluador. Con
# num_ctx=16384 un modelo local tiene ~50k caracteres de ventana; se deja sitio
# para las instrucciones y para la respuesta.
MAX_TOOL_RESULT_CHARS = 2500
MAX_TRANSCRIPT_CHARS = 36000

SYSTEM_PROMPT = """Eres un evaluador imparcial y exigente de agentes conversacionales \
conectados a herramientas (servidores MCP). Recibes una consulta del banco de pruebas, lo que \
se esperaba de ella, una rubrica de criterios, la transcripcion completa de la conversacion \
(con las llamadas a herramientas y lo que devolvieron) y unas verificaciones automaticas ya \
calculadas.

Como evaluas:
- Juzga SOLO con la evidencia de la transcripcion. Si algo no aparece, no lo supongas.
- Un dato es correcto si el agente lo afirma y se corresponde con lo esperado. Si el agente da \
una cifra que no esta respaldada por ninguna herramienta, considerala inventada.
- El valor esperado esta escrito en lenguaje natural: puede ser una cifra, una serie de cifras, \
un rango o un ambito tematico. Si dice que no aplica o que el dato no existe, el agente cumple \
solo si dice con claridad que no lo encontro y no inventa ningun valor.
- Las verificaciones automaticas son hechos; usalas como evidencia.
- Se breve y concreto en las justificaciones.

Responde UNICAMENTE con un objeto JSON valido, sin texto antes ni despues y sin bloques de \
codigo, con exactamente esta forma:
{
  "criterios": [{"id": "<id del criterio>", "cumple": true, "justificacion": "<1-2 frases>"}],
  "resumen": "<2-4 frases con la valoracion global>"
}
Incluye exactamente un elemento por cada criterio de la rubrica, con su mismo id. En el \
criterio "valor_esperado" anade ademas "valor_reportado" con el dato que afirmo el agente \
(o null si no dio ninguno)."""

REPAIR_PROMPT = """Tu respuesta anterior no era un JSON valido. Devuelve ahora SOLO el objeto \
JSON con la forma pedida, sin texto adicional."""


@dataclass
class JudgeOutcome:
    verdict: dict[str, Any] | None
    raw: str = ""
    thinking: str = ""
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    attempts: int = 0
    error: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)


def _clip(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f" ... [recortado, {len(text) - limit} caracteres mas]"


def render_transcript(transcript: list[dict[str, Any]]) -> str:
    """Transcripcion legible para el evaluador, con las llamadas al MCP."""
    lines: list[str] = []
    for entry in transcript:
        if entry["role"] == "user":
            lines.append(f"### Turno {entry['turn']} · USUARIO\n{entry['text']}")
            continue
        block = [f"### Turno {entry['turn']} · AGENTE"]
        for call in entry.get("tool_calls", []):
            status = "OK" if call.get("ok") else f"ERROR: {call.get('error') or ''}"
            block.append(
                f"[herramienta] {call.get('tool')}("
                f"{json.dumps(call.get('arguments', {}), ensure_ascii=False)}) -> {status}\n"
                f"[resultado] {_clip(call.get('result') or '', MAX_TOOL_RESULT_CHARS)}"
            )
        if entry.get("error"):
            block.append(f"[error del agente] {entry['error']}")
        block.append(f"[respuesta] {entry.get('text') or '(sin respuesta)'}")
        lines.append("\n".join(block))
    text = "\n\n".join(lines)
    if len(text) <= MAX_TRANSCRIPT_CHARS:
        return text
    # Se conservan principio y final: el planteamiento y la respuesta que cuenta.
    head = MAX_TRANSCRIPT_CHARS // 3
    tail = MAX_TRANSCRIPT_CHARS - head
    return text[:head] + "\n\n[... transcripcion recortada ...]\n\n" + text[-tail:]


def build_user_prompt(
    case: Case, persona: Persona, transcript: list[dict[str, Any]], checks: list[dict[str, Any]]
) -> str:
    rubric = "\n".join(f"- {c.id}: {c.descripcion.strip()}" for c in criterios_de(case))
    auto = "\n".join(
        f"- [{'CUMPLE' if c['cumple'] else 'NO CUMPLE'}] {c['descripcion']} ({c['detalle']})"
        for c in checks
    ) or "(ninguna)"

    return f"""## Consulta: {case.id}
Objetivo del usuario: {case.goal.strip()}
Usuario simulado: {persona.nombre}

## Lo que se esperaba
Resultado: {case.resultado_esperado.strip() or "(no se indica)"}
Valor: {case.valor_esperado.strip() or "(no se indica)"}

## Rubrica (criterios a juzgar)
{rubric}

## Verificaciones automaticas
{auto}

## Transcripcion
{render_transcript(transcript)}

Devuelve ahora el JSON de evaluacion."""


class Judge:
    """Con el prompt por defecto exige el JSON de la rubrica; con uno propio, no.

    Un prompt escrito desde la UI puede pedir cualquier cosa (un informe, una
    nota del 1 al 10, un parrafo), asi que no se le impone el contrato JSON ni
    se le reprocha no cumplirlo: se guarda su respuesta tal cual y la nota
    por rubrica queda desactivada.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        temperature: float | None = 0.0,
        thinking: bool | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.provider = provider
        self.temperature = temperature
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.custom = bool(system_prompt and system_prompt.strip())
        self.system_prompt = system_prompt.strip() if self.custom else SYSTEM_PROMPT  # type: ignore[union-attr]

    async def evaluate(
        self,
        case: Case,
        persona: Persona,
        transcript: list[dict[str, Any]],
        checks: list[dict[str, Any]],
    ) -> JudgeOutcome:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": build_user_prompt(case, persona, transcript, checks)},
        ]
        outcome = JudgeOutcome(verdict=None, messages=list(messages))

        # Un segundo intento con una peticion de reparacion: los modelos
        # locales a veces anaden prosa alrededor del JSON o lo dejan a medias.
        # Con un prompt propio no hay contrato que reparar: un solo intento.
        for attempt in (1,) if self.custom else (1, 2):
            outcome.attempts = attempt
            response = await self.provider.chat(
                messages,
                None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                thinking=self.thinking,
            )
            outcome.usage = outcome.usage.merge(response.usage)
            outcome.latency_ms += response.latency_ms
            outcome.raw = response.content
            outcome.thinking = response.thinking
            verdict = extract_json(response.content)
            if verdict is not None or self.custom:
                outcome.verdict = verdict
                return outcome
            messages = [
                *messages,
                {"role": "assistant", "content": response.content or "(vacio)"},
                {"role": "user", "content": REPAIR_PROMPT},
            ]

        outcome.error = "El evaluador no devolvio un JSON valido tras 2 intentos"
        return outcome
