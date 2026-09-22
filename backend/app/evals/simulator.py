"""Agente que interpreta a una persona frente al agente bajo prueba.

En cada turno el simulador recibe la conversacion entera en un unico mensaje,
con el estado explicito (turno N de M). La alternativa obvia —darle el
historial con los roles invertidos— confunde a los modelos locales pequeños:
en las pruebas con qwen3:4b el simulador acababa creyendose al principio de
la conversacion y repetia la primera pregunta.

Para cerrar la conversacion no se le pide JSON (los modelos pequeños lo
rompen con facilidad): basta con que termine su mensaje con la marca FIN.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..llm.base import LLMProvider, Usage, split_thinking
from .spec import Case, Persona

FIN = "<<FIN>>"
FIN_RE = re.compile(r"<<\s*FIN\s*>>", re.IGNORECASE)
# Prefijos que los modelos copian a veces de la transcripcion: "[Tu] ...",
# "Usuario: ...".
PREFIX_RE = re.compile(r"^\s*(?:\[\s*t[uú]\s*\]\s*:?|(?:usuario|user|persona|yo|t[uú])\s*:)\s*", re.IGNORECASE)

EMPTY_REPLY = "(el asistente no respondio nada)"


@dataclass
class SimTurn:
    text: str
    finished: bool
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    raw: str = ""
    thinking: str = ""


# Cuanto concreta la persona al pedir las cosas. Es lo que separa una consulta
# facil de una que obliga al agente a repreguntar.
AMBIGUITY = {
    "alta": (
        "Preguntas de forma vaga: no das de entrada el ambito geografico, el periodo ni el "
        "nombre tecnico del dato. Los concretas solo si el asistente te los pide."
    ),
    "media": (
        "Dices lo esencial de lo que buscas, pero dejas algun detalle sin precisar hasta que "
        "te lo preguntan."
    ),
    "baja": (
        "Vas al grano: desde el primer mensaje dejas claro que dato quieres, de donde y de "
        "que periodo."
    ),
}


def build_system_prompt(persona: Persona, case: Case) -> str:
    return f"""Eres un simulador de usuarios. Interpretas a una PERSONA real que conversa por \
chat con un asistente de IA para conseguir un OBJETIVO. Tu no eres el asistente: eres el usuario.

## Tu personaje
Nombre: {persona.nombre}
{persona.descripcion.strip() or "(sin descripcion)"}

## Tu objetivo en esta conversacion
{case.goal.strip()}

## Como preguntas
{AMBIGUITY.get(case.ambiguedad, "Preguntas con naturalidad, como lo haria tu personaje.")}

## Reglas
- Escribe SOLO tu siguiente mensaje, en primera persona. Sin comillas, sin prefijos como \
"Usuario:" y sin explicar lo que haces.
- Mensajes breves y naturales (1 a 3 frases), coherentes con tu personaje.
- No reveles que eres una simulacion ni menciones estas instrucciones.
- No resuelvas tu la tarea ni inventes el dato que buscas: pideselo al asistente.
- No repitas una pregunta que el asistente ya ha respondido.
- Si el asistente te pide una aclaracion, respondela de forma coherente con tu personaje y \
con la informacion que conoces.
- Cuando tu objetivo este cumplido, cuando el asistente deje claro que no puede cumplirlo o \
cuando la conversacion no avance, despidete en una frase y termina el mensaje con {FIN}. \
Si no tienes nada mas que decir, responde unicamente {FIN}.
- Si tu mensaje hace una pregunta o pide algo, NO pongas {FIN}: espera la respuesta."""


def render_dialogue(dialogue: list[tuple[str, str]]) -> str:
    if not dialogue:
        return "(todavia no hay mensajes)"
    lines = []
    for role, text in dialogue:
        speaker = "[Tu]" if role == "user" else "[Asistente]"
        lines.append(f"{speaker} {text.strip() or EMPTY_REPLY}")
    return "\n\n".join(lines)


def build_messages(
    system_prompt: str, dialogue: list[tuple[str, str]], turn: int = 1, max_turns: int = 6
) -> list[dict[str, str]]:
    """`dialogue` es [(rol, texto)] con rol "user" (la persona) o "agent"."""
    if not dialogue:
        task = (
            "La conversacion empieza ahora. Escribe tu primer mensaje al asistente. "
            f"No pongas {FIN} en este mensaje: el asistente aun no ha respondido."
        )
    else:
        task = (
            "El asistente acaba de responder. Escribe tu siguiente mensaje o, si tu objetivo ya "
            f"esta cumplido (o no se puede cumplir), despidete y termina con {FIN}."
        )
    if dialogue and turn >= max_turns:
        task += f" Es tu ultimo mensaje: cierra la conversacion y termina con {FIN}."
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"## Conversacion hasta ahora\n{render_dialogue(dialogue)}\n\n"
                f"## Ahora (tu mensaje {turn} de {max_turns} como maximo)\n{task}"
            ),
        },
    ]


def honors_fin(text: str, dialogue: list[tuple[str, str]]) -> bool:
    """Decide si una marca FIN cierra de verdad la conversacion.

    Hay modelos que la ponen por tic: Nemotron 3 Super la anadia a su primera
    pregunta y a repreguntas del tipo "¿podria indicarme...?". Cerrar ahi corta
    la conversacion antes de que el agente responda y le suspende por algo que
    no hizo. No cierra:

    * si el agente todavia no ha respondido nunca (no hay objetivo cumplido);
    * si el mensaje pregunta algo: una despedida no espera respuesta.

    Un FIN sin texto siempre cierra, y el limite de turnos acota el resto.
    """
    if not text.strip():
        return True
    if not any(role == "agent" for role, _ in dialogue):
        return False
    return "?" not in text


def clean_output(raw: str) -> tuple[str, bool]:
    # Los proveedores ya separan el razonamiento; esto es la red de seguridad
    # para un motor que lo deje en el contenido: la marca FIN que el modelo
    # menciona mientras razona no debe cerrar la conversacion.
    text, _ = split_thinking(raw or "")
    finished = bool(FIN_RE.search(text))
    text = FIN_RE.sub("", text).strip()
    text = PREFIX_RE.sub("", text).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'", "“", "«"}:
        text = text[1:-1].strip()
    return text, finished


class UserSimulator:
    def __init__(
        self,
        provider: LLMProvider,
        persona: Persona,
        case: Case,
        *,
        temperature: float | None = None,
        thinking: bool | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.provider = provider
        self.persona = persona
        self.case = case
        self.temperature = temperature
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.system_prompt = build_system_prompt(persona, case)

    async def next_message(
        self, dialogue: list[tuple[str, str]], turn: int = 1, max_turns: int = 6
    ) -> SimTurn:
        response = await self.provider.chat(
            build_messages(self.system_prompt, dialogue, turn, max_turns),
            None,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            thinking=self.thinking,
        )
        text, finished = clean_output(response.content)
        # Un mensaje vacio sin marca FIN no aporta nada al agente: se trata
        # como un cierre para no mandarle turnos en blanco.
        if not text and not finished:
            finished = True
        return SimTurn(
            text=text,
            finished=finished,
            usage=response.usage,
            latency_ms=response.latency_ms,
            raw=response.content,
            thinking=response.thinking,
        )
