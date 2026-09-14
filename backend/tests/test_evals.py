"""Evaluaciones: YAML, comprobaciones, agregacion y una bateria completa.

La bateria de extremo a extremo usa tres LLM guionizados (agente, simulador y
evaluador) y una base SQLite temporal, asi que no necesita Ollama, ni un
servidor MCP, ni MLflow:

    python -m pytest tests/test_evals.py -q
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evals import checks  # noqa: E402
from app.evals.simulator import build_messages, clean_output  # noqa: E402
from app.evals.spec import build_matrix, parse_suite  # noqa: E402
from app.llm.base import LLMProvider, LLMResponse, ToolSpec, Usage  # noqa: E402

TEMPLATES = Path(__file__).resolve().parent.parent / "app" / "evals" / "templates"
PERSONAS = (TEMPLATES / "personas.yaml").read_text(encoding="utf-8")
CASES = (TEMPLATES / "casos.yaml").read_text(encoding="utf-8")


# ---------------------------------------------------------------------- YAML -
def test_templates_are_valid() -> None:
    spec = parse_suite(PERSONAS, CASES)
    assert spec.ok, spec.errors
    summary = spec.summary()
    assert [p["id"] for p in summary["personas"]] == ["analista_experta", "ciudadano_novato"]
    assert summary["matrix"] == 4  # 2 casos x 2 personas


def test_yaml_errors_are_readable() -> None:
    broken = CASES.replace("criterios:", "criterio:", 1)
    spec = parse_suite(PERSONAS, broken)
    assert not spec.ok
    assert any("criterio" in e and "no reconocida" in e for e in spec.errors), spec.errors
    # El error lleva el id del caso, no solo el indice.
    assert any("pobreza_lima_2025" in e for e in spec.errors), spec.errors

    syntax = parse_suite("personas: [\n  - id: x", CASES)
    assert not syntax.ok and any("linea" in e for e in syntax.errors), syntax.errors


def test_unknown_persona_and_duplicates() -> None:
    cases = CASES.replace("[analista_experta, ciudadano_novato]", "[analista_experta, fantasma]")
    spec = parse_suite(PERSONAS, cases)
    assert any("fantasma" in e for e in spec.errors), spec.errors

    dup = PERSONAS.replace("id: ciudadano_novato", "id: analista_experta")
    spec = parse_suite(dup, CASES)
    assert any("repetido" in e for e in spec.errors), spec.errors


def test_decimal_comma_in_expected_value() -> None:
    spec = parse_suite(PERSONAS, CASES.replace("valor: 24.9", 'valor: "24,9"'))
    assert spec.ok, spec.errors
    assert spec.cases is not None
    assert spec.cases.casos[0].resultado_esperado.valor == 24.9


def test_matrix_filters_and_turn_limits() -> None:
    spec = parse_suite(PERSONAS, CASES)
    items = build_matrix(spec, case_ids=["pobreza_lima_2025"], repetitions=3)
    assert len(items) == 6
    # La persona puede fijar su propio limite (ciudadano_novato: 8).
    by_persona = {i.persona.id: i.max_turns for i in items}
    assert by_persona == {"analista_experta": 6, "ciudadano_novato": 8}
    only = build_matrix(spec, persona_ids=["ciudadano_novato"], max_turns_override=2)
    assert {i.persona.id for i in only} == {"ciudadano_novato"}
    assert all(i.max_turns == 2 for i in only)


# ------------------------------------------------------------- comprobaciones -
def test_numbers_spanish_and_english() -> None:
    assert checks.value_mentioned("fue de 24,9 %", 24.9, 0)[0]
    assert checks.value_mentioned("it was 24.9%", 24.9, 0)[0]
    assert checks.value_mentioned("unos 1.234,5 hogares", 1234.5, 0)[0]
    assert checks.value_mentioned("aprox. 25%", 24.9, 0.1)[0] is False
    assert checks.value_mentioned("aprox. 25%", 24.9, 0.2)[0]


def test_deterministic_checks() -> None:
    spec = parse_suite(PERSONAS, CASES)
    assert spec.cases is not None
    case = spec.cases.casos[0].model_copy(deep=True)
    case.herramientas.debe_usar = ["step4_get_data"]
    case.herramientas.no_debe_usar = ["borrar_todo"]
    case.herramientas.sin_errores = True
    events = [
        {"tool_name": "sirtod__step4_get_data", "real_tool_name": "step4_get_data", "ok": True},
        {"tool_name": "step2_get_indicators", "real_tool_name": "step2_get_indicators", "ok": False,
         "error": "timeout"},
    ]
    result = {c["id"]: c for c in checks.deterministic_checks(case, ["Fue 24,9%"], events)}
    assert result["usa:step4_get_data"]["cumple"]
    assert result["no_usa:borrar_todo"]["cumple"]
    assert not result["sin_errores_mcp"]["cumple"]
    assert result["max_llamadas"]["cumple"]
    assert result["valor_mencionado"]["cumple"]


def test_aggregate_respects_weights_and_mandatory() -> None:
    items = [
        {"id": "a", "cumple": True, "peso": 3, "obligatorio": False},
        {"id": "b", "cumple": False, "peso": 1, "obligatorio": False},
    ]
    agg = checks.aggregate(items, 0.7)
    assert agg["score"] == 0.75 and agg["passed"]

    items[1]["obligatorio"] = True
    agg = checks.aggregate(items, 0.7)
    assert not agg["passed"] and agg["failed_mandatory"] == ["b"]


def test_judged_items_missing_criterion_counts_as_failed() -> None:
    spec = parse_suite(PERSONAS, CASES)
    assert spec.cases is not None
    case = spec.cases.casos[0]
    verdict = {
        "resultado_esperado": {"cumple": True, "justificacion": "ok"},
        "criterios": [{"id": "ambito_correcto", "cumple": "si"}],
    }
    items = {i["id"]: i for i in checks.judged_items(case, verdict)}
    assert items["resultado_esperado"]["cumple"]
    assert items["ambito_correcto"]["cumple"]
    assert not items["cita_fuente"]["cumple"] and items["cita_fuente"]["sin_juicio"]


def test_extract_json_variants() -> None:
    assert checks.extract_json('{"a": 1}') == {"a": 1}
    assert checks.extract_json('Aqui va:\n```json\n{"a": {"b": "}"}}\n```\nfin') == {"a": {"b": "}"}}
    assert checks.extract_json('<think>{"no": 1}</think> Resultado {"ok": true} listo') == {"ok": True}
    assert checks.extract_json("sin json") is None


def test_simulator_output_cleanup() -> None:
    assert clean_output('Usuario: "Hola, ¿me ayudas?"') == ("Hola, ¿me ayudas?", False)
    assert clean_output("[Tú] Vale, gracias") == ("Vale, gracias", False)
    assert clean_output("Gracias, era eso. <<FIN>>") == ("Gracias, era eso.", True)
    assert clean_output("<< fin >>") == ("", True)
    # Razonamiento que se cuela en el contenido (solo con el cierre </think>):
    # la marca FIN que el modelo menciona al razonar no cierra la conversacion.
    leaked = "Debo terminar con <<FIN>> cuando acabe...\n</think>\n\n¿Y cuanto cuestan 3?"
    assert clean_output(leaked) == ("¿Y cuanto cuestan 3?", False)


def test_simulator_sees_whole_dialogue_in_one_message() -> None:
    first = build_messages("SYS", [], 1, 4)
    assert [m["role"] for m in first] == ["system", "user"]
    assert "primer mensaje" in first[1]["content"]

    later = build_messages("SYS", [("user", "hola"), ("agent", "")], 4, 4)
    assert [m["role"] for m in later] == ["system", "user"]
    body = later[1]["content"]
    assert "[Tu] hola" in body and "[Asistente] (el asistente no respondio nada)" in body
    assert "ultimo mensaje" in body  # turno 4 de 4: debe cerrar


def test_orphan_think_close_tag_is_split() -> None:
    from app.llm.base import split_thinking

    assert split_thinking("razono...\n</think>\n\nRespuesta") == ("Respuesta", "razono...")
    assert split_thinking("<think>a</think>b") == ("b", "a")
    assert checks.extract_json('pienso {"x": 1}\n</think>\n{"ok": true}') == {"ok": True}


# ---------------------------------------------------------- extremo a extremo -
class _Scripted(LLMProvider):
    name = "scripted"

    def __init__(self, base_url: str = "http://scripted", model: str = "guion", **options: Any) -> None:
        super().__init__(base_url, model, **options)

    async def list_models(self) -> list[str]:
        return [self.model]

    async def health(self) -> dict[str, Any]:
        return {"ok": True}


class AgentLLM(_Scripted):
    async def chat(self, messages: list[dict[str, Any]], tools: list[ToolSpec] | None = None,
                   **kwargs: Any) -> LLMResponse:
        return LLMResponse(
            content="La pobreza monetaria de Lima Metropolitana en 2025 fue de 24,9 % (fuente: INEI).",
            usage=Usage(100, 20, 120),
            latency_ms=5.0,
        )


class SimulatorLLM(_Scripted):
    async def chat(self, messages: list[dict[str, Any]], tools: list[ToolSpec] | None = None,
                   **kwargs: Any) -> LLMResponse:
        # El simulador recibe toda la conversacion en el ultimo mensaje.
        first = "todavia no hay mensajes" in messages[-1]["content"]
        text = "Hola, ¿cual fue la pobreza en Lima Metropolitana en 2025?" if first else "Perfecto, gracias. <<FIN>>"
        return LLMResponse(content=text, usage=Usage(50, 10, 60), latency_ms=3.0)


class JudgeLLM(_Scripted):
    async def chat(self, messages: list[dict[str, Any]], tools: list[ToolSpec] | None = None,
                   **kwargs: Any) -> LLMResponse:
        assert "24,9 %" in messages[-1]["content"], "el evaluador no recibio la transcripcion"
        return LLMResponse(
            content=(
                "Esta es mi evaluacion:\n```json\n"
                '{"resultado_esperado": {"cumple": true, "valor_reportado": "24,9 %", "justificacion": "Coincide."},'
                ' "criterios": [{"id": "ambito_correcto", "cumple": true, "justificacion": "Si."},'
                ' {"id": "cita_fuente", "cumple": true, "justificacion": "Cita al INEI."},'
                ' {"id": "sin_invenciones", "cumple": false, "justificacion": "No uso herramientas."}],'
                ' "resumen": "Dato correcto pero sin respaldo de herramientas."}\n```'
            ),
            usage=Usage(400, 80, 480),
            latency_ms=7.0,
        )


def test_full_battery_with_scripted_models(tmp_path: Path) -> None:
    asyncio.run(_battery(tmp_path))


async def _battery(tmp_path: Path) -> None:
    from app.evals.runner import AgentModel, EvalRequest, RoleModel, eval_manager
    from app.llm import registry
    from app.store import repository as repo
    from app.store.db import db

    registry.PROVIDERS.update({"s_agent": AgentLLM, "s_sim": SimulatorLLM, "s_judge": JudgeLLM})  # type: ignore[dict-item]
    db.path = str(tmp_path / "evals.sqlite3")
    await db.connect()
    try:
        job = await eval_manager.start(
            EvalRequest(
                personas_yaml=PERSONAS,
                cases_yaml=CASES,
                agent=AgentModel(provider="s_agent", base_url="http://scripted", model="agente"),
                simulator=RoleModel(provider="s_sim", base_url="http://scripted", model="simulador"),
                judge=RoleModel(provider="s_judge", base_url="http://scripted", model="juez"),
                case_ids=["pobreza_lima_2025"],
                persona_ids=["analista_experta"],
            )
        )
        events = [e async for e in eval_manager.stream(job, 0)]
        kinds = [e["type"] for e in events]
        assert kinds[0] == "run_start" and kinds[-1] == "run_end", kinds
        assert "turn_user" in kinds and "turn_agent" in kinds and "item_end" in kinds

        run = await repo.get_eval_run(job.id)
        assert run is not None and run["status"] == "done", run
        results = await repo.list_eval_results(job.id)
        assert len(results) == 1
        result = results[0]

        # 1 turno con el agente; la despedida con FIN no se le manda.
        assert result["turns"] == 1 and result["end_reason"] == "persona_termina", result
        assert [t["role"] for t in result["transcript"]] == ["user", "agent", "user"]
        assert result["transcript"][-1]["closing"] is True

        items = {i["id"]: i for i in result["items"]}
        assert items["valor_mencionado"]["cumple"]  # comprobacion determinista (24,9 con coma)
        assert not items["sin_invenciones"]["cumple"]
        # sin_invenciones es obligatorio: suspende aunque la nota pase el umbral.
        assert result["score"] > 0.7 and result["passed"] is False and result["status"] == "failed"

        assert run["summary"]["failed"] == 1 and run["summary"]["tokens"]["judge"] == 480
        # La conversacion quedo en la sesion de la evaluacion, visible en la UI.
        conv = await repo.get_conversation(result["conversation_id"])
        assert conv is not None and conv["session_id"] == job.session_id
        assert conv["metadata"]["case_id"] == "pobreza_lima_2025"
    finally:
        await db.close()
