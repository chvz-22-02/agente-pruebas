"""Evaluaciones: ficheros JSON, comprobaciones, agregacion y una bateria completa.

La bateria de extremo a extremo usa tres LLM guionizados (agente, simulador y
evaluador) y una base SQLite temporal, asi que no necesita Ollama, ni un
servidor MCP, ni MLflow:

    python -m pytest tests/test_evals.py -q
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evals import checks  # noqa: E402
from app.evals.simulator import build_messages, clean_output, honors_fin  # noqa: E402
from app.evals.spec import build_matrix, parse_suite  # noqa: E402
from app.llm.base import LLMProvider, LLMResponse, ToolSpec, Usage  # noqa: E402

TEMPLATES = Path(__file__).resolve().parent.parent / "app" / "evals" / "templates"
PERSONAS = (TEMPLATES / "personas.json").read_text(encoding="utf-8")
CASES = (TEMPLATES / "consultas.json").read_text(encoding="utf-8")


# ---------------------------------------------------------------------- JSON -
def test_templates_are_valid() -> None:
    spec = parse_suite(PERSONAS, CASES)
    assert spec.ok, spec.errors
    summary = spec.summary()
    assert [p["id"] for p in summary["personas"]] == [
        "estudiante", "ciudadano", "funcionario_junior", "funcionario_senior",
    ]
    # Cada consulta trae su persona: no hay producto cartesiano.
    assert summary["matrix"] == len(summary["cases"]) == 4


def test_json_errors_are_readable() -> None:
    broken = CASES.replace('"valor_esperado"', '"valor-esperado"', 1)
    spec = parse_suite(PERSONAS, broken)
    assert not spec.ok
    assert any("valor-esperado" in e and "no reconocida" in e for e in spec.errors), spec.errors
    # El error lleva el id de la consulta, no solo el indice.
    assert any("PG-01" in e for e in spec.errors), spec.errors

    syntax = parse_suite(PERSONAS, '[{"id": "x",}]')
    assert not syntax.ok and any("linea" in e for e in syntax.errors), syntax.errors

    shape = parse_suite("[]", CASES)
    assert not shape.ok and any("raiz debe ser un objeto" in e for e in shape.errors), shape.errors


def test_unknown_persona_and_duplicates() -> None:
    spec = parse_suite(PERSONAS, CASES.replace('"persona": "estudiante"', '"persona": "fantasma"', 1))
    assert any("fantasma" in e and "no existe" in e for e in spec.errors), spec.errors

    dup = parse_suite(PERSONAS, CASES.replace('"id": "PG-02"', '"id": "PG-01"'))
    assert any("repetido" in e for e in dup.errors), dup.errors


def test_ambiguity_is_normalized_and_checked() -> None:
    spec = parse_suite(PERSONAS, CASES.replace('"ambiguedad": "alta"', '"ambiguedad": " Alta "', 1))
    assert spec.ok, spec.errors
    assert spec.cases[0].ambiguedad == "alta"

    wrong = parse_suite(PERSONAS, CASES.replace('"ambiguedad": "alta"', '"ambiguedad": "altisima"', 1))
    assert any("ambiguedad" in e and "no permitido" in e for e in wrong.errors), wrong.errors


def test_matrix_filters_and_turn_limits() -> None:
    spec = parse_suite(PERSONAS, CASES)
    items = build_matrix(spec, case_ids=["PG-01"], repetitions=3)
    assert len(items) == 3
    assert all(i.max_turns == 6 and i.threshold == 0.7 for i in items)

    # Filtrar por persona deja fuera las consultas que no son suyas.
    only = build_matrix(spec, persona_ids=["estudiante"], max_turns=2, threshold=0.5)
    assert {i.case.id for i in only} == {"PG-01", "PG-03"}
    assert all(i.max_turns == 2 and i.threshold == 0.5 for i in only)


def test_rubric_adapts_to_the_query() -> None:
    from app.evals.spec import criterios_de

    spec = parse_suite(PERSONAS, CASES)
    rubric = {c.id: c for c in criterios_de(spec.cases[0])}
    assert list(rubric) == ["resultado_esperado", "valor_esperado", "sin_invenciones", "respuesta_util"]
    assert rubric["resultado_esperado"].obligatorio and rubric["sin_invenciones"].obligatorio
    assert "30.0% - 33.8%" in rubric["valor_esperado"].descripcion

    sin_valor = spec.cases[0].model_copy(update={"valor_esperado": ""})
    assert "valor_esperado" not in {c.id for c in criterios_de(sin_valor)}


# ------------------------------------------------------------- comprobaciones -
def test_numbers_spanish_and_english() -> None:
    assert checks.value_mentioned("fue de 24,9 %", 24.9, 0)[0]
    assert checks.value_mentioned("it was 24.9%", 24.9, 0)[0]
    assert checks.value_mentioned("unos 1.234,5 hogares", 1234.5, 0)[0]
    assert checks.value_mentioned("aprox. 25%", 24.9, 0.1)[0] is False
    assert checks.value_mentioned("aprox. 25%", 24.9, 0.2)[0]


def test_expected_numbers_from_free_text() -> None:
    assert checks.expected_numbers("330481.79") == [330481.79]
    assert checks.expected_numbers("863, 933, 4907") == [863, 933, 4907]
    assert checks.expected_numbers("30.0% - 33.8% (intervalo de confianza)") == [30.0, 33.8]
    # Un rango de anios no son dos cifras, una de ellas negativa.
    assert checks.expected_numbers("2020-2024") == [2020, 2024]
    assert checks.expected_numbers("empleo o mercado laboral") == []
    assert checks.expected_numbers("No aplica") == []


def test_deterministic_check_on_expected_value() -> None:
    spec = parse_suite(PERSONAS, CASES)
    serie = spec.cases[0].model_copy(update={"valor_esperado": "863, 933, 4907"})

    ok = checks.deterministic_checks(serie, ["Fueron 863 en 2022, 933 en 2023 y 4.907 en 2024."])
    assert ok[0]["id"] == "valor_mencionado" and ok[0]["cumple"]

    partial = checks.deterministic_checks(serie, ["Fueron 863 y 933."])
    assert not partial[0]["cumple"] and "faltan 4907" in partial[0]["detalle"]

    # Sin cifras que comprobar no se inventa ninguna comprobacion.
    tematico = spec.cases[0].model_copy(update={"valor_esperado": "empleo o mercado laboral"})
    assert checks.deterministic_checks(tematico, ["Hablamos de empleo."]) == []


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
    verdict = {
        "criterios": [
            {"id": "resultado_esperado", "cumple": "si", "justificacion": "ok"},
            {"id": "valor_esperado", "cumple": True, "valor_reportado": "31,2 %"},
        ]
    }
    items = {i["id"]: i for i in checks.judged_items(spec.cases[0], verdict)}
    assert items["resultado_esperado"]["cumple"]
    assert items["valor_esperado"]["valor_reportado"] == "31,2 %"
    assert not items["sin_invenciones"]["cumple"] and items["sin_invenciones"]["sin_juicio"]


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


def test_fin_is_only_honored_for_real_farewells() -> None:
    before = [("user", "hola")]
    after = [("user", "hola"), ("agent", "Fue 24,9 %.")]
    assert not honors_fin("¿Cual fue la pobreza en 2025?", [])  # aun no respondio
    assert not honors_fin("Gracias", before)
    assert not honors_fin("Gracias. ¿Podria darme tambien la fuente?", after)  # repregunta
    assert honors_fin("Perfecto, era justo eso. Gracias.", after)
    assert honors_fin("", after) and honors_fin("", [])  # FIN a secas siempre cierra


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


AGENT_ANSWER = "La pobreza en Ayacucho se estima entre 30,0 % y 33,8 % (fuente: INEI)."


class AgentLLM(_Scripted):
    async def chat(self, messages: list[dict[str, Any]], tools: list[ToolSpec] | None = None,
                   **kwargs: Any) -> LLMResponse:
        return LLMResponse(content=AGENT_ANSWER, usage=Usage(100, 20, 120), latency_ms=5.0)


class SimulatorLLM(_Scripted):
    async def chat(self, messages: list[dict[str, Any]], tools: list[ToolSpec] | None = None,
                   **kwargs: Any) -> LLMResponse:
        # El simulador recibe toda la conversacion en el ultimo mensaje.
        first = "todavia no hay mensajes" in messages[-1]["content"]
        text = "Hola, ¿cuanta pobreza hay en Ayacucho?" if first else "Perfecto, gracias. <<FIN>>"
        return LLMResponse(content=text, usage=Usage(50, 10, 60), latency_ms=3.0)


class JudgeLLM(_Scripted):
    async def chat(self, messages: list[dict[str, Any]], tools: list[ToolSpec] | None = None,
                   **kwargs: Any) -> LLMResponse:
        assert AGENT_ANSWER in messages[-1]["content"], "el evaluador no recibio la transcripcion"
        return LLMResponse(
            content=(
                "Esta es mi evaluacion:\n```json\n"
                '{"criterios": ['
                '{"id": "resultado_esperado", "cumple": true, "justificacion": "Da el intervalo."},'
                ' {"id": "valor_esperado", "cumple": true, "valor_reportado": "30,0 % - 33,8 %"},'
                ' {"id": "sin_invenciones", "cumple": false, "justificacion": "No uso herramientas."},'
                ' {"id": "respuesta_util", "cumple": true, "justificacion": "Clara."}],'
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
                personas_json=PERSONAS,
                consultas_json=CASES,
                agent=AgentModel(provider="s_agent", base_url="http://scripted", model="agente"),
                simulator=RoleModel(provider="s_sim", base_url="http://scripted", model="simulador"),
                judge=RoleModel(provider="s_judge", base_url="http://scripted", model="juez"),
                case_ids=["PG-01"],
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
        # El primer mensaje es el 'consulta_inicial' del fichero, no del simulador.
        assert result["transcript"][0]["source"] == "guion"
        assert result["transcript"][-1]["closing"] is True

        items = {i["id"]: i for i in result["items"]}
        # Comprobacion determinista: 30,0 y 33,8 con coma decimal.
        assert items["valor_mencionado"]["cumple"]
        assert not items["sin_invenciones"]["cumple"]
        # sin_invenciones es obligatorio: suspende aunque la nota pase el umbral.
        assert result["score"] > 0.7 and result["passed"] is False and result["status"] == "failed"

        assert run["summary"]["failed"] == 1 and run["summary"]["tokens"]["judge"] == 480
        # La conversacion quedo en la sesion de la evaluacion, visible en la UI.
        conv = await repo.get_conversation(result["conversation_id"])
        assert conv is not None and conv["session_id"] == job.session_id
        assert conv["metadata"]["case_id"] == "PG-01"
    finally:
        await db.close()


class DyingAgentLLM(_Scripted):
    """Como una conexion MCP que se cae: CancelledError subiendo desde abajo."""

    async def chat(self, messages: list[dict[str, Any]], tools: list[ToolSpec] | None = None,
                   **kwargs: Any) -> LLMResponse:
        raise asyncio.CancelledError


def test_foreign_cancellation_is_an_error_not_a_stop(tmp_path: Path) -> None:
    """Regresion: una bateria se detuvo sola y la UI la dio por 'cancelada'.

    Nadie habia pulsado detener: la conexion MCP se cayo y su CancelledError
    llego hasta arriba. Hacerlo pasar por una cancelacion del usuario esconde
    el fallo justo donde hay que verlo.
    """
    asyncio.run(_foreign_cancel(tmp_path))


async def _foreign_cancel(tmp_path: Path) -> None:
    from app.evals.runner import AgentModel, EvalRequest, RoleModel, eval_manager
    from app.llm import registry
    from app.store import repository as repo
    from app.store.db import db

    registry.PROVIDERS.update({"s_dying": DyingAgentLLM, "s_sim": SimulatorLLM, "s_judge": JudgeLLM})  # type: ignore[dict-item]
    db.path = str(tmp_path / "cancel.sqlite3")
    await db.connect()
    try:
        job = await eval_manager.start(
            EvalRequest(
                personas_json=PERSONAS,
                consultas_json=CASES,
                agent=AgentModel(provider="s_dying", base_url="http://scripted", model="agente"),
                simulator=RoleModel(provider="s_sim", base_url="http://scripted", model="simulador"),
                judge=RoleModel(provider="s_judge", base_url="http://scripted", model="juez"),
            )
        )
        [e async for e in eval_manager.stream(job, 0)]

        run = await repo.get_eval_run(job.id)
        assert run is not None and run["status"] == "error", run["status"]
        assert "Cancelacion inesperada" in run["error"], run["error"]
        assert not job.cancel_requested
        # Lo que no llego a ejecutarse queda como error, no como cancelado.
        assert {r["status"] for r in await repo.list_eval_results(job.id)} == {"error"}
    finally:
        await db.close()


def test_battery_stops_when_the_mcp_is_gone(tmp_path: Path) -> None:
    """Sin MCP los casos restantes se ejecutarian sin herramientas: eso no es un resultado."""
    asyncio.run(_mcp_gone(tmp_path))


async def _mcp_gone(tmp_path: Path) -> None:
    from app.evals.runner import AgentModel, EvalRequest, RoleModel, eval_manager
    from app.llm import registry
    from app.store import repository as repo
    from app.store.db import db

    registry.PROVIDERS.update({"s_agent": AgentLLM, "s_sim": SimulatorLLM, "s_judge": JudgeLLM})  # type: ignore[dict-item]
    db.path = str(tmp_path / "sinmcp.sqlite3")
    await db.connect()
    try:
        job = await eval_manager.start(
            EvalRequest(
                personas_json=PERSONAS,
                consultas_json=CASES,
                agent=AgentModel(provider="s_agent", base_url="http://scripted", model="agente"),
                simulator=RoleModel(provider="s_sim", base_url="http://scripted", model="simulador"),
                judge=RoleModel(provider="s_judge", base_url="http://scripted", model="juez"),
                # Se declaro un MCP que ya no esta vivo.
                mcp_conn_ids=["mcp_fantasma"],
            )
        )
        events = [e async for e in eval_manager.stream(job, 0)]

        run = await repo.get_eval_run(job.id)
        assert run is not None and run["status"] == "error", run["status"]
        assert "no se pudo reconectar" in run["error"], run["error"]
        # Se avisa en el registro en vivo, y no se ejecuta ningun caso.
        assert any(e["type"] == "log" and e.get("level") == "error" for e in events)
        assert not any(e["type"] == "item_start" for e in events)
    finally:
        await db.close()


def test_dead_mcp_is_reconnected_before_each_case(tmp_path: Path) -> None:
    """Un MCP caido entre consultas se levanta de nuevo, con el mismo id, y se sigue."""
    asyncio.run(_mcp_reconnect(tmp_path))


async def _mcp_reconnect(tmp_path: Path) -> None:
    from app.evals import runner as runner_mod
    from app.evals.runner import AgentModel, EvalRequest, RoleModel, eval_manager
    from app.llm import registry
    from app.store import repository as repo
    from app.store.db import db

    class FakeManager:
        """Primera consulta: la conexion esta caida. Tras reconectar, viva."""

        def __init__(self) -> None:
            self.alive = False
            self.reconnected: list[str] = []

        def alive_ids(self, conn_ids: list[str] | None = None) -> list[str]:
            return list(conn_ids or []) if self.alive else []

        async def reconnect(self, conn_id: str) -> dict[str, Any]:
            self.reconnected.append(conn_id)
            self.alive = True
            return {"config": {"url": "https://mcp.ejemplo/mcp"}, "tools": [{"name": "a"}, {"name": "b"}]}

        def get(self, conn_id: str) -> Any:
            raise AssertionError("el agente guionizado no usa herramientas")

    fake = FakeManager()
    registry.PROVIDERS.update({"s_agent": AgentLLM, "s_sim": SimulatorLLM, "s_judge": JudgeLLM})  # type: ignore[dict-item]
    saved, runner_mod.mcp_manager = runner_mod.mcp_manager, fake  # type: ignore[assignment]
    db.path = str(tmp_path / "reconnect.sqlite3")
    await db.connect()
    try:
        job = await eval_manager.start(
            EvalRequest(
                personas_json=PERSONAS,
                consultas_json=CASES,
                agent=AgentModel(provider="s_agent", base_url="http://scripted", model="agente"),
                simulator=RoleModel(provider="s_sim", base_url="http://scripted", model="simulador"),
                judge=RoleModel(provider="s_judge", base_url="http://scripted", model="juez"),
                mcp_conn_ids=["mcp_caido"],
                case_ids=["PG-01", "PG-02"],
            )
        )
        events = [e async for e in eval_manager.stream(job, 0)]
        run = await repo.get_eval_run(job.id)
        assert run is not None and run["status"] == "done", (run["status"], run["error"])
        # Se reconecto una vez (antes de la primera consulta) y se aviso.
        assert fake.reconnected == ["mcp_caido"], fake.reconnected
        warnings = [e["message"] for e in events if e["type"] == "log" and e.get("level") == "warn"]
        assert any("reconectada" in w and "2 herramientas" in w for w in warnings), warnings
        assert sum(1 for e in events if e["type"] == "item_end") == 2
    finally:
        runner_mod.mcp_manager = saved  # type: ignore[assignment]
        await db.close()


class SaturatedThenFineAgentLLM(_Scripted):
    """La primera consulta agota los reintentos del proveedor; la repeticion va bien."""

    calls = 0

    async def chat(self, messages: list[dict[str, Any]], tools: list[ToolSpec] | None = None,
                   **kwargs: Any) -> LLMResponse:
        from app.llm.openai_compat_provider import LLMRequestError

        type(self).calls += 1
        if type(self).calls == 1:
            raise LLMRequestError(504, "el endpoint gratuito de NVIDIA esta saturado", {"model": "x"})
        return LLMResponse(content=AGENT_ANSWER, usage=Usage(100, 20, 120), latency_ms=5.0)


class AlwaysSaturatedAgentLLM(_Scripted):
    async def chat(self, messages: list[dict[str, Any]], tools: list[ToolSpec] | None = None,
                   **kwargs: Any) -> LLMResponse:
        from app.llm.openai_compat_provider import LLMRequestError

        raise LLMRequestError(429, "Too Many Requests", {"model": "x"})


def test_saturated_endpoint_waits_and_retries_the_case(tmp_path: Path) -> None:
    """Regresion: un 504 de NVIDIA truncaba la conversacion y la daba por suspendida."""
    asyncio.run(_saturation_retry(tmp_path))


async def _saturation_retry(tmp_path: Path) -> None:
    from app.evals import runner as runner_mod
    from app.evals.runner import AgentModel, EvalRequest, RoleModel, eval_manager
    from app.llm import registry
    from app.store import repository as repo
    from app.store.db import db

    SaturatedThenFineAgentLLM.calls = 0
    registry.PROVIDERS.update({"s_sat": SaturatedThenFineAgentLLM, "s_sim": SimulatorLLM, "s_judge": JudgeLLM})  # type: ignore[dict-item]
    saved_wait, runner_mod.SATURATION_WAIT_S = runner_mod.SATURATION_WAIT_S, 0.0
    db.path = str(tmp_path / "saturado.sqlite3")
    await db.connect()
    try:
        job = await eval_manager.start(
            EvalRequest(
                personas_json=PERSONAS,
                consultas_json=CASES,
                agent=AgentModel(provider="s_sat", base_url="http://scripted", model="agente"),
                simulator=RoleModel(provider="s_sim", base_url="http://scripted", model="simulador"),
                judge=RoleModel(provider="s_judge", base_url="http://scripted", model="juez"),
                case_ids=["PG-01"],
            )
        )
        events = [e async for e in eval_manager.stream(job, 0)]

        run = await repo.get_eval_run(job.id)
        assert run is not None and run["status"] == "done", (run["status"], run["error"])
        result = (await repo.list_eval_results(job.id))[0]
        # La consulta se repitio y acabo juzgada; el intento saturado no cuenta.
        assert result["status"] == "failed" and result["score"] is not None, result
        assert result["turns"] == 1 and "504" not in (result["error"] or "")
        assert any(e["type"] == "item_phase" and e.get("phase") == "esperando" for e in events)
        warnings = [e["message"] for e in events if e["type"] == "log" and e.get("level") == "warn"]
        assert any("504" in w and "se repite" in w for w in warnings), warnings
        # Un solo item_end: el primer intento no se cierra como resultado.
        assert sum(1 for e in events if e["type"] == "item_end") == 1
        # Y la conversacion abandonada queda en la sesion, distinguible por el titulo.
        titles = sorted(c["title"] for c in await repo.list_conversations(job.session_id))
        assert any("intento 2" in t for t in titles), titles
    finally:
        runner_mod.SATURATION_WAIT_S = saved_wait
        await db.close()


def test_second_saturation_stops_the_battery(tmp_path: Path) -> None:
    asyncio.run(_saturation_stop(tmp_path))


async def _saturation_stop(tmp_path: Path) -> None:
    from app.evals import runner as runner_mod
    from app.evals.runner import AgentModel, EvalRequest, RoleModel, eval_manager
    from app.llm import registry
    from app.store import repository as repo
    from app.store.db import db

    registry.PROVIDERS.update({"s_429": AlwaysSaturatedAgentLLM, "s_sim": SimulatorLLM, "s_judge": JudgeLLM})  # type: ignore[dict-item]
    saved_wait, runner_mod.SATURATION_WAIT_S = runner_mod.SATURATION_WAIT_S, 0.0
    db.path = str(tmp_path / "saturado2.sqlite3")
    await db.connect()
    try:
        job = await eval_manager.start(
            EvalRequest(
                personas_json=PERSONAS,
                consultas_json=CASES,
                agent=AgentModel(provider="s_429", base_url="http://scripted", model="agente"),
                simulator=RoleModel(provider="s_sim", base_url="http://scripted", model="simulador"),
                judge=RoleModel(provider="s_judge", base_url="http://scripted", model="juez"),
            )
        )
        [e async for e in eval_manager.stream(job, 0)]

        run = await repo.get_eval_run(job.id)
        assert run is not None and run["status"] == "error", run["status"]
        assert "429" in run["error"] and "segundo intento" in run["error"], run["error"]
        results = await repo.list_eval_results(job.id)
        # La primera queda en error con el motivo; las demas no se ejecutan.
        assert results[0]["status"] == "error" and "429" in results[0]["error"]
        assert {r["status"] for r in results[1:]} == {"error"}
        assert all(not r["conversation_id"] for r in results[1:])
    finally:
        runner_mod.SATURATION_WAIT_S = saved_wait
        await db.close()


class FreeTextJudgeLLM(_Scripted):
    """Un evaluador con prompt propio no tiene por que devolver JSON."""

    async def chat(self, messages: list[dict[str, Any]], tools: list[ToolSpec] | None = None,
                   **kwargs: Any) -> LLMResponse:
        assert messages[0]["content"] == "Eres un critico literario. Valora la conversacion en un parrafo."
        return LLMResponse(content="Una conversacion correcta pero sin fuentes.", usage=Usage(30, 10, 40), latency_ms=2.0)


def test_custom_judge_prompt_disables_rubric_and_score(tmp_path: Path) -> None:
    asyncio.run(_custom_judge(tmp_path))


async def _custom_judge(tmp_path: Path) -> None:
    from app.evals.runner import AgentModel, EvalRequest, RoleModel, eval_manager
    from app.llm import registry
    from app.store import repository as repo
    from app.store.db import db

    registry.PROVIDERS.update({"s_agent": AgentLLM, "s_sim": SimulatorLLM, "s_free": FreeTextJudgeLLM})  # type: ignore[dict-item]
    db.path = str(tmp_path / "juezpropio.sqlite3")
    await db.connect()
    try:
        job = await eval_manager.start(
            EvalRequest(
                personas_json=PERSONAS,
                consultas_json=CASES,
                agent=AgentModel(provider="s_agent", base_url="http://scripted", model="agente"),
                simulator=RoleModel(provider="s_sim", base_url="http://scripted", model="simulador"),
                judge=RoleModel(
                    provider="s_free", base_url="http://scripted", model="juez",
                    system_prompt="Eres un critico literario. Valora la conversacion en un parrafo.",
                ),
                case_ids=["PG-01"],
            )
        )
        events = [e async for e in eval_manager.stream(job, 0)]
        result = (await repo.list_eval_results(job.id))[0]

        assert result["status"] == "evaluated", result["status"]
        assert result["score"] is None and result["passed"] is None
        assert result["verdict"]["custom_prompt"] is True
        assert result["verdict"]["resumen"] == "Una conversacion correcta pero sin fuentes."
        # Sin rubrica: solo quedan las comprobaciones deterministas.
        assert all(i["origen"] == "determinista" for i in result["items"]), result["items"]
        end = next(e for e in events if e["type"] == "item_end")
        assert end["custom_judge"] is True and end["status"] == "evaluated"
        run = await repo.get_eval_run(job.id)
        assert run is not None and run["summary"]["evaluated"] == 1 and run["summary"]["failed"] == 0
    finally:
        await db.close()


def test_simulator_prompt_template_is_rendered() -> None:
    from app.evals.simulator import DEFAULT_PROMPT, build_system_prompt

    spec = parse_suite(PERSONAS, CASES)
    persona, case = spec.personas[0], spec.cases[0]

    default = build_system_prompt(persona, case)
    assert "{persona_nombre}" not in default and persona.nombre in default and case.goal in default
    assert "<<FIN>>" in default

    custom = build_system_prompt(persona, case, "Eres {persona_nombre} y quieres: {goal}. Cierra con {fin}. {llave suelta}")
    assert custom == f"Eres {persona.nombre} y quieres: {case.goal}. Cierra con <<FIN>>. {{llave suelta}}"
    # Vacio o solo espacios => la plantilla por defecto.
    assert build_system_prompt(persona, case, "   ") == default
    assert DEFAULT_PROMPT.count("{fin}") == 3


class EagerFinSimulatorLLM(_Scripted):
    """Como Nemotron: a veces cierra su primera pregunta con <<FIN>>."""

    async def chat(self, messages: list[dict[str, Any]], tools: list[ToolSpec] | None = None,
                   **kwargs: Any) -> LLMResponse:
        first = "todavia no hay mensajes" in messages[-1]["content"]
        text = "¿Cuanta pobreza hay en Ayacucho? <<FIN>>" if first else "<<FIN>>"
        return LLMResponse(content=text, usage=Usage(10, 5, 15), latency_ms=1.0)


def test_fin_before_any_agent_reply_is_ignored(tmp_path: Path) -> None:
    asyncio.run(_eager_fin(tmp_path))


async def _eager_fin(tmp_path: Path) -> None:
    from app.evals.runner import AgentModel, EvalRequest, RoleModel, eval_manager
    from app.llm import registry
    from app.store import repository as repo
    from app.store.db import db

    registry.PROVIDERS.update({"s_agent": AgentLLM, "s_eager": EagerFinSimulatorLLM, "s_judge": JudgeLLM})  # type: ignore[dict-item]
    db.path = str(tmp_path / "eager.sqlite3")
    await db.connect()
    # Sin 'consulta_inicial' el primer mensaje lo escribe el simulador, que es
    # donde aparece el FIN prematuro.
    sin_guion = json.dumps([{**json.loads(CASES)[0], "consulta_inicial": ""}], ensure_ascii=False)
    try:
        job = await eval_manager.start(
            EvalRequest(
                personas_json=PERSONAS,
                consultas_json=sin_guion,
                agent=AgentModel(provider="s_agent", base_url="http://scripted", model="agente"),
                simulator=RoleModel(provider="s_eager", base_url="http://scripted", model="simulador"),
                judge=RoleModel(provider="s_judge", base_url="http://scripted", model="juez"),
            )
        )
        [e async for e in eval_manager.stream(job, 0)]
        result = (await repo.list_eval_results(job.id))[0]
        # La pregunta llega al agente aunque traiga FIN; el FIN posterior cierra.
        assert result["turns"] == 1, result
        assert [t["role"] for t in result["transcript"]] == ["user", "agent"], result["transcript"]
        assert "<<FIN>>" not in result["transcript"][0]["text"]
    finally:
        await db.close()
