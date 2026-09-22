"""Orquestacion de una bateria de evaluacion.

Una ejecucion (`EvalJob`) corre en segundo plano, independiente de la
peticion HTTP que la lanza: en CPU un caso puede tardar minutos, y recargar la
pagina no debe matarla. La UI se engancha a su flujo de eventos (SSE) y puede
reengancharse cuando quiera; lo definitivo queda en SQLite y en MLflow.

Mapeo sobre la jerarquia existente:

    sesion (SQLite)        <-> run padre en MLflow   (kind=evaluation)
      conversacion          <-> run hijo              (caso x persona x repeticion)
        interaccion         <-> traza del agente      (una por turno del usuario simulado)
      + traza del evaluador (eval_role=judge) y assessments sobre la ultima traza
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from typing import Any

from ..agent.loop import AgentRunner, RunConfig
from ..config import settings
from ..llm.base import Usage
from ..llm.registry import get_provider
from ..mcpclient.manager import mcp_manager
from ..observability.mlflow_tracker import (
    ATTR_CHAT_USAGE,
    ATTR_LLM_MODEL,
    ATTR_LLM_PROVIDER,
    tracker,
)
from ..store import repository as repo
from ..store.db import new_id
from .checks import aggregate, deterministic_checks, judged_items
from .judge import Judge, JudgeOutcome, render_transcript
from .simulator import UserSimulator, honors_fin
from .spec import EvalItemSpec, SuiteSpec, build_matrix, parse_suite, text_hash

logger = logging.getLogger(__name__)

# Lo que viaja a la UI de cada resultado de herramienta. El integro esta en
# SQLite (tool_events) y en MLflow, como en el chat.
UI_PREVIEW_CHARS = 600
# Lo que se guarda del resultado de cada herramienta dentro de la transcripcion
# de la evaluacion (el integro, de nuevo, esta en tool_events).
TRANSCRIPT_RESULT_CHARS = 4000
# Espera maxima del flujo SSE entre dos comprobaciones de desconexion.
WAIT_EVENTS_S = 15.0
# Ejecuciones terminadas cuyos eventos se conservan en memoria.
KEEP_FINISHED_JOBS = 10


# ------------------------------------------------------------ configuracion -
@dataclass
class RoleModel:
    """Modelo de uno de los tres papeles: agente, simulador o evaluador."""

    provider: str | None = None
    base_url: str | None = None
    model: str | None = None
    # Solo en memoria mientras dura la ejecucion; nunca se persiste.
    api_key: str | None = None
    temperature: float | None = None
    thinking: bool | None = None
    max_tokens: int | None = None

    def resolved(self) -> dict[str, Any]:
        return {
            "provider": self.provider or settings.llm_provider,
            "model": self.model or settings.llm_model,
            "base_url": self.base_url or "",
            "temperature": self.temperature,
            "thinking": self.thinking,
            "max_tokens": self.max_tokens,
            "has_api_key": bool(self.api_key),
        }


@dataclass
class AgentModel(RoleModel):
    system_prompt: str = ""
    max_iterations: int | None = None

    def resolved(self) -> dict[str, Any]:
        return {
            **super().resolved(),
            "system_prompt": self.system_prompt,
            "max_iterations": self.max_iterations or settings.agent_max_iterations,
        }


@dataclass
class EvalRequest:
    personas_yaml: str
    cases_yaml: str
    agent: AgentModel
    simulator: RoleModel
    judge: RoleModel
    mcp_conn_ids: list[str] = field(default_factory=list)
    name: str = ""
    mlflow_experiment: str = ""
    case_ids: list[str] = field(default_factory=list)
    persona_ids: list[str] = field(default_factory=list)
    repetitions: int = 1
    max_turns_override: int | None = None


class EvalConfigError(ValueError):
    """La peticion no se puede ejecutar (YAML invalido, matriz vacia...)."""


# ---------------------------------------------------------------- ejecucion -
class EvalJob:
    def __init__(self, eval_run_id: str, session_id: str, request: EvalRequest, spec: SuiteSpec,
                 items: list[EvalItemSpec], experiment: str, session_title: str) -> None:
        self.id = eval_run_id
        self.session_id = session_id
        self.request = request
        self.spec = spec
        self.items = items
        self.experiment = experiment
        self.session_title = session_title
        self.result_ids: list[str] = [new_id("evres") for _ in items]
        self.status = "running"
        self.events: list[dict[str, Any]] = []
        self.finished = False
        self.cancel_requested = False
        self.task: asyncio.Task | None = None
        self.parent_run_id = ""
        self._signal = asyncio.Event()
        self._current_conv: str = ""
        self._outcomes: list[dict[str, Any]] = []
        self._started = time.time()

    # ------------------------------------------------------------- eventos --
    def emit(self, kind: str, **payload: Any) -> None:
        self.events.append(
            {"type": kind, "seq": len(self.events), "ts": time.time(), "eval_run_id": self.id, **payload}
        )
        # Despierta a quien este esperando y deja preparada la siguiente espera.
        self._signal.set()
        self._signal = asyncio.Event()

    async def wait_events(self, since: int) -> tuple[list[dict[str, Any]], bool]:
        """Eventos desde `since`; si no hay, espera un rato a que llegue alguno.

        La espera acotada deja al endpoint SSE comprobar periodicamente si el
        cliente se fue.
        """
        signal = self._signal
        if len(self.events) > since or self.finished:
            return self.events[since:], self.finished
        try:
            async with asyncio.timeout(WAIT_EVENTS_S):
                await signal.wait()
        except TimeoutError:
            pass
        return self.events[since:], self.finished

    def cancel(self) -> None:
        self.cancel_requested = True
        if self.task and not self.task.done():
            self.task.cancel()

    # ---------------------------------------------------------------- run ---
    async def run(self) -> None:
        final_status = "done"
        error = ""
        try:
            await self._prepare()
            for index, item in enumerate(self.items):
                if self.cancel_requested:
                    break
                try:
                    await self._run_item(index, item)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - falla el caso, no la bateria
                    logger.exception("Caso %s fallido", self.result_ids[index])
                    await self._item_failed(index, item, f"{type(exc).__name__}: {exc}")
        except asyncio.CancelledError:
            final_status = "cancelled"
            logger.info("Evaluacion %s cancelada", self.id)
            # Se consume la cancelacion para poder cerrar SQLite y MLflow.
            task = asyncio.current_task()
            if task is not None and hasattr(task, "uncancel"):
                task.uncancel()
        except Exception as exc:  # noqa: BLE001 - se reporta en la UI y en SQLite
            logger.exception("Evaluacion %s fallida", self.id)
            final_status = "error"
            error = f"{type(exc).__name__}: {exc}"
            self.emit("log", level="error", message=error)
        if self.cancel_requested and final_status == "done":
            final_status = "cancelled"

        try:
            await self._finalize(final_status, error)
        except Exception:  # noqa: BLE001
            logger.exception("No se pudo cerrar la evaluacion %s", self.id)
        finally:
            self.status = final_status
            self.finished = True
            self.emit("run_end", status=final_status, error=error)

    async def _prepare(self) -> None:
        req = self.request
        agent = req.agent.resolved()
        simulator = req.simulator.resolved()
        judge = req.judge.resolved()
        alive = mcp_manager.alive_ids(req.mcp_conn_ids)
        mcp_urls = [mcp_manager.get(cid).config.url for cid in alive]

        self.emit(
            "run_start",
            session_id=self.session_id,
            name=req.name,
            suite=self.spec.cases.suite.nombre if self.spec.cases else "",
            experiment=self.experiment,
            mcp_urls=mcp_urls,
            items=[self._item_ref(i, item) for i, item in enumerate(self.items)],
        )
        if not alive:
            self.emit(
                "log",
                level="warn",
                message="No hay ningun servidor MCP conectado y marcado: el agente respondera sin herramientas.",
            )

        self.parent_run_id = await tracker.ensure_session_run(
            self.session_id,
            self.session_title,
            self.experiment,
            tags={
                "kind": "evaluation",
                "eval_run_id": self.id,
                "eval_suite": self.spec.cases.suite.nombre if self.spec.cases else "",
            },
        )
        if self.parent_run_id:
            await repo.update_eval_run(self.id, mlflow_run_id=self.parent_run_id)
            await tracker.log_params(
                self.parent_run_id,
                {
                    "eval.name": req.name,
                    "eval.suite": self.spec.cases.suite.nombre if self.spec.cases else "",
                    "eval.items": len(self.items),
                    "eval.repetitions": req.repetitions,
                    "eval.max_turns_override": req.max_turns_override or "",
                    "eval.cases": ", ".join(sorted({i.case.id for i in self.items})),
                    "eval.personas": ", ".join(sorted({i.persona.id for i in self.items})),
                    "eval.personas_sha": text_hash(req.personas_yaml),
                    "eval.cases_sha": text_hash(req.cases_yaml),
                    "agent.provider": agent["provider"],
                    "agent.model": agent["model"],
                    "agent.temperature": agent["temperature"] if agent["temperature"] is not None else settings.llm_temperature,
                    "agent.thinking": agent["thinking"] if agent["thinking"] is not None else settings.llm_thinking,
                    "agent.max_iterations": agent["max_iterations"],
                    "simulator.provider": simulator["provider"],
                    "simulator.model": simulator["model"],
                    "simulator.thinking": simulator["thinking"],
                    "judge.provider": judge["provider"],
                    "judge.model": judge["model"],
                    "judge.thinking": judge["thinking"],
                    "mcp_urls": ", ".join(mcp_urls) or "(ninguno)",
                },
            )
            # Los YAML tal cual se ejecutaron: la bateria es reproducible desde MLflow.
            await tracker.log_text(self.parent_run_id, "evaluation/personas.yaml", req.personas_yaml)
            await tracker.log_text(self.parent_run_id, "evaluation/casos.yaml", req.cases_yaml)

    def _item_ref(self, index: int, item: EvalItemSpec) -> dict[str, Any]:
        return {
            "result_id": self.result_ids[index],
            "seq": index,
            "case_id": item.case.id,
            "case_title": item.case.label,
            "persona_id": item.persona.id,
            "persona_name": item.persona.nombre,
            "repetition": item.repetition,
            "max_turns": item.max_turns,
        }

    # ------------------------------------------------------------ un caso ---
    async def _run_item(self, index: int, item: EvalItemSpec) -> None:
        req = self.request
        result_id = self.result_ids[index]
        case, persona = item.case, item.persona
        started = time.perf_counter()
        suffix = f" #{item.repetition}" if req.repetitions > 1 else ""
        title = f"[eval] {case.id} · {persona.id}{suffix}"

        agent_cfg = req.agent.resolved()
        conversation = await repo.create_conversation(
            self.session_id,
            title=title,
            provider=agent_cfg["provider"],
            model=agent_cfg["model"],
            system_prompt=req.agent.system_prompt,
            mcp_servers=req.mcp_conn_ids,
        )
        conv_id = conversation["id"]
        self._current_conv = conv_id
        await repo.update_conversation(
            conv_id,
            metadata={
                "kind": "evaluation",
                "eval_run_id": self.id,
                "eval_result_id": result_id,
                "case_id": case.id,
                "persona_id": persona.id,
                "repetition": item.repetition,
            },
        )
        await repo.update_eval_result(
            result_id, status="running", conversation_id=conv_id, started_at=time.time()
        )
        self.emit("item_start", conversation_id=conv_id, **self._item_ref(index, item))

        eval_tags = {
            "kind": "evaluation",
            "eval_run_id": self.id,
            "eval_result_id": result_id,
            "eval_case_id": case.id,
            "eval_persona_id": persona.id,
            "eval_repetition": str(item.repetition),
        }

        simulator = UserSimulator(
            await get_provider(req.simulator.provider, req.simulator.base_url, req.simulator.model,
                               req.simulator.api_key),
            persona,
            case,
            temperature=req.simulator.temperature,
            thinking=req.simulator.thinking,
            max_tokens=req.simulator.max_tokens,
        )

        transcript: list[dict[str, Any]] = []
        dialogue: list[tuple[str, str]] = []
        trace_ids: list[str] = []
        sim_usage, sim_latency, sim_calls = Usage(), 0.0, 0
        agent_totals: dict[str, float] = {}
        end_reason = "max_turnos"
        agent_turns = 0

        for turn in range(1, item.max_turns + 1):
            if self.cancel_requested:
                end_reason = "cancelada"
                break

            # --- mensaje de la persona ---------------------------------------
            closing = False
            if turn == 1 and case.mensaje_inicial.strip():
                text, source = case.mensaje_inicial.strip(), "guion"
            else:
                self.emit("item_phase", result_id=result_id, phase="simulando", turn=turn)
                sim_turn = await simulator.next_message(dialogue, turn, item.max_turns)
                sim_calls += 1
                sim_usage = sim_usage.merge(sim_turn.usage)
                sim_latency += sim_turn.latency_ms
                text, source, closing = sim_turn.text, "simulador", sim_turn.finished
                # FIN por tic del modelo (en la primera pregunta o en una
                # repregunta): el mensaje se envia igualmente. Ver `honors_fin`.
                if closing and not honors_fin(text, dialogue):
                    closing = False

            if closing:
                # La despedida se registra, pero no se le manda al agente: su
                # respuesta ("de nada") no aporta nada a la evaluacion y en CPU
                # cuesta un turno entero.
                end_reason = "persona_termina"
                if text:
                    transcript.append(
                        {"turn": turn, "role": "user", "text": text, "source": source, "closing": True}
                    )
                    self.emit("turn_user", result_id=result_id, turn=turn, text=text, source=source,
                              closing=True)
                break

            transcript.append({"turn": turn, "role": "user", "text": text, "source": source})
            dialogue.append(("user", text))
            self.emit("turn_user", result_id=result_id, turn=turn, text=text, source=source)

            # --- respuesta del agente bajo prueba ------------------------------
            self.emit("item_phase", result_id=result_id, phase="agente", turn=turn)
            entry = await self._agent_turn(result_id, conv_id, turn, text, eval_tags)
            agent_turns += 1
            transcript.append(entry)
            dialogue.append(("agent", entry["text"] or f"(error: {entry['error']})"))
            if entry.get("trace_id"):
                trace_ids.append(entry["trace_id"])
            for key, value in (entry.get("metrics") or {}).items():
                if isinstance(value, (int, float)):
                    agent_totals[key] = agent_totals.get(key, 0.0) + float(value)
            if entry["error"]:
                end_reason = "error_agente"
                break

        if self.cancel_requested:
            raise asyncio.CancelledError

        # --- evaluacion --------------------------------------------------------
        self.emit("item_phase", result_id=result_id, phase="evaluando", turn=len(transcript))
        tool_events = sorted(
            await repo.list_tool_events(conversation_id=conv_id, limit=5000),
            key=lambda e: e["created_at"],
        )
        agent_texts = [e["text"] for e in transcript if e["role"] == "agent" and e.get("text")]
        checks = deterministic_checks(case, agent_texts, tool_events)

        judge = Judge(
            await get_provider(req.judge.provider, req.judge.base_url, req.judge.model, req.judge.api_key),
            temperature=req.judge.temperature if req.judge.temperature is not None else 0.0,
            thinking=req.judge.thinking,
            max_tokens=req.judge.max_tokens,
        )
        judge_error = ""
        try:
            outcome = await judge.evaluate(case, persona, transcript, checks)
            judge_error = outcome.error
        except Exception as exc:  # noqa: BLE001 - un evaluador caido no tumba la bateria
            logger.exception("Fallo del evaluador en %s", result_id)
            outcome = JudgeOutcome(verdict=None, error=f"{type(exc).__name__}: {exc}")
            judge_error = outcome.error

        items = judged_items(case, outcome.verdict) + checks
        agg = aggregate(items, item.threshold)
        if judge_error:
            status, score, passed = "error", None, None
        else:
            status = "passed" if agg["passed"] else "failed"
            score, passed = agg["score"], agg["passed"]

        metrics = {
            "turns": agent_turns,
            "wall_ms": round((time.perf_counter() - started) * 1000, 1),
            "agent_total_tokens": agent_totals.get("total_tokens", 0),
            "agent_prompt_tokens": agent_totals.get("prompt_tokens", 0),
            "agent_completion_tokens": agent_totals.get("completion_tokens", 0),
            "agent_latency_ms": round(agent_totals.get("latency_ms", 0.0), 1),
            "agent_llm_latency_ms": round(agent_totals.get("llm_latency_ms", 0.0), 1),
            "agent_mcp_latency_ms": round(agent_totals.get("mcp_latency_ms", 0.0), 1),
            "agent_llm_calls": agent_totals.get("llm_calls", 0),
            "tool_calls": len(tool_events),
            "tool_errors": sum(1 for e in tool_events if not e["ok"]),
            "sim_calls": sim_calls,
            "sim_total_tokens": sim_usage.total_tokens,
            "sim_latency_ms": round(sim_latency, 1),
            "judge_total_tokens": outcome.usage.total_tokens,
            "judge_latency_ms": round(outcome.latency_ms, 1),
            "judge_attempts": outcome.attempts,
        }
        verdict_doc = {
            "resumen": (outcome.verdict or {}).get("resumen", ""),
            "raw": outcome.verdict,
            "raw_text": outcome.raw if outcome.verdict is None else "",
            "error": judge_error,
            "aggregate": agg,
            "judge_model": req.judge.resolved()["model"],
        }

        # --- MLflow ------------------------------------------------------------
        run_id = await self._log_item_mlflow(
            index, item, result_id, conv_id, title, eval_tags, transcript, items, agg,
            status, end_reason, metrics, verdict_doc, outcome, trace_ids,
        )

        await repo.update_eval_result(
            result_id,
            status=status,
            score=score,
            passed=passed,
            turns=agent_turns,
            end_reason=end_reason,
            items=items,
            verdict=verdict_doc,
            transcript=transcript,
            metrics=metrics,
            error=judge_error,
            ended_at=time.time(),
            mlflow_run_id=run_id,
            mlflow_trace_ids=trace_ids,
        )
        outcome_row = {
            **self._item_ref(index, item),
            "status": status,
            "score": score,
            "passed": passed,
            "turns": agent_turns,
            "end_reason": end_reason,
            "metrics": metrics,
        }
        self._outcomes.append(outcome_row)
        self._current_conv = ""
        self.emit(
            "item_end",
            result_id=result_id,
            status=status,
            score=score,
            passed=passed,
            turns=agent_turns,
            end_reason=end_reason,
            resumen=verdict_doc["resumen"],
            error=judge_error,
            failed_mandatory=agg["failed_mandatory"],
            metrics=metrics,
            mlflow_run_id=run_id,
        )

    async def _agent_turn(
        self, result_id: str, conv_id: str, turn: int, text: str, eval_tags: dict[str, str]
    ) -> dict[str, Any]:
        req = self.request
        runner = AgentRunner(
            RunConfig(
                session_id=self.session_id,
                conversation_id=conv_id,
                message=text,
                mcp_conn_ids=req.mcp_conn_ids,
                provider=req.agent.provider,
                base_url=req.agent.base_url,
                model=req.agent.model,
                api_key=req.agent.api_key,
                temperature=req.agent.temperature,
                max_tokens=req.agent.max_tokens,
                thinking=req.agent.thinking,
                system_prompt=req.agent.system_prompt,
                max_iterations=req.agent.max_iterations,
                mlflow_experiment=self.experiment,
                trace_tags={**eval_tags, "eval_role": "agent", "eval_turn": str(turn)},
                run_tags=eval_tags,
            )
        )
        entry: dict[str, Any] = {
            "turn": turn,
            "role": "agent",
            "text": "",
            "tool_calls": [],
            "metrics": {},
            "interaction_id": runner.interaction_id,
            "trace_id": "",
            "error": "",
        }
        try:
            async for event in runner.run():
                kind = event["type"]
                if kind == "tool_call":
                    self.emit("agent_tool_call", result_id=result_id, turn=turn, call_id=event["call_id"],
                              tool=event["tool"], arguments=event["arguments"])
                elif kind == "tool_result":
                    self.emit(
                        "agent_tool_result",
                        result_id=result_id,
                        turn=turn,
                        call_id=event["call_id"],
                        tool=event["tool"],
                        ok=event["ok"],
                        error=event.get("error"),
                        latency_ms=event.get("latency_ms"),
                        preview=(event.get("text") or "")[:UI_PREVIEW_CHARS],
                    )
                elif kind == "final":
                    entry["text"] = event.get("content") or ""
                    entry["metrics"] = event.get("metrics") or {}
                    entry["trace_id"] = event.get("mlflow_trace_id") or ""
                    entry["error"] = event.get("error") or entry["error"]
                elif kind == "error":
                    entry["error"] = event.get("message", "error desconocido")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - p.ej. proveedor sin clave: fallo del caso, no de la bateria
            logger.exception("El agente fallo en %s", result_id)
            entry["error"] = f"{type(exc).__name__}: {exc}"

        # Las llamadas al MCP con su resultado, tal y como quedaron persistidas.
        events = sorted(await repo.list_tool_events(interaction_id=runner.interaction_id), key=lambda e: e["seq"])
        entry["tool_calls"] = [
            {
                "seq": e["seq"],
                "tool": e["tool_name"],
                "arguments": e["arguments"],
                "ok": e["ok"],
                "error": e["error"],
                "latency_ms": e["latency_ms"],
                "result": (e["result_text"] or "")[:TRANSCRIPT_RESULT_CHARS],
                "result_chars": len(e["result_text"] or ""),
            }
            for e in events
        ]
        self.emit(
            "turn_agent",
            result_id=result_id,
            turn=turn,
            text=entry["text"],
            error=entry["error"],
            metrics=entry["metrics"],
            interaction_id=runner.interaction_id,
        )
        return entry

    # ------------------------------------------------------ MLflow por caso -
    async def _log_item_mlflow(
        self,
        index: int,
        item: EvalItemSpec,
        result_id: str,
        conv_id: str,
        title: str,
        eval_tags: dict[str, str],
        transcript: list[dict[str, Any]],
        items: list[dict[str, Any]],
        agg: dict[str, Any],
        status: str,
        end_reason: str,
        metrics: dict[str, Any],
        verdict_doc: dict[str, Any],
        outcome: Any,
        trace_ids: list[str],
    ) -> str:
        if not tracker.available:
            return ""
        req = self.request
        case, persona = item.case, item.persona
        judge_cfg = req.judge.resolved()

        # Normalmente ya existe (lo creo el primer turno del agente); si el
        # agente fallo antes de abrir la traza, se crea aqui.
        run_id, _ = await tracker.ensure_conversation_run(
            self.session_id, conv_id, title, None, self.experiment, self.session_title, tags=eval_tags
        )
        if not run_id:
            return ""

        await tracker.log_params(
            run_id,
            {
                "eval.case_id": case.id,
                "eval.case_title": case.label,
                "eval.persona_id": persona.id,
                "eval.persona_name": persona.nombre,
                "eval.repetition": item.repetition,
                "eval.max_turns": item.max_turns,
                "eval.threshold": item.threshold,
                "eval.expected_type": case.resultado_esperado.tipo,
                "simulator.model": req.simulator.resolved()["model"],
                "judge.model": judge_cfg["model"],
            },
        )
        numeric = {
            "eval.turns": metrics["turns"],
            "eval.items_passed": agg["items_passed"],
            "eval.items_total": agg["items_total"],
            "eval.wall_ms": metrics["wall_ms"],
            "sim.total_tokens": metrics["sim_total_tokens"],
            "sim.latency_ms": metrics["sim_latency_ms"],
            "judge.total_tokens": metrics["judge_total_tokens"],
            "judge.latency_ms": metrics["judge_latency_ms"],
            "agent.total_tokens": metrics["agent_total_tokens"],
            "agent.tool_calls": metrics["tool_calls"],
        }
        if status != "error":
            numeric["eval.score"] = agg["score"]
            numeric["eval.passed"] = 1.0 if agg["passed"] else 0.0
        await tracker.log_metrics(run_id, numeric)
        await tracker.set_tags(
            run_id,
            {"eval.status": status, "eval.end_reason": end_reason, "eval.score": agg["score"]},
        )
        await tracker.log_artifact(
            run_id,
            "evaluation/result.json",
            {
                "eval_run_id": self.id,
                "result_id": result_id,
                "session_id": self.session_id,
                "conversation_id": conv_id,
                "case": case.model_dump(),
                "persona": persona.model_dump(),
                "repetition": item.repetition,
                "status": status,
                "aggregate": agg,
                "items": items,
                "verdict": verdict_doc,
                "metrics": metrics,
                "transcript": transcript,
                "agent_trace_ids": trace_ids,
            },
        )
        await tracker.log_artifact(
            run_id,
            "evaluation/judge.json",
            {"messages": outcome.messages, "raw": outcome.raw, "thinking": outcome.thinking,
             "attempts": outcome.attempts, "error": outcome.error},
        )
        await tracker.log_table(
            run_id,
            [
                {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
                 for k, v in it.items()}
                for it in items
            ],
            "evaluation/criteria.json",
        )

        judge_trace = await tracker.log_standalone_trace(
            f"eval_judge::{case.id}",
            "EVALUATOR",
            inputs={"case": case.id, "persona": persona.id, "transcript": render_transcript(transcript)},
            outputs={"verdict": outcome.verdict, "aggregate": agg, "raw": outcome.raw},
            experiment=self.experiment,
            tags={
                **eval_tags,
                "session_id": self.session_id,
                "conversation_id": conv_id,
                "eval_role": "judge",
            },
            attributes={
                ATTR_LLM_MODEL: judge_cfg["model"],
                ATTR_LLM_PROVIDER: judge_cfg["provider"],
                ATTR_CHAT_USAGE: {
                    "input_tokens": outcome.usage.prompt_tokens,
                    "output_tokens": outcome.usage.completion_tokens,
                    "total_tokens": outcome.usage.total_tokens,
                },
                "latency_ms": outcome.latency_ms,
                "attempts": outcome.attempts,
            },
            source_run_id=run_id,
            error=outcome.error or None,
        )
        if judge_trace:
            await tracker.set_tags(run_id, {"eval.judge_trace_id": judge_trace})

        # Los juicios se cuelgan de la ultima traza del agente: en la vista de
        # traces de MLflow aparecen como "assessments" de esa conversacion.
        if trace_ids:
            feedbacks = [
                {
                    "name": it["id"],
                    "value": bool(it["cumple"]),
                    "rationale": it.get("detalle", ""),
                    "source_type": "LLM_JUDGE" if it["origen"] == "evaluador" else "CODE",
                    "source_id": judge_cfg["model"] if it["origen"] == "evaluador" else "agente-pruebas.checks",
                    "metadata": {"peso": it.get("peso"), "obligatorio": it.get("obligatorio"),
                                 "eval_result_id": result_id},
                }
                for it in items
            ]
            if status != "error":
                feedbacks.append({
                    "name": "eval_score", "value": agg["score"], "source_type": "CODE",
                    "source_id": "agente-pruebas.aggregate",
                    "rationale": verdict_doc["resumen"],
                    "metadata": {"threshold": agg["threshold"], "eval_result_id": result_id},
                })
                feedbacks.append({
                    "name": "eval_passed", "value": bool(agg["passed"]), "source_type": "CODE",
                    "source_id": "agente-pruebas.aggregate",
                    "rationale": (
                        "Obligatorios no cumplidos: " + ", ".join(agg["failed_mandatory"])
                        if agg["failed_mandatory"] else ""
                    ),
                })
            expected = case.resultado_esperado
            await tracker.log_assessments(
                trace_ids[-1],
                feedbacks,
                expectations=[{
                    "name": "resultado_esperado",
                    "value": {k: v for k, v in expected.model_dump().items()
                              if k in {"tipo", "descripcion", "valor", "unidad", "tolerancia"}},
                    "source_type": "HUMAN",
                    "source_id": "casos.yaml",
                }],
            )

        await tracker.log_metrics(
            self.parent_run_id,
            {"eval.item_score": agg["score"] if status != "error" else 0.0},
            step=index,
        )
        await tracker.terminate_run(run_id, "FAILED" if status == "error" else "FINISHED")
        return run_id

    async def _item_failed(self, index: int, item: EvalItemSpec, error: str) -> None:
        """Un caso que revienta (simulador caido, MCP roto...) se anota y se sigue."""
        result_id = self.result_ids[index]
        await repo.update_eval_result(result_id, status="error", error=error, ended_at=time.time())
        await repo.cancel_running_interactions(self.session_id)
        if self._current_conv:
            await tracker.close_conversation_run(self._current_conv)
            self._current_conv = ""
        self._outcomes.append(
            {**self._item_ref(index, item), "status": "error", "score": None, "passed": None,
             "turns": 0, "end_reason": "error", "metrics": {}}
        )
        self.emit("item_end", result_id=result_id, status="error", score=None, passed=None, turns=0,
                  end_reason="error", resumen="", error=error, failed_mandatory=[], metrics={})

    # ------------------------------------------------------------- cierre ---
    def _summary(self, status: str) -> dict[str, Any]:
        rows = self._outcomes
        judged = [r for r in rows if r["status"] in {"passed", "failed"}]
        scores = [r["score"] for r in judged if r["score"] is not None]

        def group(key: str) -> dict[str, dict[str, Any]]:
            out: dict[str, dict[str, Any]] = {}
            for row in judged:
                bucket = out.setdefault(row[key], {"runs": 0, "passed": 0, "scores": []})
                bucket["runs"] += 1
                bucket["passed"] += 1 if row["passed"] else 0
                bucket["scores"].append(row["score"] or 0.0)
            for bucket in out.values():
                values = bucket.pop("scores")
                bucket["avg_score"] = round(sum(values) / len(values), 4) if values else 0.0
                bucket["pass_rate"] = round(bucket["passed"] / bucket["runs"], 4) if bucket["runs"] else 0.0
            return out

        def total(metric: str) -> float:
            return sum(float(r["metrics"].get(metric, 0) or 0) for r in rows)

        passed = sum(1 for r in judged if r["passed"])
        return {
            "status": status,
            "total": len(self.items),
            "completed": len(rows),
            "passed": passed,
            "failed": len(judged) - passed,
            "errors": sum(1 for r in rows if r["status"] == "error"),
            "not_run": len(self.items) - len(rows),
            "pass_rate": round(passed / len(judged), 4) if judged else 0.0,
            "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "by_case": group("case_id"),
            "by_persona": group("persona_id"),
            "tokens": {
                "agent": int(total("agent_total_tokens")),
                "simulator": int(total("sim_total_tokens")),
                "judge": int(total("judge_total_tokens")),
            },
            "tool_calls": int(total("tool_calls")),
            "tool_errors": int(total("tool_errors")),
            "duration_s": round(time.time() - self._started, 1),
        }

    async def _finalize(self, status: str, error: str) -> None:
        # Lo que no llego a ejecutarse (o se corto a medias) queda marcado.
        done_ids = {r["result_id"] for r in self._outcomes}
        for result_id in self.result_ids:
            if result_id not in done_ids:
                await repo.update_eval_result(
                    result_id, status="error" if status == "error" else "cancelled", ended_at=time.time()
                )
        if status in {"cancelled", "error"}:
            await repo.cancel_running_interactions(self.session_id)
            if self._current_conv:
                await tracker.close_conversation_run(self._current_conv)

        summary = self._summary(status)
        await repo.update_eval_run(self.id, status=status, ended_at=time.time(), summary=summary, error=error)
        await repo.touch_session(self.session_id)

        if self.parent_run_id:
            await tracker.log_metrics(
                self.parent_run_id,
                {
                    "eval.pass_rate": summary["pass_rate"],
                    "eval.avg_score": summary["avg_score"],
                    "eval.total": summary["total"],
                    "eval.completed": summary["completed"],
                    "eval.passed": summary["passed"],
                    "eval.failed": summary["failed"],
                    "eval.errors": summary["errors"],
                    "eval.duration_s": summary["duration_s"],
                    "tokens.agent": summary["tokens"]["agent"],
                    "tokens.simulator": summary["tokens"]["simulator"],
                    "tokens.judge": summary["tokens"]["judge"],
                    "mcp.tool_calls": summary["tool_calls"],
                    "mcp.tool_errors": summary["tool_errors"],
                },
            )
            await tracker.log_artifact(self.parent_run_id, "evaluation/summary.json", summary)
            await tracker.log_table(
                self.parent_run_id,
                [
                    {
                        "case_id": r["case_id"],
                        "case_title": r["case_title"],
                        "persona_id": r["persona_id"],
                        "repetition": r["repetition"],
                        "status": r["status"],
                        "score": r["score"],
                        "passed": r["passed"],
                        "turns": r["turns"],
                        "end_reason": r["end_reason"],
                        "agent_tokens": r["metrics"].get("agent_total_tokens"),
                        "tool_calls": r["metrics"].get("tool_calls"),
                        "wall_ms": r["metrics"].get("wall_ms"),
                        "result_id": r["result_id"],
                    }
                    for r in self._outcomes
                ],
                "evaluation/results.json",
            )
            await tracker.set_tags(self.parent_run_id, {"eval.status": status})
            await tracker.terminate_run(
                self.parent_run_id,
                {"done": "FINISHED", "cancelled": "KILLED"}.get(status, "FAILED"),
            )


# ---------------------------------------------------------------- registro --
class EvalManager:
    """Ejecuciones vivas en este proceso."""

    def __init__(self) -> None:
        self._jobs: dict[str, EvalJob] = {}

    def get(self, eval_run_id: str) -> EvalJob | None:
        return self._jobs.get(eval_run_id)

    def running(self) -> list[str]:
        return [job_id for job_id, job in self._jobs.items() if not job.finished]

    def is_session_busy(self, session_id: str) -> bool:
        """La sesion pertenece a una evaluacion en marcha (no se puede borrar)."""
        return any(job.session_id == session_id and not job.finished for job in self._jobs.values())

    async def start(self, request: EvalRequest) -> EvalJob:
        spec = parse_suite(request.personas_yaml, request.cases_yaml)
        if not spec.ok:
            raise EvalConfigError("; ".join(spec.errors) or "Los ficheros YAML no son validos")
        items = build_matrix(
            spec,
            case_ids=request.case_ids or None,
            persona_ids=request.persona_ids or None,
            repetitions=request.repetitions,
            max_turns_override=request.max_turns_override,
        )
        if not items:
            raise EvalConfigError("La seleccion no produce ninguna ejecucion (revisa casos y personas marcados)")

        # Los tres modelos se resuelven antes de arrancar: una clave que falta
        # debe fallar ahora, no a mitad de la bateria.
        for role, cfg in (("agente", request.agent), ("simulador", request.simulator), ("evaluador", request.judge)):
            try:
                await get_provider(cfg.provider, cfg.base_url, cfg.model, cfg.api_key)
            except Exception as exc:  # noqa: BLE001
                raise EvalConfigError(f"Modelo del {role}: {exc}") from exc

        assert spec.cases is not None
        suite = spec.cases.suite.nombre
        name = request.name.strip() or f"{suite} · {time.strftime('%d/%m %H:%M')}"
        request.name = name
        experiment = request.mlflow_experiment.strip()
        session_title = f"eval · {name}"
        eval_run_id = new_id("evr")

        session = await repo.create_session(
            session_title,
            metadata={"kind": "evaluation", "eval_run_id": eval_run_id},
            mlflow_experiment=experiment,
        )
        config = {
            "agent": request.agent.resolved(),
            "simulator": request.simulator.resolved(),
            "judge": request.judge.resolved(),
            "mcp_conn_ids": request.mcp_conn_ids,
            "mcp_urls": [mcp_manager.get(c).config.url for c in mcp_manager.alive_ids(request.mcp_conn_ids)],
            "case_ids": request.case_ids,
            "persona_ids": request.persona_ids,
            "repetitions": request.repetitions,
            "max_turns_override": request.max_turns_override,
        }
        await repo.create_eval_run(
            eval_run_id,
            session["id"],
            name=name,
            suite=suite,
            config=config,
            personas_yaml=request.personas_yaml,
            cases_yaml=request.cases_yaml,
            mlflow_experiment=experiment,
        )

        job = EvalJob(eval_run_id, session["id"], request, spec, items, experiment, session_title)
        for index, item in enumerate(items):
            await repo.create_eval_result(
                job.result_ids[index],
                eval_run_id,
                seq=index,
                case_id=item.case.id,
                case_title=item.case.label,
                persona_id=item.persona.id,
                persona_name=item.persona.nombre,
                repetition=item.repetition,
            )
        self._forget_old_jobs()
        self._jobs[eval_run_id] = job
        job.task = asyncio.create_task(job.run(), name=f"eval:{eval_run_id}")
        return job

    def _forget_old_jobs(self) -> None:
        """Las terminadas solo sirven para reengancharse; su resultado esta en SQLite."""
        finished = [job_id for job_id, job in self._jobs.items() if job.finished]
        for job_id in finished[: max(len(finished) - KEEP_FINISHED_JOBS, 0)]:
            self._jobs.pop(job_id, None)

    async def stream(self, job: EvalJob, since: int) -> AsyncIterator[dict[str, Any]]:
        index = since
        while True:
            events, finished = await job.wait_events(index)
            for event in events:
                yield event
            index += len(events)
            if finished and index >= len(job.events):
                return

    async def shutdown(self) -> None:
        for job in self._jobs.values():
            job.cancel()
        pending = [j.task for j in self._jobs.values() if j.task and not j.task.done()]
        if pending:
            await asyncio.wait(pending, timeout=10)


def request_public(request: EvalRequest) -> dict[str, Any]:
    """La peticion sin secretos, para devolverla a la UI."""
    data = asdict(request)
    for role in ("agent", "simulator", "judge"):
        data[role].pop("api_key", None)
    return data


eval_manager = EvalManager()
