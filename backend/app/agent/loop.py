"""Bucle del agente: LLM <-> herramientas MCP.

Una "interaccion" es un mensaje del usuario y todo lo que el agente hace hasta
responder: N llamadas al modelo y M llamadas a herramientas MCP. El bucle va
emitiendo eventos segun avanza para que la UI muestre la conversacion en
tiempo real, y a la vez persiste todo en SQLite y en MLflow.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from ..config import settings
from ..llm.base import LLMProvider, LLMResponse, Usage
from ..llm.registry import get_provider
from ..mcpclient.manager import ToolRouter, mcp_manager
from ..observability.mlflow_tracker import (
    ATTR_CHAT_TOOLS,
    ATTR_CHAT_USAGE,
    ATTR_LLM_MODEL,
    ATTR_LLM_PROVIDER,
    InteractionTrace,
    tracker,
)
from ..store import repository as repo
from ..store.db import new_id
from .prompts import TOOL_BUDGET_NOTE, build_system_prompt

logger = logging.getLogger(__name__)

# Recorte del texto que se le devuelve al modelo como observacion: es un
# limite de presupuesto de contexto, no de registro.
MAX_TOOL_RESULT_CHARS = 12000
# Recorte del texto que viaja a la UI por SSE. La UI muestra un extracto; el
# contenido integro se recupera de SQLite (/api/traces) y de MLflow.
MAX_UI_RESULT_CHARS = 4000


@dataclass
class RunConfig:
    """Todo lo que la UI puede decidir por turno."""

    session_id: str
    conversation_id: str
    message: str
    mcp_conn_ids: list[str] = field(default_factory=list)
    provider: str | None = None
    base_url: str | None = None
    model: str | None = None
    # Llega desde la UI por peticion; nunca se persiste ni se registra.
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    thinking: bool | None = None
    system_prompt: str = ""
    max_iterations: int | None = None
    # Experimento de MLflow donde registrar. Vacio => el de la sesion o el del
    # backend. Se resuelve en `run()` a partir de la sesion.
    mlflow_experiment: str = ""
    # Etiquetas extra para la traza de MLflow. Las evaluaciones marcan aqui
    # la ejecucion, el caso y la persona para poder filtrar por ellos.
    trace_tags: dict[str, str] = field(default_factory=dict)
    # Etiquetas del run de conversacion, si este turno es el que lo crea.
    run_tags: dict[str, str] = field(default_factory=dict)


@dataclass
class RunMetrics:
    llm_calls: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    iterations: int = 0
    llm_latency_ms: float = 0.0
    mcp_latency_ms: float = 0.0
    usage: Usage = field(default_factory=Usage)

    def as_mlflow(self, total_ms: float) -> dict[str, float]:
        seconds = max(total_ms / 1000.0, 1e-6)
        gen_seconds = max(self.llm_latency_ms / 1000.0, 1e-6)
        return {
            "latency_ms": total_ms,
            "llm_latency_ms": self.llm_latency_ms,
            "mcp_latency_ms": self.mcp_latency_ms,
            "overhead_ms": max(total_ms - self.llm_latency_ms - self.mcp_latency_ms, 0.0),
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "total_tokens": self.usage.total_tokens,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "tool_success_rate": (
                (self.tool_calls - self.tool_errors) / self.tool_calls if self.tool_calls else 1.0
            ),
            "iterations": self.iterations,
            "output_tokens_per_second": self.usage.completion_tokens / gen_seconds,
            "total_tokens_per_second": self.usage.total_tokens / seconds,
        }


def _slug(name: str) -> str:
    """Nombre de herramienta utilizable como fichero de artifact."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return safe[:80] or "tool"


def _clip(text: str, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncado, {len(text) - limit} caracteres omitidos]"


class AgentRunner:
    """Ejecuta una interaccion y va emitiendo eventos."""

    def __init__(self, config: RunConfig) -> None:
        self.cfg = config
        self.interaction_id = new_id("int")
        self.metrics = RunMetrics()
        self.trace: InteractionTrace | None = None
        self.tool_seq = 0
        self.transcript: list[dict[str, Any]] = []

    # ------------------------------------------------------------- helpers --
    async def _load_history(self) -> list[dict[str, Any]]:
        rows = await repo.get_messages(self.cfg.conversation_id, limit=settings.agent_history_max_messages)
        history: list[dict[str, Any]] = []
        for row in rows:
            if row["role"] == "system":
                continue
            item: dict[str, Any] = {"role": row["role"], "content": row["content"]}
            if row["tool_calls"]:
                item["tool_calls"] = row["tool_calls"]
            if row["tool_call_id"]:
                item["tool_call_id"] = row["tool_call_id"]
            if row["name"]:
                item["name"] = row["name"]
            history.append(item)
        return history

    def _event(self, kind: str, **payload: Any) -> dict[str, Any]:
        return {
            "type": kind,
            "ts": time.time(),
            "session_id": self.cfg.session_id,
            "conversation_id": self.cfg.conversation_id,
            "interaction_id": self.interaction_id,
            **payload,
        }

    # ---------------------------------------------------------------- run ---
    async def run(self) -> AsyncIterator[dict[str, Any]]:
        started = time.perf_counter()
        conversation = await repo.get_conversation(self.cfg.conversation_id)
        if conversation is None:
            yield self._event("error", message=f"Conversacion no encontrada: {self.cfg.conversation_id}")
            return

        # El experimento de MLflow lo manda la UI; si no viene, se hereda de la
        # sesion y, en ultima instancia, de la configuracion del backend.
        session = await repo.get_session(self.cfg.session_id)
        experiment = (
            self.cfg.mlflow_experiment
            or (session or {}).get("mlflow_experiment", "")
            or ""
        )

        alive_ids = mcp_manager.alive_ids(self.cfg.mcp_conn_ids)
        router = ToolRouter(mcp_manager, alive_ids)
        servers = [mcp_manager.get(cid).describe() for cid in alive_ids]
        server_summary = [
            {
                "conn_id": s["conn_id"],
                "url": s["config"]["url"],
                "server_name": s["server_info"].get("name", ""),
                "tools": len(s["tools"]),
            }
            for s in servers
        ]

        provider_name = self.cfg.provider or settings.llm_provider
        model_name = self.cfg.model or conversation.get("model") or settings.llm_model
        provider = await get_provider(
            provider_name, self.cfg.base_url, model_name, self.cfg.api_key
        )

        await repo.create_interaction(
            self.interaction_id,
            self.cfg.conversation_id,
            self.cfg.session_id,
            self.cfg.message,
            provider_name,
            model_name,
            server_summary,
        )

        self.trace = await tracker.start_interaction(
            self.cfg.session_id,
            self.cfg.conversation_id,
            self.interaction_id,
            self.cfg.message,
            attributes={
                "provider": provider_name,
                "model": model_name,
                "mcp_servers": json.dumps(server_summary, ensure_ascii=False),
                "tools_available": len(router.specs),
            },
            conversation_title=conversation.get("title", ""),
            experiment=experiment,
            # Para que el run padre se llame como la sesion en la UI y no con
            # su identificador.
            session_title=(session or {}).get("title", ""),
            extra_tags=self.cfg.trace_tags,
            run_tags=self.cfg.run_tags,
            params={
                "provider": provider_name,
                "model": model_name,
                "base_url": self.cfg.base_url or settings.llm_base_url,
                "temperature": self.cfg.temperature if self.cfg.temperature is not None else settings.llm_temperature,
                "num_ctx": settings.llm_num_ctx,
                "max_tokens": self.cfg.max_tokens or settings.llm_max_tokens,
                "thinking": self.cfg.thinking if self.cfg.thinking is not None else settings.llm_thinking,
                "max_iterations": self.cfg.max_iterations or settings.agent_max_iterations,
                "mcp_urls": ", ".join(s["url"] for s in server_summary) or "(ninguno)",
            },
        )
        if self.trace.run_id:
            await repo.update_conversation(self.cfg.conversation_id, mlflow_run_id=self.trace.run_id)

        yield self._event(
            "start",
            message=self.cfg.message,
            model=model_name,
            provider=provider_name,
            servers=server_summary,
            tools=router.catalog(),
            mlflow_trace_id=self.trace.trace_id,
            mlflow_experiment=self.trace.experiment,
        )

        await repo.add_message(
            self.cfg.conversation_id, "user", self.cfg.message, interaction_id=self.interaction_id
        )

        system_prompt = build_system_prompt(
            self.cfg.system_prompt or conversation.get("system_prompt", "") or settings.agent_system_prompt,
            has_tools=not router.is_empty,
            servers=server_summary,
        )
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages.extend(await self._load_history())

        final_answer = ""
        error_text = ""
        max_iterations = self.cfg.max_iterations or settings.agent_max_iterations

        try:
            for iteration in range(1, max_iterations + 1):
                self.metrics.iterations = iteration
                yield self._event("status", stage="llm", iteration=iteration, message="Consultando al modelo")

                response = await self._call_llm(provider, messages, router, iteration)

                if response.thinking:
                    yield self._event("thinking", iteration=iteration, text=response.thinking)
                yield self._event(
                    "llm_usage",
                    iteration=iteration,
                    usage=response.usage.as_dict(),
                    latency_ms=response.latency_ms,
                    finish_reason=response.finish_reason,
                    timings=response.raw,
                )

                if not response.tool_calls:
                    final_answer = response.content
                    await repo.add_message(
                        self.cfg.conversation_id,
                        "assistant",
                        final_answer,
                        interaction_id=self.interaction_id,
                        thinking=response.thinking,
                        usage=response.usage.as_dict(),
                    )
                    break

                # El modelo pide herramientas: se guarda su turno y se ejecutan.
                # `extra` lleva lo que el proveedor exige recibir de vuelta
                # (Gemini 3 firma cada llamada); va dentro de `tool_calls`, asi
                # que se persiste y sobrevive a recargar el historial.
                serialized_calls = [
                    {
                        "id": c.id,
                        "name": c.name,
                        "arguments": c.arguments,
                        **({"extra": c.extra} if c.extra else {}),
                    }
                    for c in response.tool_calls
                ]
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content,
                        "tool_calls": serialized_calls,
                        # Claude exige recibir el turno intacto (con sus bloques
                        # de razonamiento firmados) junto a los tool_result.
                        "_provider_blocks": response.raw_blocks,
                    }
                )
                await repo.add_message(
                    self.cfg.conversation_id,
                    "assistant",
                    response.content,
                    interaction_id=self.interaction_id,
                    thinking=response.thinking,
                    tool_calls=serialized_calls,
                    usage=response.usage.as_dict(),
                )
                if response.content:
                    yield self._event("assistant_partial", iteration=iteration, content=response.content)

                for call in response.tool_calls:
                    async for event in self._execute_tool(router, call, iteration, messages):
                        yield event
            else:
                # Se agoto el presupuesto de iteraciones: se fuerza un cierre.
                messages.append({"role": "user", "content": TOOL_BUDGET_NOTE})
                closing = await self._call_llm(provider, messages, None, max_iterations + 1)
                final_answer = closing.content
                await repo.add_message(
                    self.cfg.conversation_id,
                    "assistant",
                    final_answer,
                    interaction_id=self.interaction_id,
                    thinking=closing.thinking,
                    usage=closing.usage.as_dict(),
                )
                yield self._event("status", stage="budget", message="Limite de iteraciones alcanzado")

        except Exception as exc:  # noqa: BLE001 - el error se reporta, no tumba el servidor
            logger.exception("Fallo en la interaccion %s", self.interaction_id)
            error_text = f"{type(exc).__name__}: {exc}"
            yield self._event("error", message=error_text)

        total_ms = (time.perf_counter() - started) * 1000
        metrics = self.metrics.as_mlflow(total_ms)

        await repo.finish_interaction(
            self.interaction_id,
            final_answer=final_answer,
            status="error" if error_text else "ok",
            error=error_text,
            ended_at=time.time(),
            latency_ms=total_ms,
            llm_latency_ms=self.metrics.llm_latency_ms,
            mcp_latency_ms=self.metrics.mcp_latency_ms,
            llm_calls=self.metrics.llm_calls,
            tool_calls=self.metrics.tool_calls,
            tool_errors=self.metrics.tool_errors,
            iterations=self.metrics.iterations,
            prompt_tokens=self.metrics.usage.prompt_tokens,
            completion_tokens=self.metrics.usage.completion_tokens,
            total_tokens=self.metrics.usage.total_tokens,
            mlflow_trace_id=self.trace.trace_id if self.trace else "",
            mlflow_run_id=self.trace.run_id if self.trace else "",
        )
        await repo.touch_session(self.cfg.session_id)

        if self.trace is not None:
            await tracker.end_interaction(
                self.trace,
                final_answer,
                metrics,
                artifact={
                    "session_id": self.cfg.session_id,
                    "conversation_id": self.cfg.conversation_id,
                    "interaction_id": self.interaction_id,
                    "experiment": self.trace.experiment,
                    "user_message": self.cfg.message,
                    "final_answer": final_answer,
                    "provider": provider_name,
                    "model": model_name,
                    "mcp_servers": server_summary,
                    "metrics": metrics,
                    "transcript": self.transcript,
                },
                error=error_text or None,
            )

        yield self._event(
            "final",
            content=final_answer,
            error=error_text,
            metrics=metrics,
            mlflow_trace_id=self.trace.trace_id if self.trace else "",
            mlflow_run_id=self.trace.run_id if self.trace else "",
            mlflow_experiment=self.trace.experiment if self.trace else "",
        )

    # ------------------------------------------------------------ llm span --
    async def _call_llm(
        self,
        provider: LLMProvider,
        messages: list[dict[str, Any]],
        router: ToolRouter | None,
        iteration: int,
    ) -> LLMResponse:
        tools = router.specs if router and not router.is_empty else None
        span = await tracker.start_span(
            self.trace,  # type: ignore[arg-type]
            name=f"llm_call_{iteration}",
            span_type="LLM",
            inputs={"messages": messages[-8:]},
            attributes={
                "iteration": iteration,
                "tools_offered": len(tools or []),
                # Claves estandar: MLflow las usa para su vista de LLM.
                ATTR_LLM_MODEL: provider.model,
                ATTR_LLM_PROVIDER: provider.name,
                ATTR_CHAT_TOOLS: [t.to_openai() for t in tools] if tools else [],
            },
        )
        try:
            response = await provider.chat(
                messages,
                tools,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
                thinking=self.cfg.thinking,
            )
        except Exception as exc:  # noqa: BLE001
            await tracker.end_span(span, error=f"{type(exc).__name__}: {exc}")
            raise

        self.metrics.llm_calls += 1
        self.metrics.llm_latency_ms += response.latency_ms
        self.metrics.usage = self.metrics.usage.merge(response.usage)

        await tracker.end_span(
            span,
            outputs={
                "content": response.content,
                "thinking": response.thinking,
                "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in response.tool_calls],
            },
            attributes={
                "latency_ms": response.latency_ms,
                "finish_reason": response.finish_reason,
                # MLflow agrega este atributo a nivel de traza automaticamente.
                ATTR_CHAT_USAGE: {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                **{f"timing.{k}": v for k, v in response.raw.items()},
            },
        )
        self.transcript.append(
            {
                "kind": "llm",
                "iteration": iteration,
                "content": response.content,
                "thinking": response.thinking,
                "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in response.tool_calls],
                "usage": response.usage.as_dict(),
                "latency_ms": response.latency_ms,
            }
        )
        return response

    # ---------------------------------------------------------- tool span ---
    async def _execute_tool(
        self,
        router: ToolRouter,
        call: Any,
        iteration: int,
        messages: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        self.tool_seq += 1
        origin = router.server_of(call.name)

        yield self._event(
            "tool_call",
            call_id=call.id,
            seq=self.tool_seq,
            iteration=iteration,
            tool=call.name,
            arguments=call.arguments,
            server=origin,
        )

        span = await tracker.start_span(
            self.trace,  # type: ignore[arg-type]
            name=f"mcp_tool::{call.name}",
            span_type="TOOL",
            inputs={"tool": call.name, "arguments": call.arguments},
            attributes={
                "iteration": iteration,
                "seq": self.tool_seq,
                "mcp_server_name": origin.get("server_name", ""),
                "mcp_server_url": origin.get("url", ""),
                "mcp_conn_id": origin.get("conn_id", ""),
                "mcp_real_tool": origin.get("real_tool", call.name),
            },
        )

        result = await router.call(call.name, call.arguments)

        self.metrics.tool_calls += 1
        self.metrics.mcp_latency_ms += result.latency_ms
        if not result.ok:
            self.metrics.tool_errors += 1

        observation = result.text if result.ok else f"ERROR: {result.error}"
        if result.structured is not None and not result.text:
            observation = json.dumps(result.structured, ensure_ascii=False)
        observation = _clip(observation or "(sin contenido)")

        # Auditoria completa en MLflow: ademas del span (que ya lleva el texto
        # integro), se sube un artifact por llamada con argumentos, resultado,
        # datos estructurados y los frames JSON-RPC en crudo. Asi el resultado
        # se puede recuperar entero aunque la UI muestre solo un extracto.
        artifact_file = (
            f"mcp_tool_calls/{self.interaction_id}/"
            f"{self.tool_seq:03d}_{_slug(call.name)}.json"
        )
        artifact_path = ""
        if self.trace is not None and self.trace.run_id:
            artifact_path = await tracker.log_artifact(
                self.trace.run_id,
                artifact_file,
                {
                    "session_id": self.cfg.session_id,
                    "conversation_id": self.cfg.conversation_id,
                    "interaction_id": self.interaction_id,
                    "seq": self.tool_seq,
                    "iteration": iteration,
                    "tool": call.name,
                    "real_tool": origin.get("real_tool", call.name),
                    "server": origin,
                    "arguments": call.arguments,
                    "ok": result.ok,
                    "error": result.error,
                    "latency_ms": result.latency_ms,
                    "result_chars": len(result.text or ""),
                    # Integro, sin recortar.
                    "result_text": result.text,
                    "structured": result.structured,
                    "jsonrpc_frames": result.frames,
                },
            )

        await tracker.end_span(
            span,
            outputs={
                "ok": result.ok,
                # Sin recortar: MLflow guarda los atributos de span completos.
                "text": result.text,
                "structured": result.structured,
                "error": result.error,
            },
            attributes={
                "latency_ms": result.latency_ms,
                "ok": result.ok,
                "jsonrpc_frames": len(result.frames),
                "result_chars": len(result.text or ""),
                # Ruta del artifact con el volcado integro, dentro del run de
                # la conversacion.
                "result_artifact": artifact_path,
            },
            error=result.error if not result.ok else None,
        )

        await repo.add_tool_event(
            interaction_id=self.interaction_id,
            conversation_id=self.cfg.conversation_id,
            session_id=self.cfg.session_id,
            seq=self.tool_seq,
            iteration=iteration,
            tool_name=call.name,
            real_tool_name=origin.get("real_tool", call.name),
            server_name=origin.get("server_name", ""),
            server_url=origin.get("url", ""),
            conn_id=origin.get("conn_id", ""),
            arguments=call.arguments,
            result_text=result.text,
            structured=result.structured,
            ok=result.ok,
            error=result.error or "",
            latency_ms=result.latency_ms,
            frames=result.frames,
        )

        if self.trace is not None:
            self.trace.tool_rows.append(
                {
                    "interaction_id": self.interaction_id,
                    "conversation_id": self.cfg.conversation_id,
                    "session_id": self.cfg.session_id,
                    "seq": self.tool_seq,
                    "iteration": iteration,
                    "tool": call.name,
                    "server": origin.get("server_name", ""),
                    "url": origin.get("url", ""),
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    "ok": result.ok,
                    "latency_ms": round(result.latency_ms, 2),
                    "error": result.error or "",
                    "result_chars": len(result.text or ""),
                    # La tabla es un indice legible; el resultado entero esta
                    # en el artifact al que apunta esta columna.
                    "result_artifact": artifact_path,
                    "result_preview": _clip(result.text, 500),
                }
            )

        self.transcript.append(
            {
                "kind": "tool",
                "iteration": iteration,
                "seq": self.tool_seq,
                "tool": call.name,
                "server": origin,
                "arguments": call.arguments,
                "ok": result.ok,
                # Integro: este transcript acaba en el artifact de la interaccion.
                "result": result.text,
                "result_artifact": artifact_path,
                "structured": result.structured,
                "error": result.error,
                "latency_ms": result.latency_ms,
                "jsonrpc_frames": result.frames,
            }
        )

        messages.append(
            {
                "role": "tool",
                "content": observation,
                "tool_call_id": call.id,
                "name": call.name,
            }
        )
        await repo.add_message(
            self.cfg.conversation_id,
            "tool",
            observation,
            interaction_id=self.interaction_id,
            tool_call_id=call.id,
            name=call.name,
        )

        yield self._event(
            "tool_result",
            call_id=call.id,
            seq=self.tool_seq,
            iteration=iteration,
            tool=call.name,
            ok=result.ok,
            # Solo la UI se recorta: el resultado completo esta en SQLite
            # (/api/traces/tool-events) y en MLflow.
            text=_clip(result.text, MAX_UI_RESULT_CHARS),
            truncated=len(result.text or "") > MAX_UI_RESULT_CHARS,
            result_chars=len(result.text or ""),
            structured=result.structured,
            error=result.error,
            latency_ms=result.latency_ms,
            server=origin,
            frames=result.frames,
            mlflow_artifact=artifact_path,
        )
