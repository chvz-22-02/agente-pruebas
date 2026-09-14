"""Prueba de extremo a extremo del bucle del agente con un LLM simulado.

Comprueba, sin depender de que haya un modelo descargado:
  * el agente encadena LLM -> herramienta MCP -> LLM,
  * la memoria de la conversacion persiste entre mensajes,
  * SQLite guarda interacciones, eventos de herramienta y tramas JSON-RPC,
  * MLflow recibe la traza con sus metricas.

Requiere el servidor MCP de ejemplo:
    python scripts/demo_mcp_server.py --port 3333
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.loop import AgentRunner, RunConfig  # noqa: E402
from app.llm import registry  # noqa: E402
from app.llm.base import LLMProvider, LLMResponse, ToolCall, ToolSpec, Usage  # noqa: E402
from app.mcpclient.manager import mcp_manager  # noqa: E402
from app.mcpclient.models import MCPServerConfig  # noqa: E402
from app.observability.mlflow_tracker import tracker  # noqa: E402
from app.store import repository as repo  # noqa: E402
from app.store.db import db  # noqa: E402

MCP_URL = "http://127.0.0.1:3333/mcp"


class ScriptedProvider(LLMProvider):
    """LLM de mentira: primero pide una herramienta, luego responde."""

    name = "scripted"

    def __init__(self, base_url: str = "http://scripted", model: str = "modelo-de-prueba", **options: Any) -> None:
        super().__init__(base_url, model, **options)
        self.calls = 0
        self.seen_tools: list[str] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls += 1
        self.seen_tools = [t.name for t in (tools or [])]
        if self.calls == 1:
            return LLMResponse(
                content="",
                thinking="Necesito consultar el inventario antes de responder.",
                tool_calls=[ToolCall(name="consultar_inventario", arguments={"sku": "SKU-002"})],
                usage=Usage(120, 30, 150),
                latency_ms=42.0,
                model=self.model,
            )
        observation = next(m for m in reversed(messages) if m["role"] == "tool")["content"]
        return LLMResponse(
            content=f"El monitor tiene stock. Datos crudos: {observation[:60]}",
            usage=Usage(240, 45, 285),
            latency_ms=61.0,
            model=self.model,
        )

    async def list_models(self) -> list[str]:
        return [self.model]

    async def health(self) -> dict[str, Any]:
        return {"ok": True, "provider": self.name}


async def main() -> int:
    # Cierre garantizado: si una asercion falla, la task que sostiene la
    # conexion MCP mantendria vivo el bucle de eventos y el proceso quedaria
    # colgado en vez de mostrar el fallo.
    try:
        return await _run()
    finally:
        await mcp_manager.disconnect_all()
        await db.close()


async def _run() -> int:
    await db.connect()
    await tracker.init()
    print(f"MLflow: {tracker.status}")

    # Se registra como un proveedor mas y se deja que el registro lo construya:
    # asi la prueba no depende de la forma interna de la cache.
    registry.PROVIDERS["scripted"] = ScriptedProvider  # type: ignore[assignment]
    provider = await registry.get_provider("scripted", "http://scripted", "modelo-de-prueba")
    assert isinstance(provider, ScriptedProvider)

    conn = await mcp_manager.connect(MCPServerConfig(url=MCP_URL, name="demo"))
    conn_id = conn["conn_id"]
    print(f"MCP conectado: {conn['server_info']['name']} con {len(conn['tools'])} herramientas")

    session_id = (await repo.create_session("test automatico"))["id"]
    conversation = await repo.create_conversation(session_id, title="hilo de prueba")
    conversation_id = conversation["id"]

    def run(message: str) -> RunConfig:
        return RunConfig(
            session_id=session_id,
            conversation_id=conversation_id,
            message=message,
            mcp_conn_ids=[conn_id],
            provider="scripted",
            base_url="http://scripted",
            model="modelo-de-prueba",
        )

    print("\n--- turno 1 ---")
    events: list[dict] = []
    async for event in AgentRunner(run("Cuanto stock hay del SKU-002?")).run():
        events.append(event)
        print(f"  {event['type']:18} {str(event.get('message') or event.get('tool') or event.get('content', ''))[:70]}")

    kinds = [e["type"] for e in events]
    assert "tool_call" in kinds, "el agente no llamo a ninguna herramienta"
    assert "tool_result" in kinds
    assert kinds[-1] == "final"

    tool_result = next(e for e in events if e["type"] == "tool_result")
    assert tool_result["ok"], f"la herramienta fallo: {tool_result.get('error')}"
    assert len(tool_result["frames"]) >= 2, "no se capturaron las tramas JSON-RPC"
    print(f"  -> {len(tool_result['frames'])} tramas JSON-RPC capturadas")

    final = events[-1]
    metrics = final["metrics"]
    assert metrics["total_tokens"] == 435, metrics
    assert metrics["llm_calls"] == 2 and metrics["tool_calls"] == 1
    print(f"  -> metricas: {metrics['total_tokens']} tokens, {metrics['tool_calls']} tool calls")

    # --- memoria entre mensajes -------------------------------------------
    print("\n--- turno 2 (comprueba la memoria) ---")
    provider.calls = 1  # fuerza la rama de respuesta directa
    async for event in AgentRunner(run("Y del SKU-001?")).run():
        if event["type"] == "final":
            print(f"  respuesta: {event['content'][:70]}")

    messages = await repo.get_messages(conversation_id)
    roles = [m["role"] for m in messages]
    assert roles.count("user") == 2, roles
    assert "tool" in roles, roles
    print(f"  -> {len(messages)} mensajes en memoria: {roles}")

    # --- persistencia ------------------------------------------------------
    interactions = await repo.list_interactions(session_id=session_id)
    assert len(interactions) == 2, interactions
    tool_events = await repo.list_tool_events(session_id=session_id)
    assert tool_events and tool_events[0]["frames"], "no se persistieron las tramas"
    stats = await repo.session_stats(session_id)
    print(f"  -> SQLite: {len(interactions)} interacciones, {len(tool_events)} llamadas MCP")
    print(f"  -> stats sesion: {stats['total_tokens']} tokens, {stats['tool_calls']} tools")

    # --- MLflow ------------------------------------------------------------
    if tracker.available:
        trace_id = interactions[-1]["mlflow_trace_id"]
        run_id = interactions[-1]["mlflow_run_id"]
        assert run_id, "no se registro el run de conversacion"
        print(f"  -> MLflow run={run_id} trace={trace_id or '(sin trace)'}")
        if trace_id:
            import mlflow

            await asyncio.sleep(2)  # el exportador de traces es asincrono
            trace = mlflow.get_trace(trace_id)
            spans = [s.name for s in trace.data.spans]
            print(f"  -> spans en MLflow: {spans}")
            assert any(s.startswith("mcp_tool::") for s in spans), spans
            assert any(s.startswith("llm_call") for s in spans), spans
            usage = trace.info.token_usage
            print(f"  -> token usage agregado por MLflow: {usage}")

    print("\nTODO OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
