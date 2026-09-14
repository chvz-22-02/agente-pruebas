"""Prueba de humo contra el backend en marcha, con el LLM real.

Recorre el mismo camino que la UI: conectar MCP, mandar dos mensajes por SSE y
comprobar que el agente usa herramientas y mantiene la memoria.

    python tests\test_http_smoke.py [--backend http://127.0.0.1:8090] [--mcp http://127.0.0.1:3333/mcp]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="http://127.0.0.1:8090")
parser.add_argument("--mcp", default="http://127.0.0.1:3333/mcp")
args = parser.parse_args()


async def chat(client: httpx.AsyncClient, payload: dict) -> dict:
    """Consume el stream SSE y devuelve un resumen del turno."""
    summary = {"tools": [], "final": "", "metrics": {}, "errors": [], "ids": {}}
    started = time.perf_counter()
    async with client.stream("POST", f"{args.backend}/api/chat", json=payload, timeout=900) as response:
        response.raise_for_status()
        buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk.replace("\r\n", "\n")  # sse-starlette usa CRLF
            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                data = "\n".join(
                    line[5:].lstrip() for line in block.split("\n") if line.startswith("data:")
                )
                if not data:
                    continue
                event = json.loads(data)
                kind = event["type"]
                if kind == "start":
                    summary["ids"] = {
                        "session": event["session_id"],
                        "conversation": event["conversation_id"],
                        "interaction": event["interaction_id"],
                    }
                    print(f"    tools ofrecidas: {[t['name'] for t in event['tools']]}")
                elif kind == "thinking":
                    print(f"    [pensando] {event['text'][:90]}...")
                elif kind == "tool_call":
                    print(f"    -> {event['tool']}({json.dumps(event['arguments'], ensure_ascii=False)})")
                elif kind == "tool_result":
                    flag = "ok" if event["ok"] else "ERROR"
                    print(
                        f"    <- {flag} {round(event['latency_ms'])} ms, "
                        f"{len(event['frames'])} tramas: {(event['text'] or event['error'] or '')[:70]}"
                    )
                    summary["tools"].append(event["tool"])
                elif kind == "llm_usage":
                    print(f"    [llm] {event['usage']} en {round(event['latency_ms'])} ms")
                elif kind == "final":
                    summary["final"] = event["content"]
                    summary["metrics"] = event["metrics"]
                elif kind == "error":
                    summary["errors"].append(event["message"])
    summary["wall_s"] = round(time.perf_counter() - started, 1)
    return summary


async def main() -> int:
    async with httpx.AsyncClient() as client:
        health = (await client.get(f"{args.backend}/api/health")).json()
        print(f"backend ok | LLM {health['llm']['provider']} {health['llm'].get('model')} "
              f"disponible={health['llm'].get('model_available')} | MLflow={health['mlflow']['available']}")
        if not health["llm"]["ok"]:
            print("El motor LLM no responde. Arranca Ollama o revisa LLM_BASE_URL.")
            return 1

        conn = (await client.post(f"{args.backend}/api/mcp/connect", json={"url": args.mcp})).json()
        conn_id = conn["conn_id"]
        print(f"MCP: {conn['server_info']['name']} via {conn['transport']}, "
              f"{len(conn['tools'])} herramientas\n")

        session = (await client.post(f"{args.backend}/api/sessions", json={"title": "smoke http"})).json()

        print("== turno 1: debe usar una herramienta ==")
        first = await chat(client, {
            "message": "Cuantas unidades quedan del SKU-002 y cuanto costarian 3 unidades?",
            "session_id": session["id"],
            "mcp_conn_ids": [conn_id],
        })
        print(f"  respuesta ({first['wall_s']}s): {first['final'][:220]}\n")

        assert not first["errors"], first["errors"]
        assert first["tools"], "el agente no llamo a ninguna herramienta"
        assert first["final"], "no hubo respuesta final"

        print("== turno 2: debe recordar el contexto ==")
        second = await chat(client, {
            "message": "Y sin llamar a ninguna herramienta: que SKU acabo de preguntarte?",
            "session_id": session["id"],
            "conversation_id": first["ids"]["conversation"],
            "mcp_conn_ids": [conn_id],
        })
        print(f"  respuesta ({second['wall_s']}s): {second['final'][:220]}\n")
        assert "002" in second["final"], f"parece que perdio la memoria: {second['final']}"

        stats = (await client.get(f"{args.backend}/api/sessions/{session['id']}/stats")).json()
        print(f"stats sesion: {stats['interactions']} interacciones, {stats['total_tokens']} tokens, "
              f"{stats['tool_calls']} llamadas MCP")

        detail = (await client.get(
            f"{args.backend}/api/traces/interactions/{first['ids']['interaction']}"
        )).json()
        frames = sum(len(e["frames"]) for e in detail["tool_events"])
        print(f"traza persistida: {len(detail['tool_events'])} eventos MCP, {frames} tramas JSON-RPC")
        assert frames > 0, "no se persistieron las tramas JSON-RPC"

        print("\nTODO OK")
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
