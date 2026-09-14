import { useCallback, useEffect, useState } from "react";
import Json from "./Json";
import { api, getBackendUrl } from "../lib/api";
import type { Interaction, ToolEvent } from "../lib/types";

type Props = {
  sessionId: string | null;
  conversationId: string | null;
  refreshKey: number;
  mlflowUrl: string;
};

type Scope = "conversacion" | "sesion";

const ms = (v: number) => `${Math.round(v)} ms`;
const when = (ts: number) => new Date(ts * 1000).toLocaleTimeString();

function Detail({ interactionId }: { interactionId: string }) {
  const [data, setData] = useState<{ interaction: Interaction; tool_events: ToolEvent[]; mlflow: any } | null>(null);

  useEffect(() => {
    api.get(`/api/traces/interactions/${interactionId}`).then(setData as any).catch(() => setData(null));
  }, [interactionId]);

  if (!data) return <p className="muted">Cargando traza...</p>;
  const i = data.interaction;

  return (
    <div style={{ marginTop: 8 }}>
      <div className="stat-grid">
        <div className="stat">
          <div className="label">Latencia total</div>
          <div className="value">{ms(i.latency_ms)}</div>
        </div>
        <div className="stat">
          <div className="label">Tokens</div>
          <div className="value">{i.total_tokens}</div>
        </div>
        <div className="stat">
          <div className="label">LLM / MCP</div>
          <div className="value" style={{ fontSize: 14 }}>
            {ms(i.llm_latency_ms)} / {ms(i.mcp_latency_ms)}
          </div>
        </div>
        <div className="stat">
          <div className="label">Llamadas</div>
          <div className="value" style={{ fontSize: 14 }}>
            {i.llm_calls} LLM · {i.tool_calls} tools
          </div>
        </div>
      </div>

      <dl className="kv">
        <dt>Interaccion</dt>
        <dd>{i.id}</dd>
        <dt>Conversacion</dt>
        <dd>{i.conversation_id}</dd>
        <dt>Sesion</dt>
        <dd>{i.session_id}</dd>
        <dt>Modelo</dt>
        <dd>
          {i.provider} / {i.model}
        </dd>
        <dt>Prompt / salida</dt>
        <dd>
          {i.prompt_tokens} / {i.completion_tokens}
        </dd>
        <dt>Iteraciones</dt>
        <dd>{i.iterations}</dd>
        <dt>Errores tool</dt>
        <dd style={{ color: i.tool_errors ? "var(--err)" : undefined }}>{i.tool_errors}</dd>
        {i.mlflow_trace_id && (
          <>
            <dt>Trace MLflow</dt>
            <dd>{i.mlflow_trace_id}</dd>
          </>
        )}
        {data.mlflow?.experiment && (
          <>
            <dt>Experimento</dt>
            <dd>
              {data.mlflow.experiment_url ? (
                <a
                  href={data.mlflow.experiment_url}
                  target="_blank"
                  rel="noreferrer"
                  style={{ color: "var(--accent)" }}
                >
                  {data.mlflow.experiment}
                </a>
              ) : (
                data.mlflow.experiment
              )}
            </dd>
          </>
        )}
      </dl>

      {i.error && <div className="error-box" style={{ marginTop: 8 }}>{i.error}</div>}

      <hr className="sep" />
      <div className="muted" style={{ marginBottom: 6 }}>
        Llamadas al MCP ({data.tool_events.length})
      </div>

      {data.tool_events.map((event) => (
        <details key={event.id} className={`block tool ${event.ok ? "" : "err"}`}>
          <summary>
            <span className={`dot ${event.ok ? "ok" : "err"}`} />
            <span className="name">#{event.seq} {event.tool_name}</span>
            <div style={{ flex: 1 }} />
            <span className="muted">{ms(event.latency_ms)}</span>
          </summary>
          <div className="inner">
            <dl className="kv">
              <dt>Servidor</dt>
              <dd>{event.server_name || "-"}</dd>
              <dt>URL</dt>
              <dd>{event.server_url || "-"}</dd>
              <dt>Iteracion</dt>
              <dd>{event.iteration}</dd>
            </dl>
            <div className="muted" style={{ marginTop: 8 }}>Argumentos</div>
            <Json value={event.arguments} />
            {event.error && (
              <>
                <div className="muted" style={{ marginTop: 8, color: "var(--err)" }}>Error</div>
                <pre style={{ color: "#ffb3ae" }}>{event.error}</pre>
              </>
            )}
            <div className="muted" style={{ marginTop: 8 }}>
              Resultado completo ({(event.result_text || "").length.toLocaleString()} caracteres)
            </div>
            <pre>{event.result_text || "(vacio)"}</pre>
            <div className="muted">
              En MLflow:{" "}
              <code>
                {data.mlflow?.tool_artifacts || `mcp_tool_calls/${event.interaction_id}/`}
                {String(event.seq).padStart(3, "0")}_{event.tool_name}.json
              </code>
            </div>
            {event.frames?.length > 0 && (
              <details style={{ marginTop: 8 }}>
                <summary className="muted">JSON-RPC ({event.frames.length} tramas)</summary>
                {event.frames.map((f, idx) => (
                  <div key={idx} className={`frame ${f.direction === "agent->mcp" ? "out" : "in"}`}>
                    <div className="dir">{f.direction}</div>
                    <Json value={f.payload} max={6000} />
                  </div>
                ))}
              </details>
            )}
          </div>
        </details>
      ))}
    </div>
  );
}

/** Historico persistido de interacciones, con el detalle de cada llamada al MCP. */
export default function TracesPanel({ sessionId, conversationId, refreshKey, mlflowUrl }: Props) {
  const [scope, setScope] = useState<Scope>("conversacion");
  const [items, setItems] = useState<Interaction[]>([]);
  const [openId, setOpenId] = useState<string | null>(null);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setError("");
    const query =
      scope === "conversacion"
        ? conversationId && `conversation_id=${conversationId}`
        : sessionId && `session_id=${sessionId}`;
    if (!query) {
      setItems([]);
      return;
    }
    try {
      const data = await api.get<{ interactions: Interaction[] }>(`/api/traces/interactions?${query}`);
      setItems(data.interactions);
    } catch (e: any) {
      setError(e.message);
    }
  }, [scope, sessionId, conversationId]);

  useEffect(() => {
    load();
  }, [load, refreshKey]);

  return (
    <div className="panel-body">
      <div className="row" style={{ marginBottom: 10 }}>
        <select value={scope} onChange={(e) => setScope(e.target.value as Scope)} style={{ flex: 1 }}>
          <option value="conversacion">Esta conversacion</option>
          <option value="sesion">Toda la sesion</option>
        </select>
        <button className="tiny" onClick={load}>
          Recargar
        </button>
        {sessionId && (
          <a
            className="badge"
            href={`${getBackendUrl()}/api/traces/export?session_id=${sessionId}`}
            target="_blank"
            rel="noreferrer"
            style={{ padding: "6px 9px", textDecoration: "none" }}
          >
            Exportar
          </a>
        )}
      </div>

      {mlflowUrl && (
        <a href={mlflowUrl} target="_blank" rel="noreferrer" className="muted" style={{ color: "var(--accent)" }}>
          Abrir el experimento en MLflow
        </a>
      )}

      {error && <div className="error-box" style={{ marginTop: 10 }}>{error}</div>}

      <hr className="sep" />

      {items.length === 0 && <p className="muted">Sin interacciones registradas.</p>}

      {items.map((item) => (
        <div
          key={item.id}
          className={`list-item ${openId === item.id ? "active" : ""}`}
          style={{ cursor: "default" }}
        >
          <div className="row" onClick={() => setOpenId(openId === item.id ? null : item.id)} style={{ cursor: "pointer" }}>
            <span className={`dot ${item.status === "ok" ? "ok" : "err"}`} />
            <div className="title" style={{ flex: 1 }}>
              {item.user_message}
            </div>
            <span className="muted">{when(item.started_at)}</span>
          </div>
          <div className="meta">
            {ms(item.latency_ms)} · {item.total_tokens} tok · {item.tool_calls} tools
            {item.tool_errors > 0 && <span style={{ color: "var(--err)" }}> · {item.tool_errors} err</span>}
          </div>
          {openId === item.id && <Detail interactionId={item.id} />}
        </div>
      ))}
    </div>
  );
}
