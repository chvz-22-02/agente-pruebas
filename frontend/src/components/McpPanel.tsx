import { useState } from "react";
import Json from "./Json";
import { api } from "../lib/api";
import type { Frame, McpConnection } from "../lib/types";

type Props = {
  connections: McpConnection[];
  selected: string[];
  onToggle: (connId: string) => void;
  onConnected: () => void;
  onDisconnect: (connId: string) => void;
};

/** Convierte "Authorization: Bearer x" (una por linea) en un objeto de cabeceras. */
function parseHeaders(raw: string): Record<string, string> {
  const headers: Record<string, string> = {};
  for (const line of raw.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const index = trimmed.indexOf(":");
    if (index <= 0) continue;
    headers[trimmed.slice(0, index).trim()] = trimmed.slice(index + 1).trim();
  }
  return headers;
}

function ConnectionCard({
  conn,
  selected,
  onToggle,
  onDisconnect,
  onRefresh,
}: {
  conn: McpConnection;
  selected: boolean;
  onToggle: () => void;
  onDisconnect: () => void;
  onRefresh: () => void;
}) {
  const [tool, setTool] = useState(conn.tools[0]?.name || "");
  const [args, setArgs] = useState("{}");
  const [result, setResult] = useState<any>(null);
  const [running, setRunning] = useState(false);
  const [frames, setFrames] = useState<Frame[] | null>(null);

  const runTool = async () => {
    setRunning(true);
    setResult(null);
    try {
      const parsed = JSON.parse(args || "{}");
      setResult(await api.post(`/api/mcp/connections/${conn.conn_id}/call`, { tool, arguments: parsed }));
    } catch (e: any) {
      setResult({ ok: false, error: e.message });
    } finally {
      setRunning(false);
    }
  };

  const loadFrames = async () => {
    const data = await api.get<{ frames: Frame[] }>(`/api/mcp/connections/${conn.conn_id}/frames?limit=60`);
    setFrames(data.frames);
  };

  const schema = conn.tools.find((t) => t.name === tool)?.input_schema;

  return (
    <div style={{ border: "1px solid var(--line)", borderRadius: 8, padding: 11, marginBottom: 11 }}>
      <div className="row">
        <input type="checkbox" checked={selected} onChange={onToggle} style={{ width: 14 }} title="Usar en el agente" />
        <span className={`dot ${conn.alive ? "ok" : "err"}`} />
        <strong style={{ flex: 1, fontSize: 13 }}>{conn.server_info?.name || conn.conn_id}</strong>
        <button className="tiny" onClick={onRefresh} title="Releer herramientas">
          Refrescar
        </button>
        <button className="tiny danger" onClick={onDisconnect}>
          Cerrar
        </button>
      </div>

      <dl className="kv" style={{ marginTop: 8 }}>
        <dt>URL</dt>
        <dd>{conn.config.url}</dd>
        <dt>Transporte</dt>
        <dd>{conn.transport}</dd>
        <dt>Protocolo</dt>
        <dd>{conn.server_info?.protocol_version || "-"}</dd>
        <dt>Version</dt>
        <dd>{conn.server_info?.version || "-"}</dd>
      </dl>

      {conn.server_info?.instructions && (
        <details style={{ marginTop: 8 }}>
          <summary className="muted">Instrucciones del servidor</summary>
          <pre>{conn.server_info.instructions}</pre>
        </details>
      )}

      <details style={{ marginTop: 8 }} open>
        <summary className="muted">Herramientas ({conn.tools.length})</summary>
        <table className="data" style={{ marginTop: 6 }}>
          <tbody>
            {conn.tools.map((t) => (
              <tr key={t.name}>
                <td style={{ color: "var(--tool)", whiteSpace: "nowrap" }}>{t.name}</td>
                <td style={{ fontFamily: "inherit", color: "var(--fg-dim)" }}>{t.description || "-"}</td>
              </tr>
            ))}
            {conn.tools.length === 0 && (
              <tr>
                <td colSpan={2} className="muted">
                  El servidor no expone herramientas.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </details>

      <details style={{ marginTop: 8 }}>
        <summary className="muted">Probar una herramienta sin el LLM</summary>
        <div style={{ marginTop: 8 }}>
          <select value={tool} onChange={(e) => setTool(e.target.value)}>
            {conn.tools.map((t) => (
              <option key={t.name} value={t.name}>
                {t.name}
              </option>
            ))}
          </select>
          {schema && (
            <details style={{ marginTop: 6 }}>
              <summary className="muted">Esquema de entrada</summary>
              <Json value={schema} />
            </details>
          )}
          <textarea
            value={args}
            onChange={(e) => setArgs(e.target.value)}
            rows={3}
            spellCheck={false}
            style={{ marginTop: 6, fontFamily: "var(--mono)", fontSize: 12 }}
          />
          <button className="tiny" onClick={runTool} disabled={running || !tool} style={{ marginTop: 6 }}>
            {running ? "Ejecutando..." : "Ejecutar"}
          </button>
          {result && (
            <>
              <div className="muted" style={{ marginTop: 8 }}>
                {result.ok === false ? "Error" : `OK · ${Math.round(result.latency_ms || 0)} ms`}
              </div>
              <Json value={result.ok === false ? result.error : result.text || result.structured} />
              {result.frames?.length > 0 && (
                <details style={{ marginTop: 6 }}>
                  <summary className="muted">JSON-RPC ({result.frames.length})</summary>
                  <Json value={result.frames} />
                </details>
              )}
            </>
          )}
        </div>
      </details>

      <details style={{ marginTop: 8 }} onToggle={(e) => (e.currentTarget as HTMLDetailsElement).open && loadFrames()}>
        <summary className="muted">Trafico reciente del socket</summary>
        {frames === null ? <p className="muted">Cargando...</p> : <Json value={frames} />}
      </details>
    </div>
  );
}

export default function McpPanel({ connections, selected, onToggle, onConnected, onDisconnect }: Props) {
  const [url, setUrl] = useState("http://127.0.0.1:3333/mcp");
  const [name, setName] = useState("");
  const [transport, setTransport] = useState("auto");
  const [headers, setHeaders] = useState("");
  const [connecting, setConnecting] = useState(false);
  const [error, setError] = useState("");

  const connect = async () => {
    setConnecting(true);
    setError("");
    try {
      await api.post("/api/mcp/connect", {
        url: url.trim(),
        name: name.trim(),
        transport,
        headers: parseHeaders(headers),
        timeout_s: 60,
        capture_raw: true,
      });
      onConnected();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setConnecting(false);
    }
  };

  return (
    <div className="panel-body">
      <label className="field">
        <span>Enlace del servidor MCP</span>
        <input
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && connect()}
          placeholder="http://host:puerto/mcp"
          spellCheck={false}
        />
      </label>

      <div className="row" style={{ marginBottom: 10 }}>
        <label className="field" style={{ flex: 1, marginBottom: 0 }}>
          <span>Alias (opcional)</span>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="mi-mcp" />
        </label>
        <label className="field" style={{ width: 140, marginBottom: 0 }}>
          <span>Transporte</span>
          <select value={transport} onChange={(e) => setTransport(e.target.value)}>
            <option value="auto">auto</option>
            <option value="streamable_http">streamable http</option>
            <option value="sse">sse</option>
          </select>
        </label>
      </div>

      <details style={{ marginBottom: 10 }}>
        <summary className="muted">Cabeceras (autenticacion)</summary>
        <textarea
          value={headers}
          onChange={(e) => setHeaders(e.target.value)}
          rows={2}
          placeholder="Authorization: Bearer eyJ..."
          spellCheck={false}
          style={{ marginTop: 6, fontFamily: "var(--mono)", fontSize: 12 }}
        />
      </details>

      <button className="primary" onClick={connect} disabled={connecting || !url.trim()} style={{ width: "100%" }}>
        {connecting ? "Conectando..." : "Conectar"}
      </button>

      {error && (
        <div className="error-box" style={{ marginTop: 10 }}>
          {error}
        </div>
      )}

      <hr className="sep" />

      <div className="muted" style={{ marginBottom: 8 }}>
        Marca la casilla de los servidores que el agente puede usar.
      </div>

      {connections.length === 0 && <p className="muted">Ningun servidor conectado.</p>}
      {connections.map((conn) => (
        <ConnectionCard
          key={conn.conn_id}
          conn={conn}
          selected={selected.includes(conn.conn_id)}
          onToggle={() => onToggle(conn.conn_id)}
          onDisconnect={() => onDisconnect(conn.conn_id)}
          onRefresh={async () => {
            await api.post(`/api/mcp/connections/${conn.conn_id}/refresh`);
            onConnected();
          }}
        />
      ))}
    </div>
  );
}
