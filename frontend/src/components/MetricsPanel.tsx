import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/api";

type Props = {
  sessionId: string | null;
  refreshKey: number;
  mlflow: any;
  mlflowUrl: string;
  /** Experimento activo, el elegido en la barra lateral. */
  experiment: string;
};

type Stats = {
  session_id: string;
  conversations: number;
  interactions: number;
  total_tokens: number;
  prompt_tokens: number;
  completion_tokens: number;
  tool_calls: number;
  tool_errors: number;
  avg_latency_ms: number;
  max_latency_ms: number;
  per_tool: {
    tool_name: string;
    server_name: string;
    calls: number;
    errors: number;
    avg_latency_ms: number;
  }[];
};

/** Agregados de la sesion actual: lo mismo que se registra en MLflow. */
export default function MetricsPanel({ sessionId, refreshKey, mlflow, mlflowUrl, experiment }: Props) {
  const [stats, setStats] = useState<Stats | null>(null);

  const load = useCallback(async () => {
    if (!sessionId) return setStats(null);
    try {
      setStats(await api.get<Stats>(`/api/sessions/${sessionId}/stats`));
    } catch {
      setStats(null);
    }
  }, [sessionId]);

  useEffect(() => {
    load();
  }, [load, refreshKey]);

  return (
    <div className="panel-body">
      <div className="panel-head" style={{ padding: 0, border: "none", marginBottom: 10 }}>
        Sesion actual
        <div style={{ flex: 1 }} />
        <button className="tiny" onClick={load}>
          Recargar
        </button>
      </div>

      {!stats && <p className="muted">Sin datos todavia.</p>}

      {stats && (
        <>
          <div className="stat-grid">
            <div className="stat">
              <div className="label">Conversaciones</div>
              <div className="value">{stats.conversations}</div>
            </div>
            <div className="stat">
              <div className="label">Interacciones</div>
              <div className="value">{stats.interactions}</div>
            </div>
            <div className="stat">
              <div className="label">Tokens totales</div>
              <div className="value">{stats.total_tokens.toLocaleString()}</div>
            </div>
            <div className="stat">
              <div className="label">Llamadas MCP</div>
              <div className="value">
                {stats.tool_calls}
                {stats.tool_errors > 0 && (
                  <span style={{ color: "var(--err)", fontSize: 13 }}> / {stats.tool_errors} err</span>
                )}
              </div>
            </div>
            <div className="stat">
              <div className="label">Latencia media</div>
              <div className="value">{Math.round(stats.avg_latency_ms)} ms</div>
            </div>
            <div className="stat">
              <div className="label">Latencia maxima</div>
              <div className="value">{Math.round(stats.max_latency_ms)} ms</div>
            </div>
            <div className="stat">
              <div className="label">Tokens entrada</div>
              <div className="value" style={{ fontSize: 15 }}>{stats.prompt_tokens.toLocaleString()}</div>
            </div>
            <div className="stat">
              <div className="label">Tokens salida</div>
              <div className="value" style={{ fontSize: 15 }}>{stats.completion_tokens.toLocaleString()}</div>
            </div>
          </div>

          <div className="muted" style={{ marginBottom: 6 }}>Uso por herramienta</div>
          <table className="data">
            <thead>
              <tr>
                <th>Herramienta</th>
                <th>Servidor</th>
                <th>Llamadas</th>
                <th>Err</th>
                <th>Media</th>
              </tr>
            </thead>
            <tbody>
              {stats.per_tool.map((row) => (
                <tr key={`${row.server_name}:${row.tool_name}`}>
                  <td style={{ color: "var(--tool)" }}>{row.tool_name}</td>
                  <td>{row.server_name || "-"}</td>
                  <td>{row.calls}</td>
                  <td style={{ color: row.errors ? "var(--err)" : undefined }}>{row.errors}</td>
                  <td>{Math.round(row.avg_latency_ms)} ms</td>
                </tr>
              ))}
              {stats.per_tool.length === 0 && (
                <tr>
                  <td colSpan={5} className="muted">Todavia no se ha llamado a ninguna herramienta.</td>
                </tr>
              )}
            </tbody>
          </table>
        </>
      )}

      <hr className="sep" />

      <div className="muted" style={{ marginBottom: 6 }}>MLflow</div>
      <dl className="kv">
        <dt>Estado</dt>
        <dd style={{ color: mlflow?.available ? "var(--ok)" : "var(--warn)" }}>
          {mlflow?.available ? "activo" : "no disponible"}
        </dd>
        <dt>Experimento</dt>
        <dd>{experiment || mlflow?.experiment || "-"}</dd>
        <dt>Traces</dt>
        <dd>{mlflow?.tracing_supported ? "si" : "no"}</dd>
      </dl>
      <p className="muted" style={{ marginTop: 6 }}>{mlflow?.status}</p>
      {mlflowUrl && (
        <a href={mlflowUrl} target="_blank" rel="noreferrer" style={{ color: "var(--accent)", fontSize: 12 }}>
          Abrir experimento
        </a>
      )}
      <p className="muted" style={{ marginTop: 10 }}>
        En MLflow, filtra las trazas de una sesion con{" "}
        <code>tags.session_id = '{sessionId || "ses_..."}'</code>.
      </p>
      <p className="muted" style={{ marginTop: 6 }}>
        Cada llamada MCP se guarda entera como artifact del run de su conversacion, en{" "}
        <code>mcp_tool_calls/&lt;interaccion&gt;/</code>, ademas de en el span de la traza. Lo
        que la UI recorta se recupera ahi.
      </p>
    </div>
  );
}
