import Json from "../Json";
import { END_REASON_LABEL, FINISHED, PHASE_LABEL, STATUS_LABEL, summarize } from "../../lib/evalState";
import type { EvalItem, EvalResultView, EvalRunDetail, EvalToolView } from "../../lib/types";

type Props = {
  detail: EvalRunDetail | null;
  results: EvalResultView[];
  live: boolean;
  logs: { level: string; message: string }[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onCancel: () => void;
  onDelete: () => void;
  onOpenInChat: (sessionId: string, conversationId: string) => void;
};

const pct = (v: number | null | undefined) => (v === null || v === undefined ? "-" : `${Math.round(v * 100)}%`);
/** Latencia legible: en CPU un turno se mide en segundos, no en milisegundos. */
const ms = (v?: number) =>
  v === undefined || v === null ? "-" : v >= 10000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`;
const secs = (v?: number) => (v ? `${(v / 1000).toFixed(1)} s` : "-");
const num = (v?: number) => (v ? Math.round(v).toLocaleString() : "0");

function StatusBadge({ status }: { status: string }) {
  return <span className={`badge status-${status}`}>{STATUS_LABEL[status] || status}</span>;
}

function ToolCall({ call }: { call: EvalToolView }) {
  const running = call.ok === undefined;
  return (
    <details className={`block tool ${call.ok === false ? "err" : ""}`}>
      <summary>
        {running ? <span className="spin" /> : <span className={`dot ${call.ok ? "ok" : "err"}`} />}
        <span className="name">{call.tool}</span>
        <div style={{ flex: 1 }} />
        <span className="muted">{running ? "ejecutando..." : ms(call.latency_ms)}</span>
      </summary>
      <div className="inner">
        <div className="muted">Argumentos</div>
        <Json value={call.arguments} />
        {call.error && <pre style={{ color: "#ffb3ae" }}>{call.error}</pre>}
        {call.preview !== undefined && (
          <>
            <div className="muted" style={{ marginTop: 8 }}>
              Resultado (extracto{call.result_chars ? ` de ${call.result_chars.toLocaleString()} caracteres` : ""})
            </div>
            <pre>{call.preview || "(vacio)"}</pre>
          </>
        )}
      </div>
    </details>
  );
}

function Criteria({ items, failedMandatory }: { items: EvalItem[]; failedMandatory: string[] }) {
  return (
    <table className="data eval-criteria">
      <thead>
        <tr>
          <th />
          <th>Criterio</th>
          <th>Origen</th>
          <th>Peso</th>
          <th>Justificacion</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.id} className={item.cumple ? "" : "miss"}>
            <td>
              <span className={`dot ${item.cumple ? "ok" : "err"}`} />
            </td>
            <td>
              <div>
                <code>{item.id}</code>
                {item.obligatorio && (
                  <span className={`badge ${failedMandatory.includes(item.id) ? "err" : ""}`} style={{ marginLeft: 6 }}>
                    obligatorio
                  </span>
                )}
              </div>
              <div className="muted">{item.descripcion}</div>
            </td>
            <td>
              <span className={`badge ${item.origen === "evaluador" ? "tool" : ""}`}>
                {item.origen === "evaluador" ? "LLM" : "codigo"}
              </span>
            </td>
            <td>{item.peso}</td>
            <td>
              {item.detalle}
              {item.valor_reportado && <div className="muted">reportado: {item.valor_reportado}</div>}
              {item.sin_juicio && <div style={{ color: "var(--warn)", fontSize: 11 }}>el evaluador no se pronuncio</div>}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ResultDetail({
  result,
  sessionId,
  experimentUrl,
  onOpenInChat,
}: {
  result: EvalResultView;
  sessionId: string;
  experimentUrl: string;
  onOpenInChat: (sessionId: string, conversationId: string) => void;
}) {
  const finished = FINISHED.includes(result.status);
  const runUrl = experimentUrl && result.mlflow_run_id ? `${experimentUrl}/runs/${result.mlflow_run_id}` : "";

  return (
    <div className="eval-detail">
      <div className="row" style={{ marginBottom: 8, flexWrap: "wrap" }}>
        <strong>{result.case_title}</strong>
        <span className="muted">con {result.persona_name}</span>
        {result.repetition > 1 && <span className="badge">#{result.repetition}</span>}
        <StatusBadge status={result.status} />
        {result.score !== null && <span className="badge">nota {pct(result.score)}</span>}
        <div style={{ flex: 1 }} />
        {result.conversation_id && finished && (
          <button
            className="tiny"
            onClick={() => onOpenInChat(sessionId, result.conversation_id!)}
            title="Abre esta conversacion en el chat, con el razonamiento y las tramas JSON-RPC"
          >
            Ver en el chat
          </button>
        )}
        {runUrl && (
          <a className="badge" href={runUrl} target="_blank" rel="noreferrer" style={{ padding: "4px 8px" }}>
            Run en MLflow
          </a>
        )}
      </div>

      {result.status === "running" && (
        <p className="muted">
          <span className="spin" style={{ display: "inline-block", marginRight: 6 }} />
          Turno {result.turn ?? 1}: {PHASE_LABEL[result.phase || ""] || "en curso"}...
        </p>
      )}
      {result.error && <div className="error-box">{result.error}</div>}

      {finished && result.resumen && (
        <div className="eval-verdict">
          <div className="muted" style={{ marginBottom: 2 }}>Veredicto del evaluador</div>
          {result.resumen}
        </div>
      )}

      {finished && result.metrics && Object.keys(result.metrics).length > 0 && (
        <div className="metrics-line">
          <span>{result.turns} turno(s) · {END_REASON_LABEL[result.end_reason] || result.end_reason}</span>
          <span>agente {num(result.metrics.agent_total_tokens)} tok</span>
          <span>simulador {num(result.metrics.sim_total_tokens)} tok</span>
          <span>evaluador {num(result.metrics.judge_total_tokens)} tok</span>
          <span>{result.metrics.tool_calls ?? 0} llamadas MCP</span>
          <span>{secs(result.metrics.wall_ms)}</span>
        </div>
      )}

      {result.items && result.items.length > 0 && (
        <Criteria items={result.items} failedMandatory={result.failed_mandatory || []} />
      )}

      <div className="muted" style={{ margin: "12px 0 6px" }}>Transcripcion</div>
      {result.transcript.length === 0 && <p className="muted">Todavia no hay mensajes.</p>}
      {result.transcript.map((turn, index) =>
        turn.role === "user" ? (
          <div key={index} className="msg user">
            <div className="who">
              {result.persona_name} · turno {turn.turn}
              {turn.source === "guion" && " · consulta_inicial del fichero"}
              {turn.closing && " · cierre (no se envia al agente)"}
            </div>
            <div className="bubble">{turn.text}</div>
          </div>
        ) : (
          <div key={index} className="msg assistant">
            <div className="who">Agente · turno {turn.turn}</div>
            {turn.tools.map((call, i) => (
              <ToolCall key={call.call_id || i} call={call} />
            ))}
            {(turn.text || turn.error) && (
              <div className="bubble">{turn.text || <span style={{ color: "var(--err)" }}>{turn.error}</span>}</div>
            )}
            {turn.metrics?.latency_ms !== undefined && (
              <div className="metrics">
                <span>{ms(turn.metrics.latency_ms)}</span>
                <span>{turn.metrics.total_tokens} tokens</span>
                <span>{turn.metrics.tool_calls} tool calls</span>
              </div>
            )}
          </div>
        ),
      )}
    </div>
  );
}

/** Progreso, resultados y detalle de una ejecucion de evaluacion. */
export default function EvalRunView({
  detail,
  results,
  live,
  logs,
  selectedId,
  onSelect,
  onCancel,
  onDelete,
  onOpenInChat,
}: Props) {
  if (!detail) {
    return (
      <div className="eval-empty muted">
        <p>
          Carga un fichero de <strong>personas</strong> y otro de <strong>casos</strong> (o pulsa
          <strong> Plantillas</strong>), revisa la seleccion y ejecuta la evaluacion.
        </p>
        <p>
          Para cada caso y persona, un agente simulador conversa con tu agente conectado al MCP; al terminar,
          un agente evaluador juzga la conversacion con la rubrica del caso. Todo queda en MLflow.
        </p>
      </div>
    );
  }

  const run = detail.run;
  const stats = summarize(results);
  const progress = stats.total ? stats.finished / stats.total : 0;
  const selected = results.find((r) => r.result_id === selectedId) || null;

  return (
    <div className="eval-run">
      <div className="row" style={{ marginBottom: 6, flexWrap: "wrap" }}>
        <strong style={{ fontSize: 14 }}>{run.name}</strong>
        <span className={`badge status-${live ? "running" : run.status}`}>
          {live ? "en curso" : STATUS_LABEL[run.status] || run.status}
        </span>
        <div style={{ flex: 1 }} />
        {live ? (
          <button className="tiny danger" onClick={onCancel} title="Detiene la evaluacion; lo ya evaluado se conserva">
            Cancelar
          </button>
        ) : (
          <button className="tiny danger" onClick={onDelete} title="Borra la ejecucion, su sesion y su rastro en MLflow">
            Borrar
          </button>
        )}
        {detail.mlflow.run_url && (
          <a className="badge" href={detail.mlflow.run_url} target="_blank" rel="noreferrer" style={{ padding: "4px 8px" }}>
            Abrir en MLflow
          </a>
        )}
      </div>

      <div className="eval-progress" title={`${stats.finished} de ${stats.total}`}>
        <div style={{ width: `${progress * 100}%` }} className={live ? "live" : ""} />
      </div>
      <div className="muted" style={{ margin: "4px 0 10px" }}>
        {stats.finished}/{stats.total} conversaciones · experimento{" "}
        {detail.mlflow.experiment_url ? (
          <a href={detail.mlflow.experiment_url} target="_blank" rel="noreferrer" style={{ color: "var(--accent)" }}>
            {detail.mlflow.experiment}
          </a>
        ) : (
          detail.mlflow.experiment || "-"
        )}{" "}
        · trazas: <code>{detail.mlflow.trace_filter}</code>
      </div>

      {logs.map((log, i) => (
        <div key={i} className={log.level === "error" ? "error-box" : "warn-box"}>
          {log.message}
        </div>
      ))}
      {run.error && <div className="error-box">{run.error}</div>}

      <div className="stat-grid eval-stats">
        <div className="stat">
          <div className="label">Aprobados</div>
          <div className="value">
            {stats.passed}/{stats.judged}
            <span className="muted"> {pct(stats.passRate)}</span>
          </div>
        </div>
        <div className="stat">
          <div className="label">Nota media</div>
          <div className="value">{pct(stats.judged ? stats.avgScore : null)}</div>
        </div>
        <div className="stat">
          <div className="label">Errores</div>
          <div className="value" style={{ color: stats.errors ? "var(--err)" : undefined }}>{stats.errors}</div>
        </div>
        <div className="stat">
          <div className="label">Tokens agente / sim. / eval.</div>
          <div className="value" style={{ fontSize: 13 }}>
            {num(stats.tokens.agent)} / {num(stats.tokens.simulator)} / {num(stats.tokens.judge)}
          </div>
        </div>
      </div>

      {(stats.byCase.length > 1 || stats.byPersona.length > 1) && (
        <div className="eval-groups">
          {[
            { title: "Por caso", rows: stats.byCase },
            { title: "Por persona", rows: stats.byPersona },
          ].map((group) => (
            <table key={group.title} className="data">
              <thead>
                <tr>
                  <th>{group.title}</th>
                  <th>Aprob.</th>
                  <th>Nota</th>
                </tr>
              </thead>
              <tbody>
                {group.rows.map((row) => (
                  <tr key={row.id}>
                    <td title={row.id} style={{ fontFamily: "inherit" }}>{row.label}</td>
                    <td>
                      {row.passed}/{row.runs}
                    </td>
                    <td>{pct(row.avg)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ))}
        </div>
      )}

      <table className="data eval-results">
        <thead>
          <tr>
            <th>#</th>
            <th>Caso</th>
            <th>Persona</th>
            <th>Estado</th>
            <th>Nota</th>
            <th>Turnos</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r) => (
            <tr
              key={r.result_id}
              className={r.result_id === selectedId ? "selected" : ""}
              onClick={() => onSelect(r.result_id)}
            >
              <td>{r.seq + 1}</td>
              <td style={{ fontFamily: "inherit" }} title={r.case_id}>
                {r.case_title}
                {r.repetition > 1 && <span className="muted"> #{r.repetition}</span>}
              </td>
              <td style={{ fontFamily: "inherit" }} title={r.persona_id}>{r.persona_name}</td>
              <td>
                {r.status === "running" ? (
                  <span className="row" style={{ gap: 6 }}>
                    <span className="spin" /> {PHASE_LABEL[r.phase || ""] || "en curso"}
                  </span>
                ) : (
                  <StatusBadge status={r.status} />
                )}
              </td>
              <td>{pct(r.score)}</td>
              <td>{r.turns || (r.status === "running" ? r.turn ?? "-" : "-")}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {selected ? (
        <ResultDetail
          result={selected}
          sessionId={run.session_id}
          experimentUrl={detail.mlflow.experiment_url}
          onOpenInChat={onOpenInChat}
        />
      ) : (
        <p className="muted" style={{ marginTop: 10 }}>Selecciona una fila para ver la conversacion y su evaluacion.</p>
      )}
    </div>
  );
}
