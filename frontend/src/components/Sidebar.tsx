import { useEffect, useState } from "react";
import type { Conversation, MlflowExperiment, Session } from "../lib/types";

type Props = {
  sessions: Session[];
  currentSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onDeleteSession: (id: string) => void;
  onRenameSession: (id: string, title: string) => void;
  onPruneSessions: () => void;
  experiment: string;
  experiments: MlflowExperiment[];
  onExperimentChange: (name: string) => void;
  onExperimentApply: (name: string) => void;
  mlflowAvailable: boolean;
  conversations: Conversation[];
  currentConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  onDeleteConversation: (id: string) => void;
};

const when = (ts: number) => new Date(ts * 1000).toLocaleString();

/**
 * Selector del experimento de MLflow.
 *
 * Combina desplegable y campo libre: se elige uno de los que ya existen o se
 * escribe un nombre nuevo, que el backend crea al aplicarlo. Lo elegido se
 * guarda en la sesion abierta, asi que cada sesion puede registrar en un
 * experimento distinto.
 */
function ExperimentPicker({
  experiment,
  experiments,
  available,
  onChange,
  onApply,
}: {
  experiment: string;
  experiments: MlflowExperiment[];
  available: boolean;
  onChange: (name: string) => void;
  onApply: (name: string) => void;
}) {
  const [draft, setDraft] = useState(experiment);

  // Al cambiar de sesion, el experimento cambia desde fuera.
  useEffect(() => setDraft(experiment), [experiment]);

  const known = experiments.some((e) => e.name === draft.trim());
  const dirty = draft.trim() !== experiment.trim();
  const current = experiments.find((e) => e.name === experiment.trim());

  return (
    <div style={{ padding: "8px 8px 10px", borderBottom: "1px solid var(--line)" }}>
      <div className="muted" style={{ marginBottom: 4 }}>
        Experimento de MLflow
      </div>

      <div className="row" style={{ marginBottom: 4 }}>
        <input
          value={draft}
          onChange={(e) => {
            setDraft(e.target.value);
            onChange(e.target.value);
          }}
          onKeyDown={(e) => e.key === "Enter" && onApply(draft)}
          list="mlflow-experiments"
          placeholder="nombre del experimento"
          spellCheck={false}
          style={{ flex: 1 }}
        />
        <button
          className="tiny primary"
          onClick={() => onApply(draft)}
          disabled={!draft.trim() || (!dirty && known)}
          title={known ? "Registrar en este experimento" : "Se creara al aplicarlo"}
        >
          {known ? "Aplicar" : "Crear"}
        </button>
      </div>

      <datalist id="mlflow-experiments">
        {experiments.map((e) => (
          <option key={e.experiment_id} value={e.name} />
        ))}
      </datalist>

      {!available && <p className="muted">MLflow no responde; el nombre se aplicara cuando vuelva.</p>}
      {available && draft.trim() && !known && (
        <p className="muted">No existe todavia: se creara al aplicarlo.</p>
      )}
      {available && current && (
        <a href={current.url} target="_blank" rel="noreferrer" style={{ color: "var(--accent)", fontSize: 11 }}>
          Abrir en MLflow
        </a>
      )}
    </div>
  );
}

/**
 * Una sesion en la lista, con renombrado en linea.
 *
 * El nombre es solo una etiqueta: la sesion se identifica por su id tanto en
 * SQLite como en MLflow (`tags.session_id`), asi que puede repetirse. Al
 * guardarlo, el backend actualiza tambien el nombre del run de MLflow.
 */
function SessionRow({
  session,
  active,
  onSelect,
  onRename,
  onDelete,
}: {
  session: Session;
  active: boolean;
  onSelect: () => void;
  onRename: (title: string) => void;
  onDelete: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(session.title);

  const start = () => {
    setDraft(session.title || "");
    setEditing(true);
  };

  const save = () => {
    setEditing(false);
    if (draft.trim() && draft.trim() !== session.title) onRename(draft);
  };

  return (
    <div className={`list-item ${active ? "active" : ""}`} onClick={editing ? undefined : onSelect}>
      <div className="row">
        {editing ? (
          <input
            autoFocus
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onClick={(e) => e.stopPropagation()}
            onBlur={save}
            onKeyDown={(e) => {
              if (e.key === "Enter") save();
              if (e.key === "Escape") setEditing(false);
            }}
            style={{ flex: 1 }}
            spellCheck={false}
          />
        ) : (
          <div
            className="title"
            style={{ flex: 1 }}
            title="Doble clic para renombrar"
            onDoubleClick={(e) => {
              e.stopPropagation();
              start();
            }}
          >
            {session.title || session.id}
          </div>
        )}

        {!editing && (
          <div className="actions">
            <button
              className="tiny"
              title="Renombrar la sesion"
              onClick={(e) => {
                e.stopPropagation();
                start();
              }}
            >
              ✎
            </button>
            <button
              className="tiny danger"
              title="Eliminar la sesion y su registro en MLflow"
              onClick={(e) => {
                e.stopPropagation();
                onDelete();
              }}
            >
              ✕
            </button>
          </div>
        )}
      </div>

      <div className="meta" title={when(session.updated_at)}>
        {session.conversations} conv · {session.interactions} inter ·{" "}
        {session.total_tokens.toLocaleString()} tok
      </div>
      {session.mlflow_experiment && <div className="meta">exp: {session.mlflow_experiment}</div>}
    </div>
  );
}

/** Navegacion por sesiones (N conversaciones) y por conversaciones. */
export default function Sidebar({
  sessions,
  currentSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  onRenameSession,
  onPruneSessions,
  experiment,
  experiments,
  onExperimentChange,
  onExperimentApply,
  mlflowAvailable,
  conversations,
  currentConversationId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
}: Props) {
  return (
    <aside className="sidebar">
      <ExperimentPicker
        experiment={experiment}
        experiments={experiments}
        available={mlflowAvailable}
        onChange={onExperimentChange}
        onApply={onExperimentApply}
      />

      <div className="panel-head">
        Sesiones
        <div style={{ flex: 1 }} />
        <button
          className="tiny"
          onClick={onPruneSessions}
          title="Borrar las sesiones que no llegaron a registrar ninguna interaccion"
        >
          Limpiar
        </button>
        <button className="tiny" onClick={onNewSession} title="Crear una sesion nueva">
          + Sesion
        </button>
      </div>

      <div style={{ maxHeight: "34%", overflowY: "auto", padding: "8px 8px 0", flexShrink: 0 }}>
        {sessions.length === 0 && (
          <p className="muted">
            Sin sesiones. Se crea una al pulsar <strong>+ Sesion</strong> o al enviar el primer mensaje.
          </p>
        )}
        {sessions.map((s) => (
          <SessionRow
            key={s.id}
            session={s}
            active={s.id === currentSessionId}
            onSelect={() => onSelectSession(s.id)}
            onRename={(title) => onRenameSession(s.id, title)}
            onDelete={() => onDeleteSession(s.id)}
          />
        ))}
      </div>

      <div className="panel-head" style={{ borderTop: "1px solid var(--line)" }}>
        Conversaciones
        <div style={{ flex: 1 }} />
        <button className="tiny" onClick={onNewConversation}>
          + Nueva
        </button>
      </div>

      <div className="panel-body" style={{ padding: "8px" }}>
        {conversations.length === 0 && <p className="muted">Escribe un mensaje para empezar.</p>}
        {conversations.map((c) => (
          <div
            key={c.id}
            className={`list-item ${c.id === currentConversationId ? "active" : ""}`}
            onClick={() => onSelectConversation(c.id)}
          >
            <div className="row">
              <div className="title" style={{ flex: 1 }}>
                {c.title || c.id}
              </div>
              <button
                className="tiny danger"
                title="Eliminar conversacion"
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteConversation(c.id);
                }}
              >
                x
              </button>
            </div>
            <div className="meta" title={when(c.updated_at)}>
              {c.interactions ?? 0} interacciones · {c.model || "-"}
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}
