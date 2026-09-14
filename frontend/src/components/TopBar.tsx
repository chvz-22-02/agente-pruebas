import { useState } from "react";
import { getBackendUrl, setBackendUrl } from "../lib/api";
import type { BackendConfig } from "../lib/types";

type Props = {
  health: any;
  config: BackendConfig | null;
  models: string[];
  model: string;
  onModelChange: (model: string) => void;
  onRefresh: () => void;
  busy: boolean;
  /** Lleva a la pestana donde se descargan e intercambian modelos. */
  onOpenModelTab: () => void;
  mode: "chat" | "eval";
  onModeChange: (mode: "chat" | "eval") => void;
};

/**
 * Barra superior: apunta el frontend a un backend concreto (pueden estar en
 * maquinas distintas) y resume el estado del motor LLM y de MLflow.
 */
export default function TopBar({
  health,
  config,
  models,
  model,
  onModelChange,
  onRefresh,
  busy,
  onOpenModelTab,
  mode,
  onModeChange,
}: Props) {
  const [url, setUrl] = useState(getBackendUrl());
  const [dirty, setDirty] = useState(false);

  const applyBackend = () => {
    setBackendUrl(url.trim());
    setDirty(false);
    onRefresh();
  };

  const llmOk = health?.llm?.ok;
  const llmModelOk = health?.llm?.model_available;
  const mlflowOk = health?.mlflow?.available;

  return (
    <header className="topbar">
      <h1>Agente de pruebas MCP</h1>

      <div className="segmented" role="tablist" aria-label="Modo">
        <button className={mode === "chat" ? "active" : ""} onClick={() => onModeChange("chat")}>
          Chat
        </button>
        <button
          className={mode === "eval" ? "active" : ""}
          onClick={() => onModeChange("eval")}
          title="Usuarios simulados contra el agente, con un agente evaluador"
        >
          Evaluacion
        </button>
      </div>

      <div className="inline-field">
        <span>Backend</span>
        <input
          value={url}
          onChange={(e) => {
            setUrl(e.target.value);
            setDirty(true);
          }}
          onKeyDown={(e) => e.key === "Enter" && applyBackend()}
          placeholder="http://192.168.1.50:8090"
          spellCheck={false}
        />
        <button className="tiny" onClick={applyBackend} disabled={!dirty}>
          Aplicar
        </button>
      </div>

      <div className="inline-field">
        <span>Modelo</span>
        {models.length > 0 ? (
          <select value={model} onChange={(e) => onModelChange(e.target.value)} style={{ width: 190 }}>
            {!models.includes(model) && <option value={model}>{model} (no instalado)</option>}
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        ) : (
          <input value={model} onChange={(e) => onModelChange(e.target.value)} spellCheck={false} />
        )}
      </div>

      <div className="spacer" />

      <div className="inline-field" title={health?.llm?.error || health?.llm?.base_url || ""}>
        <span className={`dot ${llmOk ? (llmModelOk ? "ok" : "warn") : "err"}`} />
        <span>
          LLM {health?.llm?.provider || "?"}
          {!llmOk ? " · sin conexion" : ""}
        </span>
        {llmOk && !llmModelOk && (
          <button className="tiny" onClick={onOpenModelTab} title="Abrir la pestana Modelo para descargarlo">
            Descargar {model}
          </button>
        )}
      </div>

      <div className="inline-field" title={health?.mlflow?.status || ""}>
        <span className={`dot ${mlflowOk ? "ok" : "off"}`} />
        {config?.mlflow_url && mlflowOk ? (
          <a href={config.mlflow_url} target="_blank" rel="noreferrer" style={{ color: "var(--accent)", fontSize: 11 }}>
            MLflow
          </a>
        ) : (
          <span>MLflow off</span>
        )}
      </div>

      <button className="tiny" onClick={onRefresh} disabled={busy}>
        {busy ? "..." : "Recargar"}
      </button>
    </header>
  );
}
