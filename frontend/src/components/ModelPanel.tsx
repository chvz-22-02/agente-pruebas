import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, pullModel, type PullEvent } from "../lib/api";
import { accountFromUrl, withAccount } from "../lib/providers";
import type {
  BackendConfig,
  CatalogModel,
  ModelFeatures,
  ProbeResult,
  ProviderInfo,
} from "../lib/types";

type Props = {
  config: BackendConfig | null;
  health: any;
  models: string[];
  provider: string;
  onProviderChange: (v: string) => void;
  baseUrl: string;
  onBaseUrlChange: (v: string) => void;
  model: string;
  onModelChange: (v: string) => void;
  apiKeys: Record<string, string>;
  onApiKeyChange: (provider: string, key: string) => void;
  temperature: number;
  onTemperatureChange: (v: number) => void;
  thinking: boolean;
  onThinkingChange: (v: boolean) => void;
  maxIterations: number;
  onMaxIterationsChange: (v: number) => void;
  systemPrompt: string;
  onSystemPromptChange: (v: string) => void;
  onModelsRefreshed: (models: string[]) => void;
};

const gb = (bytes?: number) => (bytes ? `${(bytes / 1e9).toFixed(2)} GB` : "");

const THINKING_LABEL: Record<NonNullable<ModelFeatures["thinking"]>, string> = {
  adaptive: "razonamiento adaptativo",
  effort: "razonamiento por esfuerzo",
  toggle: "razonamiento conmutable",
  always: "razona siempre",
  none: "sin razonamiento",
  unknown: "razonamiento sin determinar",
};

/**
 * Pregunta al backend que sabe hacer el modelo elegido.
 *
 * El catalogo estatico solo cubre los modelos que venian de fabrica, asi que
 * para uno descargado despues (o escrito a mano) la unica fuente fiable es el
 * propio motor: Ollama publica las capacidades del modelo en `/api/show`. De
 * ahi sale si el interruptor de razonamiento tiene sentido para este modelo.
 */
function useModelFeatures(provider: string, baseUrl: string, model: string) {
  const [features, setFeatures] = useState<ModelFeatures | null>(null);

  useEffect(() => {
    if (!provider || !model) {
      setFeatures(null);
      return;
    }
    let cancelled = false;
    const query = new URLSearchParams({ provider, model });
    if (baseUrl) query.set("base_url", baseUrl);
    api
      .get<ModelFeatures>(`/api/llm/features?${query.toString()}`)
      .then((data) => !cancelled && setFeatures(data))
      .catch(() => !cancelled && setFeatures(null));
    return () => {
      cancelled = true;
    };
  }, [provider, baseUrl, model]);

  return features;
}

/** Credenciales del proveedor de nube + validacion contra su API. */
/**
 * Identificador de cuenta para proveedores cuya URL lo lleva (Cloudflare).
 * No es un secreto: solo compone la URL. Vacio = lo pone el backend desde
 * su variable de entorno.
 */
function AccountIdBox({
  info,
  baseUrl,
  onBaseUrlChange,
}: {
  info: ProviderInfo;
  baseUrl: string;
  onBaseUrlChange: (url: string) => void;
}) {
  const value = accountFromUrl(info.default_base_url, baseUrl);
  return (
    <label className="field">
      <span>
        {info.account_label || "Account ID"} <code>({info.account_env})</code>
      </span>
      <input
        value={value}
        onChange={(e) => onBaseUrlChange(withAccount(info.default_base_url, e.target.value))}
        placeholder={`vacio = ${info.account_env} de backend/.env`}
        spellCheck={false}
        autoComplete="off"
      />
    </label>
  );
}

function ApiKeyBox({
  info,
  value,
  baseUrl,
  onChange,
  onModels,
}: {
  info: ProviderInfo;
  value: string;
  /** URL efectiva: en Cloudflare ya lleva el Account ID. */
  baseUrl: string;
  onChange: (v: string) => void;
  onModels: (models: string[]) => void;
}) {
  const [reveal, setReveal] = useState(false);
  const [probe, setProbe] = useState<ProbeResult | null>(null);
  const [checking, setChecking] = useState(false);

  const check = async () => {
    setChecking(true);
    setProbe(null);
    try {
      const result = await api.post<ProbeResult>("/api/llm/probe", {
        provider: info.name,
        base_url: baseUrl || info.default_base_url || null,
        api_key: value || null,
      });
      setProbe(result);
      if (result.ok && result.models?.length) onModels(result.models);
    } catch (e: any) {
      setProbe({ ok: false, error: e.message, models: [] });
    } finally {
      setChecking(false);
    }
  };

  return (
    <div style={{ marginBottom: 14 }}>
      <label className="field" style={{ marginBottom: 6 }}>
        <span>Clave de API {info.key_env && <code>({info.key_env})</code>}</span>
        <div className="row">
          <input
            type={reveal ? "text" : "password"}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && check()}
            placeholder={info.key_hint || "pega aqui tu clave"}
            spellCheck={false}
            autoComplete="off"
          />
          <button className="tiny" onClick={() => setReveal(!reveal)} title="Mostrar u ocultar">
            {reveal ? "Ocultar" : "Ver"}
          </button>
          <button className="tiny primary" onClick={check} disabled={checking}>
            {checking ? "..." : "Validar"}
          </button>
        </div>
      </label>

      {probe && (
        <div
          className={probe.ok ? "" : "error-box"}
          style={probe.ok ? { fontSize: 12, color: "var(--ok)" } : undefined}
        >
          {probe.ok
            ? `Clave valida. ${probe.models.length} modelos disponibles en la cuenta.`
            : probe.error}
        </div>
      )}

      <p className="muted" style={{ marginTop: 6 }}>
        Se guarda en este navegador para no repegarla cada vez. Si prefieres no dejarla ahi, deja el
        campo vacio y define <code>{info.key_env}</code> en <code>backend/.env</code>.
        {info.console_url && (
          <>
            {" "}
            <a href={info.console_url} target="_blank" rel="noreferrer" style={{ color: "var(--accent)" }}>
              Obtener una clave
            </a>
          </>
        )}
      </p>
    </div>
  );
}

/** Descarga de modelos en el motor local, con progreso en vivo. */
function ModelDownloader({
  config,
  provider,
  baseUrl,
  target,
  installed,
  onSelect,
  onModelsRefreshed,
}: {
  config: BackendConfig | null;
  provider: string;
  baseUrl: string;
  target: string;
  installed: string[];
  onSelect: (model: string) => void;
  onModelsRefreshed: (models: string[]) => void;
}) {
  const [name, setName] = useState(target);
  const [event, setEvent] = useState<PullEvent | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  const start = async (which?: string) => {
    const wanted = (which ?? name).trim();
    if (!wanted || running) return;
    setName(wanted);
    setRunning(true);
    setError("");
    setEvent(null);

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await pullModel({ model: wanted, provider, base_url: baseUrl }, (e) => {
        setEvent(e);
        if (e.type === "error") setError(e.message || "Fallo la descarga");
        if (e.type === "done") {
          onModelsRefreshed(e.models || []);
          if (e.installed) onSelect(wanted);
        }
      }, controller.signal);
    } catch (e: any) {
      if (e.name !== "AbortError") setError(e.message);
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  };

  const percent = event?.percent ?? null;
  const done = event?.type === "done";

  return (
    <>
      <div className="row" style={{ marginBottom: 8 }}>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && start()}
          placeholder="qwen3:4b"
          spellCheck={false}
          disabled={running}
        />
        {running ? (
          <button className="danger" onClick={() => abortRef.current?.abort()}>
            Cancelar
          </button>
        ) : (
          <button className="primary" onClick={() => start()} disabled={!name.trim()}>
            Descargar
          </button>
        )}
      </div>

      <div className="row" style={{ flexWrap: "wrap", gap: 5, marginBottom: 10 }}>
        {(config?.suggested_models || []).map((s) => {
          const have = installed.includes(s.name);
          return (
            <button
              key={s.name}
              className="tiny"
              disabled={running || have}
              title={have ? "Ya instalado" : `${s.size} · ${s.note}`}
              onClick={() => start(s.name)}
            >
              {have ? "✓ " : "↓ "}
              {s.name}
              <span className="muted"> {s.size}</span>
            </button>
          );
        })}
      </div>

      {event && (
        <div style={{ marginBottom: 10 }}>
          <div className="row" style={{ justifyContent: "space-between", marginBottom: 4 }}>
            <span className="muted">{done ? "Descarga completada" : event.status || "..."}</span>
            <span className="muted" style={{ fontFamily: "var(--mono)" }}>
              {percent !== null ? `${percent}%` : ""}
              {event.total ? ` · ${gb(event.completed)} / ${gb(event.total)}` : ""}
            </span>
          </div>
          <div
            style={{
              height: 6,
              background: "var(--bg)",
              border: "1px solid var(--line)",
              borderRadius: 4,
              overflow: "hidden",
            }}
          >
            <div
              style={{
                height: "100%",
                width: `${done ? 100 : percent || 0}%`,
                background: done ? "var(--ok)" : "var(--accent)",
                transition: "width 0.3s",
              }}
            />
          </div>
        </div>
      )}

      {error && <div className="error-box">{error}</div>}

      <p className="muted">
        La descarga ocurre en el motor, no en el navegador: puedes cerrar esta pestana y seguira. Si la
        cancelas y la relanzas, Ollama reaprovecha las capas ya bajadas.
      </p>
    </>
  );
}

export default function ModelPanel(props: Props) {
  const {
    config, health, models, provider, baseUrl, model, temperature,
    thinking, maxIterations, systemPrompt, apiKeys,
  } = props;

  const catalog = config?.catalog || [];
  const info = useMemo(() => catalog.find((p) => p.name === provider), [catalog, provider]);
  const catalogModel = info?.models.find((m) => m.id === model);
  // Lo que dice el motor manda sobre lo que dice el catalogo.
  const features = useModelFeatures(provider, baseUrl, model);
  const thinkingMode = features?.thinking ?? catalogModel?.thinking ?? "unknown";
  // Solo se bloquea el interruptor cuando hay certeza: o el motor ha dicho que
  // el modelo no razona, o el catalogo dice que razona siempre.
  const thinkingLocked =
    thinkingMode === "always" || (features ? features.can_disable_thinking === false : false);
  const isCloud = info?.kind === "cloud";
  const canPull = (config?.pull_capable || []).includes(provider);
  const installed = models.includes(model);
  const apiKey = apiKeys[provider] || "";

  /** Al cambiar de proveedor, arrastra su URL base y su modelo recomendado. */
  const switchProvider = useCallback(
    (next: string) => {
      const target = catalog.find((p) => p.name === next);
      props.onProviderChange(next);
      props.onBaseUrlChange(target?.default_base_url || "");
      const suggested = target?.models.find((m) => m.recommended) || target?.models[0];
      if (suggested) props.onModelChange(suggested.id);
      props.onModelsRefreshed([]);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [catalog],
  );

  // Modelos a mostrar: los del catalogo, mas los que devuelva la cuenta o el motor.
  const options = useMemo(() => {
    const seen = new Set<string>();
    const rows: { id: string; label: string; note: string; have: boolean }[] = [];
    for (const m of info?.models || []) {
      seen.add(m.id);
      rows.push({
        id: m.id,
        label: m.label,
        note: [m.context, m.notes].filter(Boolean).join(" · "),
        have: !isCloud ? models.includes(m.id) : models.length === 0 || models.includes(m.id),
      });
    }
    for (const id of models) {
      if (!seen.has(id)) rows.push({ id, label: id, note: "no esta en el catalogo", have: true });
    }
    return rows;
  }, [info, models, isCloud]);

  return (
    <div className="panel-body">
      <label className="field">
        <span>Proveedor</span>
        <select value={provider} onChange={(e) => switchProvider(e.target.value)}>
          {catalog.map((p) => (
            <option key={p.name} value={p.name}>
              {p.label}
            </option>
          ))}
        </select>
      </label>

      {info?.account_env && (
        <AccountIdBox info={info} baseUrl={baseUrl} onBaseUrlChange={props.onBaseUrlChange} />
      )}

      {info?.needs_api_key && (
        <ApiKeyBox
          info={info}
          value={apiKey}
          baseUrl={baseUrl}
          onChange={(v) => props.onApiKeyChange(provider, v)}
          onModels={props.onModelsRefreshed}
        />
      )}

      {!isCloud && (
        <label className="field">
          <span>URL del motor de inferencia</span>
          <input value={baseUrl} onChange={(e) => props.onBaseUrlChange(e.target.value)} spellCheck={false} />
        </label>
      )}

      <label className="field">
        <span>Modelo</span>
        <select
          value={options.some((o) => o.id === model) ? model : "__custom__"}
          onChange={(e) => e.target.value !== "__custom__" && props.onModelChange(e.target.value)}
        >
          {options.map((o) => (
            <option key={o.id} value={o.id}>
              {o.have ? "" : "↓ "}
              {o.label}
              {o.note ? ` — ${o.note}` : ""}
            </option>
          ))}
          <option value="__custom__">(otro, escribir abajo)</option>
        </select>
      </label>

      <label className="field">
        <span>Identificador exacto</span>
        <input value={model} onChange={(e) => props.onModelChange(e.target.value)} spellCheck={false} />
      </label>

      <p className="muted" style={{ marginTop: -4, marginBottom: 12 }}>
        <span className="badge">{THINKING_LABEL[thinkingMode]}</span>{" "}
        {catalogModel && !catalogModel.sampling && <span className="badge">ignora temperature</span>}{" "}
        {features?.source === "engine" && (
          <span title={(features.capabilities || []).join(", ")}>
            segun el motor
            {features.tools === false && " · sin soporte de herramientas"}
          </span>
        )}
        {features?.source === "desconocido" && <span>modelo fuera del catalogo</span>}
      </p>

      {isCloud && !apiKey && !health?.llm?.ok && (
        <div
          className="error-box"
          style={{ borderColor: "var(--warn)", background: "rgba(210,153,34,0.1)", color: "#f0d08a" }}
        >
          Este proveedor necesita una clave de API. Pegala arriba o define <code>{info?.key_env}</code>{" "}
          en <code>backend/.env</code>.
        </div>
      )}

      {canPull && model && !installed && (
        <div
          className="error-box"
          style={{ borderColor: "var(--warn)", background: "rgba(210,153,34,0.1)", color: "#f0d08a" }}
        >
          <strong>{model}</strong> no esta instalado en el motor. Descargalo aqui abajo antes de usarlo.
        </div>
      )}

      {canPull && (
        <details className="block" style={{ marginBottom: 14 }} open={!installed}>
          <summary>Descargar un modelo</summary>
          <div className="inner">
            <ModelDownloader
              config={config}
              provider={provider}
              baseUrl={baseUrl}
              target={model}
              installed={models}
              onSelect={props.onModelChange}
              onModelsRefreshed={props.onModelsRefreshed}
            />
          </div>
        </details>
      )}

      <label className="field">
        <span>
          Temperatura: {temperature}
          {catalogModel && !catalogModel.sampling && " (este modelo la ignora)"}
        </span>
        <input
          type="range"
          min={0}
          max={1.5}
          step={0.05}
          value={temperature}
          disabled={!!catalogModel && !catalogModel.sampling}
          onChange={(e) => props.onTemperatureChange(Number(e.target.value))}
        />
      </label>

      <label className="field">
        <span>Maximo de iteraciones de herramientas</span>
        <input
          type="number"
          min={1}
          max={40}
          value={maxIterations}
          onChange={(e) => props.onMaxIterationsChange(Number(e.target.value))}
        />
      </label>

      <label className="row" style={{ marginBottom: 12 }}>
        <input
          type="checkbox"
          checked={thinking && !thinkingLocked}
          onChange={(e) => props.onThinkingChange(e.target.checked)}
          style={{ width: 14 }}
          disabled={thinkingLocked}
        />
        <span className="muted">
          Modo razonamiento
          {thinkingMode === "always" && " (este modelo razona siempre)"}
          {thinkingLocked && thinkingMode !== "always" && " (este modelo no razona)"}
        </span>
      </label>
      {features?.tools === false && (
        <div
          className="error-box"
          style={{ borderColor: "var(--warn)", background: "rgba(210,153,34,0.1)", color: "#f0d08a" }}
        >
          El motor declara que <strong>{model}</strong> no admite herramientas: no podra llamar al
          servidor MCP.
        </div>
      )}

      <label className="field">
        <span>System prompt (vacio = el del backend)</span>
        <textarea
          rows={8}
          value={systemPrompt}
          onChange={(e) => props.onSystemPromptChange(e.target.value)}
          placeholder="Instrucciones especificas para esta bateria de pruebas..."
        />
      </label>

      <p className="muted">Los cambios se aplican al siguiente mensaje.</p>

      <hr className="sep" />
      <div className="muted" style={{ marginBottom: 6 }}>Estado del proveedor</div>
      <dl className="kv">
        <dt>Estado</dt>
        <dd style={{ color: health?.llm?.ok ? "var(--ok)" : "var(--err)" }}>
          {health?.llm?.ok ? "conectado" : health?.llm?.needs_api_key ? "falta clave" : "sin conexion"}
        </dd>
        {health?.llm?.version && (
          <>
            <dt>Version</dt>
            <dd>{health.llm.version}</dd>
          </>
        )}
        <dt>Modelos</dt>
        <dd>{models.length}</dd>
      </dl>
      {health?.llm?.error && <div className="error-box" style={{ marginTop: 8 }}>{health.llm.error}</div>}
    </div>
  );
}
