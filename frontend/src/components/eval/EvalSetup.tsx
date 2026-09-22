import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../../lib/api";
import { accountFromUrl, withAccount } from "../../lib/providers";
import type { BackendConfig, EvalValidation, MlflowExperiment, RoleConfig } from "../../lib/types";

export type AgentSettings = {
  provider: string;
  baseUrl: string;
  model: string;
  temperature: number;
  thinking: boolean;
  maxIterations: number;
  systemPrompt: string;
};

type Props = {
  config: BackendConfig | null;
  agent: AgentSettings;
  agentModels: string[];
  apiKeys: Record<string, string>;
  mcp: { connIds: string[]; servers: number; tools: number };
  experiment: string;
  experiments: MlflowExperiment[];
  starting: boolean;
  onStart: (payload: any) => void;
};

type FileTab = "personas" | "consultas";

const STORAGE_KEY = "agente-pruebas:eval-setup";

const DEFAULT_MAX_TURNS = 6;
const DEFAULT_THRESHOLD = 0.7;

const DEFAULT_SIM: RoleConfig = {
  sameAsAgent: true,
  provider: "",
  base_url: "",
  model: "",
  // Algo de variedad hace que las personas suenen menos a plantilla.
  temperature: 0.8,
  // Escribir como un usuario no necesita razonar y en CPU ahorra mucho tiempo.
  thinking: false,
};

const DEFAULT_JUDGE: RoleConfig = {
  sameAsAgent: true,
  provider: "",
  base_url: "",
  model: "",
  // El juicio debe ser reproducible.
  temperature: 0,
  thinking: true,
};

type Stored = {
  personas: string;
  cases: string;
  /** Nombre del banco de pruebas: el del fichero de consultas cargado. */
  suite: string;
  simulator: RoleConfig;
  judge: RoleConfig;
  repetitions: number;
  maxTurns: string;
  threshold: string;
};

function load(): Stored {
  const fallback: Stored = {
    personas: "",
    cases: "",
    suite: "",
    simulator: DEFAULT_SIM,
    judge: DEFAULT_JUDGE,
    repetitions: 1,
    maxTurns: "",
    threshold: "",
  };
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    return {
      ...fallback,
      ...stored,
      simulator: { ...DEFAULT_SIM, ...(stored.simulator || {}) },
      judge: { ...DEFAULT_JUDGE, ...(stored.judge || {}) },
    };
  } catch {
    return fallback;
  }
}

function download(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "application/json;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}

/** Modelo de un papel (simulador o evaluador). */
function RoleEditor({
  title,
  hint,
  value,
  onChange,
  config,
  agent,
  agentModels,
}: {
  title: string;
  hint: string;
  value: RoleConfig;
  onChange: (value: RoleConfig) => void;
  config: BackendConfig | null;
  agent: AgentSettings;
  agentModels: string[];
}) {
  const set = (patch: Partial<RoleConfig>) => onChange({ ...value, ...patch });
  const provider = value.sameAsAgent ? agent.provider : value.provider || agent.provider;
  const info = config?.catalog?.find((p) => p.name === provider);
  const suggestions = [
    ...(provider === agent.provider ? agentModels : []),
    ...(info?.models.map((m) => m.id) || []),
  ].filter((m, i, all) => all.indexOf(m) === i);
  const listId = `eval-models-${title}`;

  return (
    <div className="eval-role">
      <div className="row" style={{ marginBottom: 6 }}>
        <strong style={{ fontSize: 12 }}>{title}</strong>
        <div style={{ flex: 1 }} />
        <label className="check">
          <input
            type="checkbox"
            checked={value.sameAsAgent}
            onChange={(e) =>
              set({
                sameAsAgent: e.target.checked,
                // Al desmarcar se parte del modelo del agente, no de un hueco.
                provider: value.provider || agent.provider,
                model: value.model || agent.model,
                base_url: value.base_url || agent.baseUrl,
              })
            }
          />
          mismo modelo que el agente
        </label>
      </div>

      {!value.sameAsAgent && (
        <>
          <div className="row" style={{ marginBottom: 6 }}>
            <select
              value={value.provider || agent.provider}
              onChange={(e) => set({ provider: e.target.value, base_url: "", model: "" })}
              style={{ width: 150, flexShrink: 0 }}
            >
              {(config?.catalog || []).map((p) => (
                <option key={p.name} value={p.name}>
                  {p.label}
                </option>
              ))}
            </select>
            <input
              value={value.model}
              onChange={(e) => set({ model: e.target.value })}
              list={listId}
              placeholder="modelo"
              spellCheck={false}
            />
            <datalist id={listId}>
              {suggestions.map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
          </div>
          {info?.account_env ? (
            // La URL de Cloudflare lleva la cuenta dentro: se pide solo el Account ID.
            <input
              value={accountFromUrl(info.default_base_url, value.base_url)}
              onChange={(e) => set({ base_url: withAccount(info.default_base_url, e.target.value) })}
              placeholder={`Account ID (vacio = ${info.account_env} de backend/.env)`}
              spellCheck={false}
              style={{ marginBottom: 6 }}
            />
          ) : (
            <input
              value={value.base_url}
              onChange={(e) => set({ base_url: e.target.value })}
              placeholder={`URL base (por defecto ${info?.default_base_url || "la del backend"})`}
              spellCheck={false}
              style={{ marginBottom: 6 }}
            />
          )}
          {info?.needs_api_key && (
            <p className="muted" style={{ margin: "0 0 6px" }}>
              Usa la clave de {info.label} guardada en la pestana <strong>Modelo</strong> (o la del backend).
            </p>
          )}
        </>
      )}

      <div className="row">
        <label className="check" title="Temperatura del muestreo">
          temp.
          <input
            type="number"
            min={0}
            max={2}
            step={0.1}
            value={value.temperature}
            onChange={(e) => set({ temperature: Number(e.target.value) })}
            style={{ width: 64 }}
          />
        </label>
        <label className="check">
          <input type="checkbox" checked={value.thinking} onChange={(e) => set({ thinking: e.target.checked })} />
          razonamiento
        </label>
      </div>
      <p className="muted" style={{ margin: "4px 0 0" }}>{hint}</p>
    </div>
  );
}

/** Configuracion de una bateria: ficheros, seleccion, modelos y MLflow. */
export default function EvalSetup({
  config,
  agent,
  agentModels,
  apiKeys,
  mcp,
  experiment,
  experiments,
  starting,
  onStart,
}: Props) {
  const initial = useRef(load()).current;
  const [personas, setPersonas] = useState(initial.personas);
  const [cases, setCases] = useState(initial.cases);
  const [suite, setSuite] = useState(initial.suite);
  const [simulator, setSimulator] = useState<RoleConfig>(initial.simulator);
  const [judge, setJudge] = useState<RoleConfig>(initial.judge);
  const [repetitions, setRepetitions] = useState(initial.repetitions);
  const [maxTurns, setMaxTurns] = useState(initial.maxTurns);
  const [threshold, setThreshold] = useState(initial.threshold);
  const [tab, setTab] = useState<FileTab>("personas");
  const [validation, setValidation] = useState<EvalValidation | null>(null);
  const [validating, setValidating] = useState(false);
  const [validationError, setValidationError] = useState("");
  // Se guardan los descartes (no los marcados) para que una consulta o persona
  // nuevas aparezcan marcadas al editar el fichero.
  const [excludedCases, setExcludedCases] = useState<string[]>([]);
  const [excludedPersonas, setExcludedPersonas] = useState<string[]>([]);
  const [runName, setRunName] = useState("");
  const [runExperiment, setRunExperiment] = useState(experiment);
  const fileRef = useRef<HTMLInputElement>(null);

  // El experimento de la barra lateral es el punto de partida.
  useEffect(() => setRunExperiment((current) => current || experiment), [experiment]);

  useEffect(() => {
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          personas, cases, suite, simulator, judge, repetitions, maxTurns, threshold,
        } satisfies Stored),
      );
    } catch {
      /* almacenamiento lleno o bloqueado: se sigue sin recordar */
    }
  }, [personas, cases, suite, simulator, judge, repetitions, maxTurns, threshold]);

  // Validacion automatica, con un pequeño retardo para no llamar al backend
  // en cada tecla.
  useEffect(() => {
    if (!personas.trim() && !cases.trim()) {
      setValidation(null);
      return;
    }
    const timer = window.setTimeout(async () => {
      setValidating(true);
      setValidationError("");
      try {
        setValidation(
          await api.post<EvalValidation>("/api/eval/validate", {
            personas_json: personas,
            consultas_json: cases,
          }),
        );
      } catch (e: any) {
        setValidationError(e.message);
      } finally {
        setValidating(false);
      }
    }, 600);
    return () => window.clearTimeout(timer);
  }, [personas, cases]);

  const loadTemplates = async () => {
    if ((personas.trim() || cases.trim()) && !window.confirm("Se sustituira el contenido de los dos ficheros por las plantillas.")) {
      return;
    }
    const data = await api.get<{ personas_json: string; consultas_json: string }>("/api/eval/templates");
    setPersonas(data.personas_json);
    setCases(data.consultas_json);
    setSuite("");
  };

  const loadFile = async (file: File | undefined) => {
    if (!file) return;
    const text = await file.text();
    if (tab === "personas") {
      setPersonas(text);
      return;
    }
    setCases(text);
    // El nombre del fichero de consultas identifica el banco de pruebas en MLflow.
    setSuite(file.name.replace(/\.json$/i, ""));
  };

  const includedCases = (validation?.cases || []).filter((c) => !excludedCases.includes(c.id));
  const includedPersonas = (validation?.personas || []).filter((p) => !excludedPersonas.includes(p.id));
  // Cada consulta trae su persona: descartar una persona descarta sus consultas.
  const planned = useMemo(
    () => includedCases.filter((c) => !excludedPersonas.includes(c.persona)).length * Math.max(repetitions, 1),
    [includedCases, excludedPersonas, repetitions],
  );

  const toggle = (list: string[], setList: (v: string[]) => void, id: string) =>
    setList(list.includes(id) ? list.filter((x) => x !== id) : [...list, id]);

  const rolePayload = (role: RoleConfig) => {
    const provider = role.sameAsAgent ? agent.provider : role.provider || agent.provider;
    return {
      provider,
      base_url: (role.sameAsAgent ? agent.baseUrl : role.base_url) || null,
      model: (role.sameAsAgent ? agent.model : role.model) || null,
      api_key: apiKeys[provider] || null,
      temperature: role.temperature,
      thinking: role.thinking,
    };
  };

  const start = () => {
    onStart({
      personas_json: personas,
      consultas_json: cases,
      name: runName.trim(),
      suite: suite.trim(),
      mlflow_experiment: runExperiment.trim(),
      agent: {
        provider: agent.provider,
        base_url: agent.baseUrl || null,
        model: agent.model,
        api_key: apiKeys[agent.provider] || null,
        temperature: agent.temperature,
        thinking: agent.thinking,
        system_prompt: agent.systemPrompt,
        max_iterations: agent.maxIterations,
      },
      simulator: rolePayload(simulator),
      judge: rolePayload(judge),
      mcp_conn_ids: mcp.connIds,
      // Vacio = todos; se manda la lista solo si hay descartes.
      case_ids: excludedCases.length ? includedCases.map((c) => c.id) : [],
      persona_ids: excludedPersonas.length ? includedPersonas.map((p) => p.id) : [],
      repetitions,
      max_turns: Number(maxTurns) > 0 ? Math.min(Number(maxTurns), 30) : DEFAULT_MAX_TURNS,
      threshold: Number(threshold) > 0 ? Math.min(Number(threshold), 1) : DEFAULT_THRESHOLD,
    });
  };

  const personasOk = validation && !validation.errors.some((e) => e.startsWith("personas.json"));
  const casesOk = validation && !validation.errors.some((e) => e.startsWith("consultas.json"));
  const text = tab === "personas" ? personas : cases;
  const setText = tab === "personas" ? setPersonas : setCases;

  return (
    <div className="eval-setup">
      {/* ------------------------------------------------------ ficheros */}
      <section>
        <div className="eval-section-title">
          1 · Ficheros
          <div style={{ flex: 1 }} />
          <button className="tiny" onClick={loadTemplates} title="Cargar las plantillas con la estructura propuesta">
            Plantillas
          </button>
        </div>

        <div className="tabs eval-file-tabs">
          {(["personas", "consultas"] as FileTab[]).map((t) => {
            const ok = t === "personas" ? personasOk : casesOk;
            const empty = !(t === "personas" ? personas : cases).trim();
            return (
              <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
                <span className={`dot ${empty ? "off" : ok ? "ok" : "err"}`} style={{ marginRight: 6 }} />
                {t}.json
              </button>
            );
          })}
        </div>

        <div className="row" style={{ margin: "8px 0 6px" }}>
          <button className="tiny" onClick={() => fileRef.current?.click()}>
            Cargar archivo
          </button>
          <button className="tiny" onClick={() => download(`${tab}.json`, text)} disabled={!text.trim()}>
            Descargar
          </button>
          <div style={{ flex: 1 }} />
          <span className="muted">{validating ? "validando..." : `${text.split("\n").length} lineas`}</span>
          <input
            ref={fileRef}
            type="file"
            accept=".json,application/json"
            hidden
            onChange={(e) => {
              loadFile(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
        </div>

        <textarea
          className="file-editor"
          value={text}
          onChange={(e) => setText(e.target.value)}
          spellCheck={false}
          placeholder={
            tab === "personas"
              ? 'Carga personas.json: {"id_de_persona": {"nombre": "...", "descripcion": "..."}}'
              : 'Carga consultas.json: [{"id": "...", "persona": "...", "goal": "...", ...}]'
          }
        />

        {validationError && <div className="error-box">{validationError}</div>}
        {validation && validation.errors.length > 0 && (
          <div className="error-box">
            {validation.errors.map((e, i) => (
              <div key={i}>• {e}</div>
            ))}
          </div>
        )}
        {validation && validation.warnings.length > 0 && (
          <div className="warn-box">
            {validation.warnings.map((w, i) => (
              <div key={i}>• {w}</div>
            ))}
          </div>
        )}
      </section>

      {/* ----------------------------------------------------- seleccion */}
      {validation?.ok && (
        <section>
          <div className="eval-section-title">
            2 · Que ejecutar
            <div style={{ flex: 1 }} />
            <span className="muted" style={{ textTransform: "none", letterSpacing: 0 }}>
              {suite || "consultas"}
            </span>
          </div>

          <div className="muted" style={{ marginBottom: 4 }}>Consultas</div>
          {validation.cases.map((c) => (
            <label
              key={c.id}
              className="eval-pick"
              title={`${c.goal}\n\nEsperado: ${c.resultado_esperado || "(no se indica)"}\nValor: ${
                c.valor_esperado || "(no se indica)"
              }`}
            >
              <input
                type="checkbox"
                checked={!excludedCases.includes(c.id) && !excludedPersonas.includes(c.persona)}
                disabled={excludedPersonas.includes(c.persona)}
                onChange={() => toggle(excludedCases, setExcludedCases, c.id)}
              />
              <span className="title">
                {c.id} · {c.goal}
              </span>
              <span className="badge">{c.persona}</span>
              {c.ambiguedad && <span className="muted">amb. {c.ambiguedad}</span>}
            </label>
          ))}

          <div className="muted" style={{ margin: "8px 0 4px" }}>Personas</div>
          {validation.personas.map((p) => (
            <label key={p.id} className="eval-pick" title={p.descripcion}>
              <input
                type="checkbox"
                checked={!excludedPersonas.includes(p.id)}
                onChange={() => toggle(excludedPersonas, setExcludedPersonas, p.id)}
              />
              <span className="title">{p.nombre}</span>
              <span className="badge">{p.id}</span>
              <span className="muted">
                {validation.cases.filter((c) => c.persona === p.id).length} consulta(s)
              </span>
            </label>
          ))}

          <div className="row" style={{ marginTop: 8 }}>
            <label className="field" style={{ flex: 1, marginBottom: 0 }}>
              <span>Repeticiones</span>
              <input
                type="number"
                min={1}
                max={20}
                value={repetitions}
                onChange={(e) => setRepetitions(Math.min(Math.max(Number(e.target.value) || 1, 1), 20))}
              />
            </label>
            <label className="field" style={{ flex: 1, marginBottom: 0 }}>
              <span>Max. turnos</span>
              <input
                type="number"
                min={1}
                max={30}
                value={maxTurns}
                placeholder={String(DEFAULT_MAX_TURNS)}
                onChange={(e) => setMaxTurns(e.target.value)}
              />
            </label>
            <label className="field" style={{ flex: 1, marginBottom: 0 }} title="Nota minima para aprobar (0-1)">
              <span>Umbral</span>
              <input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={threshold}
                placeholder={String(DEFAULT_THRESHOLD)}
                onChange={(e) => setThreshold(e.target.value)}
              />
            </label>
          </div>
        </section>
      )}

      {/* ------------------------------------------------------- modelos */}
      <section>
        <div className="eval-section-title">3 · Modelos</div>
        <div className="eval-role">
          <div className="row" style={{ marginBottom: 4 }}>
            <strong style={{ fontSize: 12 }}>Agente bajo prueba</strong>
          </div>
          <div className="muted">
            {agent.provider} / <code>{agent.model || "-"}</code> · temp. {agent.temperature} ·{" "}
            {agent.thinking ? "con" : "sin"} razonamiento · {agent.maxIterations} iter.
          </div>
          <div className="muted" style={{ color: mcp.servers ? undefined : "var(--warn)" }}>
            {mcp.servers
              ? `${mcp.servers} servidor(es) MCP · ${mcp.tools} herramientas`
              : "Ningun servidor MCP marcado: el agente respondera sin herramientas."}
          </div>
          <p className="muted" style={{ margin: "4px 0 0" }}>
            Se ajusta en las pestanas <strong>Modelo</strong> y <strong>Servidor MCP</strong>, igual que en el chat.
          </p>
        </div>
        <RoleEditor
          title="Simulador de usuario"
          hint="Interpreta a cada persona. No usa herramientas."
          value={simulator}
          onChange={setSimulator}
          config={config}
          agent={agent}
          agentModels={agentModels}
        />
        <RoleEditor
          title="Evaluador"
          hint="Juzga cada criterio con la transcripcion completa. Conviene temperatura 0 y, si puedes, un modelo mas capaz que el agente."
          value={judge}
          onChange={setJudge}
          config={config}
          agent={agent}
          agentModels={agentModels}
        />
      </section>

      {/* -------------------------------------------------------- MLflow */}
      <section>
        <div className="eval-section-title">4 · Registro en MLflow</div>
        <label className="field">
          <span>Experimento</span>
          <input
            value={runExperiment}
            onChange={(e) => setRunExperiment(e.target.value)}
            list="eval-mlflow-experiments"
            placeholder={config?.mlflow?.experiment || "experimento"}
            spellCheck={false}
          />
          <datalist id="eval-mlflow-experiments">
            {experiments.map((e) => (
              <option key={e.experiment_id} value={e.name} />
            ))}
          </datalist>
        </label>
        <label className="field">
          <span>Nombre de la ejecucion</span>
          <input
            value={runName}
            onChange={(e) => setRunName(e.target.value)}
            placeholder={`${suite || "consultas"} · fecha`}
          />
        </label>
        <p className="muted" style={{ marginTop: 0 }}>
          La ejecucion es el run padre; cada consulta, un run hijo con su nota, y cada turno una traza
          con los juicios del evaluador como <em>assessments</em>. Si el experimento no existe, se crea.
          {!config?.mlflow?.available && (
            <span style={{ color: "var(--warn)" }}> MLflow no esta disponible: solo se guardara en SQLite.</span>
          )}
        </p>
      </section>

      <button
        className="primary eval-start"
        onClick={start}
        disabled={!validation?.ok || planned === 0 || starting}
        title={!validation?.ok ? "Corrige los ficheros para poder ejecutar" : ""}
      >
        {starting ? "Arrancando..." : `Ejecutar evaluacion · ${planned} conversacion${planned === 1 ? "" : "es"}`}
      </button>
    </div>
  );
}
