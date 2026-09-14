import { useCallback, useEffect, useRef, useState } from "react";
import EvalRunView from "./eval/EvalRunView";
import EvalSetup, { type AgentSettings } from "./eval/EvalSetup";
import { api, followEval } from "../lib/api";
import { applyEvent, FINISHED, fromPersisted, mergePersisted, STATUS_LABEL } from "../lib/evalState";
import type { BackendConfig, EvalEvent, EvalResultView, EvalRunDetail, EvalRunRow, MlflowExperiment } from "../lib/types";

type Props = {
  /** La vista esta delante (modo Evaluacion). Sigue montada aunque no lo este. */
  active: boolean;
  config: BackendConfig | null;
  agent: AgentSettings;
  agentModels: string[];
  apiKeys: Record<string, string>;
  mcp: { connIds: string[]; servers: number; tools: number };
  experiment: string;
  experiments: MlflowExperiment[];
  /** Apunta el inspector (Trazas, Metricas) a la conversacion de un caso. */
  onFocusConversation: (sessionId: string, conversationId: string | null) => void;
  onOpenInChat: (sessionId: string, conversationId: string) => void;
  /** Una evaluacion crea y borra sesiones: la barra lateral se refresca. */
  onSessionsChanged: () => void;
};

const when = (ts: number) =>
  new Date(ts * 1000).toLocaleString([], { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });

/**
 * Banco de evaluacion: usuarios simulados contra el agente, con un evaluador.
 *
 * La ejecucion corre en el backend, no en el navegador: esta vista solo la
 * lanza y la sigue. Si se recarga la pagina, se reengancha a la que este en
 * marcha y reconstruye su estado repitiendo los eventos desde el principio.
 */
export default function EvalWorkbench({
  active,
  config,
  agent,
  agentModels,
  apiKeys,
  mcp,
  experiment,
  experiments,
  onFocusConversation,
  onOpenInChat,
  onSessionsChanged,
}: Props) {
  const [runs, setRuns] = useState<EvalRunRow[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [detail, setDetail] = useState<EvalRunDetail | null>(null);
  const [results, setResults] = useState<EvalResultView[]>([]);
  const [live, setLive] = useState(false);
  const [logs, setLogs] = useState<{ level: string; message: string }[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const [showSetup, setShowSetup] = useState(true);

  const stopRef = useRef<(() => void) | null>(null);
  // Mientras sea true, la seleccion salta sola al caso que se esta ejecutando.
  const followLive = useRef(true);
  const currentRef = useRef<string | null>(null);

  const loadRuns = useCallback(async () => {
    try {
      const data = await api.get<{ runs: EvalRunRow[] }>("/api/eval/runs");
      setRuns(data.runs);
      return data.runs;
    } catch (e: any) {
      setError(e.message);
      return [];
    }
  }, []);

  const refreshDetail = useCallback(async (id: string) => {
    const data = await api.get<EvalRunDetail>(`/api/eval/runs/${id}`);
    if (currentRef.current !== id) return data;
    setDetail(data);
    setResults((current) => mergePersisted(current, data.results));
    return data;
  }, []);

  const onEvent = useCallback(
    (event: EvalEvent) => {
      if (event.eval_run_id !== currentRef.current) return;
      setResults((current) => applyEvent(current, event));
      switch (event.type) {
        case "log":
          setLogs((current) => [...current, { level: event.level, message: event.message }]);
          break;
        case "item_start":
          if (followLive.current) setSelectedId(event.result_id);
          break;
        case "item_end":
          // Lo persistido trae la rubrica completa del caso recien cerrado.
          refreshDetail(event.eval_run_id).catch(() => undefined);
          onSessionsChanged();
          break;
        case "run_end":
          setLive(false);
          stopRef.current = null;
          refreshDetail(event.eval_run_id).catch(() => undefined);
          loadRuns();
          onSessionsChanged();
          break;
      }
    },
    [refreshDetail, loadRuns, onSessionsChanged],
  );

  const openRun = useCallback(
    async (id: string) => {
      stopRef.current?.();
      stopRef.current = null;
      currentRef.current = id;
      setCurrentId(id);
      setLogs([]);
      setError("");
      try {
        const data = await api.get<EvalRunDetail>(`/api/eval/runs/${id}`);
        if (currentRef.current !== id) return;
        setDetail(data);
        const persisted = data.results.map(fromPersisted);
        setResults(persisted);
        setLive(data.live);
        const running = persisted.find((r) => r.status === "running");
        followLive.current = true;
        setSelectedId(running?.result_id || persisted.find((r) => FINISHED.includes(r.status))?.result_id || null);
        if (data.live) {
          // Desde 0: reconstruye la transcripcion del caso en curso. Los
          // eventos de casos ya cerrados se ignoran (manda SQLite).
          stopRef.current = followEval(id, 0, onEvent, () => {
            setLive(false);
            refreshDetail(id).catch(() => undefined);
          });
        }
      } catch (e: any) {
        setError(e.message);
      }
    },
    [onEvent, refreshDetail],
  );

  // Al entrar: la evaluacion en marcha, o si no la mas reciente.
  useEffect(() => {
    loadRuns().then((list) => {
      const target = list.find((r) => r.live) || list[0];
      if (target) openRun(target.id);
    });
    return () => stopRef.current?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // El inspector sigue a la conversacion seleccionada, pero solo con esta
  // vista delante: con el chat abierto no se le cambia la sesion al usuario.
  const focusedConversation = results.find((r) => r.result_id === selectedId)?.conversation_id;
  const focusedSession = detail?.run.session_id;
  useEffect(() => {
    if (active && focusedSession && focusedConversation) onFocusConversation(focusedSession, focusedConversation);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, focusedSession, focusedConversation]);

  const start = async (payload: any) => {
    setStarting(true);
    setError("");
    try {
      const started = await api.post<{ eval_run_id: string }>("/api/eval/runs", payload);
      await loadRuns();
      await openRun(started.eval_run_id);
      onSessionsChanged();
      // En pantallas estrechas la configuracion va encima: se pliega para
      // que el progreso quede a la vista.
      if (window.matchMedia("(max-width: 1500px)").matches) setShowSetup(false);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setStarting(false);
    }
  };

  const cancel = async () => {
    if (!currentId) return;
    if (!window.confirm("Se detendra la evaluacion. Lo ya evaluado se conserva en SQLite y en MLflow.")) return;
    try {
      await api.post(`/api/eval/runs/${currentId}/cancel`);
    } catch (e: any) {
      setError(e.message);
    }
  };

  const remove = async () => {
    if (!currentId || !detail) return;
    if (
      !window.confirm(
        `Se borrara "${detail.run.name}" con su sesion, sus conversaciones y sus runs y trazas en MLflow.`,
      )
    ) {
      return;
    }
    try {
      await api.del(`/api/eval/runs/${currentId}`);
      currentRef.current = null;
      setCurrentId(null);
      setDetail(null);
      setResults([]);
      setSelectedId(null);
      onFocusConversation("", null);
      const list = await loadRuns();
      if (list[0]) openRun(list[0].id);
      onSessionsChanged();
    } catch (e: any) {
      setError(e.message);
    }
  };

  const select = (id: string) => {
    const chosen = results.find((r) => r.result_id === id);
    // Pinchar en otro caso deja de seguir al que corre; pinchar en el que
    // corre vuelve a seguirlo.
    followLive.current = chosen?.status === "running";
    setSelectedId(id);
  };

  const anyLive = runs.some((r) => r.live);

  return (
    <section className="center eval-workbench">
      <div className="panel-head">
        Evaluacion
        <select
          value={currentId || ""}
          onChange={(e) => e.target.value && openRun(e.target.value)}
          style={{ maxWidth: 360, padding: "3px 6px", fontSize: 12, textTransform: "none", letterSpacing: 0 }}
        >
          {!currentId && <option value="">(sin ejecuciones)</option>}
          {runs.map((r) => (
            <option key={r.id} value={r.id}>
              {r.live ? "● " : ""}
              {r.name} · {when(r.created_at)} · {r.live ? "en curso" : STATUS_LABEL[r.status] || r.status}
              {!r.live && r.summary?.pass_rate !== undefined
                ? ` · ${r.summary.passed}/${r.summary.passed + r.summary.failed} aprob.`
                : ""}
            </option>
          ))}
        </select>
        <button className="tiny" onClick={loadRuns}>
          Recargar
        </button>
        <div style={{ flex: 1 }} />
        {anyLive && !live && <span className="muted" style={{ textTransform: "none" }}>hay otra evaluacion en marcha</span>}
        <button className="tiny" onClick={() => setShowSetup((v) => !v)}>
          {showSetup ? "Ocultar configuracion" : "Nueva evaluacion"}
        </button>
      </div>

      {error && <div className="error-box" style={{ margin: "10px 12px 0" }}>{error}</div>}

      <div className="eval-body">
        {showSetup && (
          <div className="eval-col-setup">
            <EvalSetup
              config={config}
              agent={agent}
              agentModels={agentModels}
              apiKeys={apiKeys}
              mcp={mcp}
              experiment={experiment}
              experiments={experiments}
              starting={starting}
              onStart={start}
            />
          </div>
        )}
        <div className="eval-col-run">
          <EvalRunView
            detail={detail}
            results={results}
            live={live}
            logs={logs}
            selectedId={selectedId}
            onSelect={select}
            onCancel={cancel}
            onDelete={remove}
            onOpenInChat={onOpenInChat}
          />
        </div>
      </div>
    </section>
  );
}
