import { useCallback, useEffect, useRef, useState } from "react";
import Chat from "./components/Chat";
import EvalWorkbench from "./components/EvalWorkbench";
import McpPanel from "./components/McpPanel";
import MetricsPanel from "./components/MetricsPanel";
import ModelPanel from "./components/ModelPanel";
import Sidebar from "./components/Sidebar";
import TopBar from "./components/TopBar";
import TracesPanel from "./components/TracesPanel";
import { api, loadApiKeys, saveApiKeys, streamChat } from "./lib/api";
import type {
  AgentEvent,
  BackendConfig,
  ChatItem,
  Conversation,
  McpConnection,
  MlflowExperiment,
  Session,
} from "./lib/types";

type Tab = "mcp" | "trazas" | "metricas" | "modelo";
type Mode = "chat" | "eval";

const SELECTED_KEY = "agente-pruebas:mcp-selected";
const EXPERIMENT_KEY = "agente-pruebas:mlflow-experiment";
const MODE_KEY = "agente-pruebas:mode";

export default function App() {
  const [config, setConfig] = useState<BackendConfig | null>(null);
  const [health, setHealth] = useState<any>(null);
  const [models, setModels] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [globalError, setGlobalError] = useState("");

  const [sessions, setSessions] = useState<Session[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);

  const [connections, setConnections] = useState<McpConnection[]>([]);
  const [selectedConns, setSelectedConns] = useState<string[]>([]);

  // Experimento de MLflow donde se registra lo que se ejecute. Se recuerda en
  // el navegador y se guarda en cada sesion, para que reabrir una sesion vieja
  // siga apuntando a su experimento.
  const [experiment, setExperiment] = useState<string>(() => localStorage.getItem(EXPERIMENT_KEY) || "");
  const [experiments, setExperiments] = useState<MlflowExperiment[]>([]);

  const [items, setItems] = useState<ChatItem[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [tab, setTab] = useState<Tab>("mcp");
  const [refreshKey, setRefreshKey] = useState(0);
  // Chat manual o banco de evaluacion. La vista de evaluacion sigue montada
  // aunque no se vea, para no perder el seguimiento en vivo al cambiar.
  const [mode, setModeState] = useState<Mode>(() =>
    localStorage.getItem(MODE_KEY) === "eval" ? "eval" : "chat",
  );
  // La vista de evaluacion apunta el inspector a sus conversaciones. Al volver
  // al chat se restaura el hilo en el que se estaba: escribir por error en la
  // conversacion de una evaluacion en marcha la contaminaria.
  const chatStale = useRef(false);
  const chatContext = useRef<{ session: string | null; conversation: string | null }>({
    session: null,
    conversation: null,
  });

  // Ajustes del modelo, editables en caliente desde la pestana "Modelo".
  const [model, setModel] = useState("");
  const [provider, setProvider] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [temperature, setTemperature] = useState(0.6);
  const [thinking, setThinking] = useState(true);
  const [maxIterations, setMaxIterations] = useState(12);
  const [systemPrompt, setSystemPrompt] = useState("");
  // Claves de API por proveedor. Viven en el navegador y viajan en cada
  // peticion; el backend no las persiste (ver lib/api.ts).
  const [apiKeys, setApiKeys] = useState<Record<string, string>>(() => loadApiKeys());

  const abortRef = useRef<AbortController | null>(null);
  const defaultsApplied = useRef(false);
  // Ids reales del turno en curso: la sesion puede haberla creado el backend
  // en este mismo mensaje, y el estado de React todavia no la conoce.
  const liveIds = useRef<{ session: string | null; conversation: string | null }>({
    session: null,
    conversation: null,
  });

  // ------------------------------------------------------------- arranque --
  const loadSystem = useCallback(async () => {
    setBusy(true);
    setGlobalError("");
    try {
      const cfg = await api.get<BackendConfig>("/api/config");
      setConfig(cfg);
      setModel((current) => current || cfg.defaults.model);
      setProvider((current) => current || cfg.defaults.provider);
      setBaseUrl((current) => current || cfg.defaults.base_url);
      if (!defaultsApplied.current) {
        defaultsApplied.current = true;
        setTemperature(cfg.defaults.temperature);
        setThinking(cfg.defaults.thinking);
        setMaxIterations(cfg.defaults.max_iterations);
      }
      setExperiments(cfg.mlflow_experiments || []);
      setExperiment((current) => current || cfg.mlflow?.experiment || "");
      setHealth(await api.get("/api/health"));
      setConnections((await api.get<{ connections: McpConnection[] }>("/api/mcp/connections")).connections);
      setSessions((await api.get<{ sessions: Session[] }>("/api/sessions")).sessions);
    } catch (e: any) {
      setGlobalError(`No se pudo contactar con el backend: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    const stored = localStorage.getItem(SELECTED_KEY);
    if (stored) setSelectedConns(JSON.parse(stored));
    loadSystem();
  }, [loadSystem]);

  /**
   * Lista los modelos del proveedor seleccionado en la UI (no el del .env).
   * `probe` no lanza por credenciales invalidas: responde ok:false con motivo,
   * asi que una clave mal pegada deja el desplegable vacio en vez de romper.
   */
  const apiKey = apiKeys[provider] || "";
  useEffect(() => {
    if (!config || !provider) return;
    let cancelled = false;
    const info = config.catalog?.find((p) => p.name === provider);
    if (info?.needs_api_key && !apiKey) {
      setModels([]);
      return;
    }
    api
      .post<{ ok: boolean; models: string[] }>("/api/llm/probe", {
        provider,
        base_url: baseUrl || info?.default_base_url || null,
        api_key: apiKey || null,
      })
      .then((probe) => {
        if (!cancelled) setModels(probe.ok ? probe.models || [] : []);
      })
      .catch(() => {
        if (!cancelled) setModels([]);
      });
    return () => {
      cancelled = true;
    };
  }, [config, provider, baseUrl, apiKey]);

  useEffect(() => {
    localStorage.setItem(SELECTED_KEY, JSON.stringify(selectedConns));
  }, [selectedConns]);

  useEffect(() => {
    saveApiKeys(apiKeys);
  }, [apiKeys]);

  /**
   * Al arrancar se abre la sesion mas reciente, si la hay. Nunca se crea una
   * automaticamente: una sesion nueva nace solo al pulsar "+ Sesion" o al
   * enviar el primer mensaje sin sesion abierta. Antes se creaba aqui, y como
   * este efecto se dispara en cada recarga (a veces antes de que llegue la
   * lista de sesiones), acababa dejando sesiones vacias sueltas.
   */
  const bootstrapped = useRef(false);
  useEffect(() => {
    if (!config || sessionId || bootstrapped.current || sessions.length === 0) return;
    bootstrapped.current = true;
    // La mas reciente de chat: las de evaluacion se abren desde su vista.
    const first = sessions.find((s) => s.metadata?.kind !== "evaluation") || sessions[0];
    setSessionId(first.id);
    chatContext.current = { session: first.id, conversation: null };
    if (first.mlflow_experiment) setExperiment(first.mlflow_experiment);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config, sessions]);

  useEffect(() => {
    if (experiment) localStorage.setItem(EXPERIMENT_KEY, experiment);
  }, [experiment]);

  const loadConversations = useCallback(async (sid: string) => {
    const data = await api.get<{ conversations: Conversation[] }>(`/api/conversations?session_id=${sid}`);
    setConversations(data.conversations);
    return data.conversations;
  }, []);

  useEffect(() => {
    if (sessionId) loadConversations(sessionId);
  }, [sessionId, loadConversations, refreshKey]);

  // ------------------------------------------------------------ acciones ---
  const refreshSessions = useCallback(async () => {
    const data = await api.get<{ sessions: Session[] }>("/api/sessions");
    setSessions(data.sessions);
    return data.sessions;
  }, []);

  const newSession = async () => {
    // Nombre corto y ordenable ("10/09 10:13"). Es solo una etiqueta: la
    // sesion se identifica por su id, asi que puede repetirse sin problema y
    // se renombra desde la propia lista.
    const title = new Date().toLocaleString([], {
      day: "2-digit",
      month: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
    const session = await api.post<Session>("/api/sessions", {
      title,
      mlflow_experiment: experiment,
    });
    setSessionId(session.id);
    setConversationId(null);
    setConversations([]);
    setItems([]);
    await refreshSessions();
  };

  const selectSession = async (id: string) => {
    setSessionId(id);
    setConversationId(null);
    setItems([]);
    const chosen = sessions.find((s) => s.id === id);
    if (chosen?.mlflow_experiment) setExperiment(chosen.mlflow_experiment);
    await loadConversations(id);
  };

  /**
   * Renombra la sesion. El backend propaga el nombre al run de MLflow, que no
   * exige nombres unicos: alli tambien se identifica por `tags.session_id`.
   */
  const renameSession = async (id: string, title: string) => {
    const wanted = title.trim();
    if (!wanted) return;
    await api.patch(`/api/sessions/${id}`, { title: wanted });
    await refreshSessions();
  };

  /** Borra la sesion en el backend y, con ella, sus runs y trazas en MLflow. */
  const deleteSession = async (id: string) => {
    const target = sessions.find((s) => s.id === id);
    const label = target?.title || id;
    if (
      !window.confirm(
        `Se borrara la sesion "${label}" con sus conversaciones e interacciones, ` +
          "y tambien sus runs y trazas en MLflow. Esta accion no se puede deshacer.",
      )
    ) {
      return;
    }
    await api.del(`/api/sessions/${id}`);
    const remaining = await refreshSessions();
    if (id === sessionId) {
      setSessionId(null);
      setConversationId(null);
      setConversations([]);
      setItems([]);
      if (remaining.length > 0) await selectSession(remaining[0].id);
    }
  };

  /** Borra de una vez las sesiones que no llegaron a registrar nada. */
  const pruneSessions = async () => {
    const empty = sessions.filter((s) => !s.interactions && s.id !== sessionId);
    if (empty.length === 0) {
      window.alert("No hay sesiones vacias que borrar.");
      return;
    }
    if (
      !window.confirm(
        `Se borraran ${empty.length} sesiones sin interacciones, y su rastro en MLflow. ` +
          "La sesion abierta se conserva.",
      )
    ) {
      return;
    }
    const result = await api.post<{ deleted: number }>(
      `/api/sessions/prune?keep=${sessionId || ""}`,
    );
    await refreshSessions();
    window.alert(`Sesiones borradas: ${result.deleted}`);
  };

  /**
   * Fija el experimento de MLflow: se crea si el nombre no existe y se guarda
   * en la sesion abierta, para que sus proximos runs vayan ahi.
   */
  const applyExperiment = async (name: string) => {
    const wanted = name.trim();
    if (!wanted) return;
    setExperiment(wanted);
    try {
      await api.post<MlflowExperiment>("/api/mlflow/experiments", { name: wanted });
      const list = await api.get<{ experiments: MlflowExperiment[] }>("/api/mlflow/experiments");
      setExperiments(list.experiments || []);
    } catch (e: any) {
      // MLflow caido: el nombre se guarda igual y se aplicara cuando vuelva.
      setGlobalError(`MLflow: ${e.message}`);
    }
    if (sessionId) {
      await api.patch(`/api/sessions/${sessionId}`, { mlflow_experiment: wanted });
      await refreshSessions();
    }
  };

  /** Reconstruye el hilo visible a partir de los mensajes persistidos. */
  const openConversation = async (id: string) => {
    setConversationId(id);
    const data = await api.get<{ messages: any[] }>(`/api/conversations/${id}/messages`);
    const rebuilt: ChatItem[] = [];
    for (const message of data.messages) {
      if (message.role === "user") {
        rebuilt.push({ kind: "user", text: message.content, ts: message.created_at });
      } else if (message.role === "assistant") {
        if (message.thinking) rebuilt.push({ kind: "thinking", text: message.thinking, ts: message.created_at });
        for (const call of message.tool_calls || []) {
          rebuilt.push({
            kind: "tool",
            callId: call.id,
            seq: 0,
            tool: call.name,
            args: call.arguments,
            status: "ok",
            ts: message.created_at,
          });
        }
        if (message.content && !(message.tool_calls || []).length) {
          rebuilt.push({ kind: "assistant", text: message.content, ts: message.created_at });
        }
      } else if (message.role === "tool") {
        const previous = [...rebuilt].reverse().find((x) => x.kind === "tool" && x.tool === message.name);
        if (previous && previous.kind === "tool") previous.result = message.content;
      }
    }
    setItems(rebuilt);
  };

  const newConversation = async () => {
    if (!sessionId) return;
    const conversation = await api.post<Conversation>("/api/conversations", {
      session_id: sessionId,
      provider,
      model,
      system_prompt: systemPrompt,
    });
    setConversationId(conversation.id);
    setItems([]);
    await loadConversations(sessionId);
  };

  /** Boton "Reiniciar conversacion": abre un hilo limpio y conserva el historico. */
  const resetConversation = async () => {
    if (!conversationId) {
      setItems([]);
      return;
    }
    const data = await api.post<{ conversation: Conversation }>(
      `/api/conversations/${conversationId}/reset?new_thread=true`,
    );
    setConversationId(data.conversation.id);
    setItems([]);
    if (sessionId) await loadConversations(sessionId);
  };

  const deleteConversation = async (id: string) => {
    await api.del(`/api/conversations/${id}`);
    if (id === conversationId) {
      setConversationId(null);
      setItems([]);
    }
    if (sessionId) await loadConversations(sessionId);
  };

  const refreshConnections = async () => {
    const data = await api.get<{ connections: McpConnection[] }>("/api/mcp/connections");
    setConnections(data.connections);
    // Un servidor recien conectado se marca como usable por defecto.
    setSelectedConns((current) => {
      const alive = data.connections.filter((c) => c.alive).map((c) => c.conn_id);
      const kept = current.filter((id) => alive.includes(id));
      const added = alive.filter((id) => !current.includes(id));
      return [...kept, ...added];
    });
  };

  const disconnect = async (connId: string) => {
    await api.del(`/api/mcp/connections/${connId}`);
    setSelectedConns((current) => current.filter((id) => id !== connId));
    await refreshConnections();
  };

  // --------------------------------------------------------- evaluacion ---
  const setMode = (next: Mode) => {
    if (next === mode) return;
    if (next === "eval") chatContext.current = { session: sessionId, conversation: conversationId };
    setModeState(next);
    localStorage.setItem(MODE_KEY, next);
    if (next === "chat" && chatStale.current) {
      chatStale.current = false;
      const { session, conversation } = chatContext.current;
      setSessionId(session);
      setConversationId(conversation);
      if (conversation) openConversation(conversation);
      else setItems([]);
    }
  };

  /**
   * La evaluacion apunta el inspector al caso seleccionado: asi las pestanas
   * Trazas y Metricas muestran sus llamadas al MCP sin salir de la vista.
   */
  const focusConversation = useCallback((sid: string, cid: string | null) => {
    if (!sid) return;
    setSessionId(sid);
    setConversationId(cid);
    chatStale.current = true;
  }, []);

  const openInChat = async (sid: string, cid: string) => {
    chatStale.current = false;
    setModeState("chat");
    localStorage.setItem(MODE_KEY, "chat");
    setSessionId(sid);
    await loadConversations(sid);
    await openConversation(cid);
  };

  const sessionsChanged = useCallback(() => {
    refreshSessions();
    setRefreshKey((k) => k + 1);
  }, [refreshSessions]);

  // ---------------------------------------------------------------- chat ---
  const send = async (text: string) => {
    setStreaming(true);
    setItems((current) => [...current, { kind: "user", text, ts: Date.now() / 1000 }]);

    const controller = new AbortController();
    abortRef.current = controller;

    const apply = (event: AgentEvent) => {
      setItems((current) => {
        const next = [...current];
        switch (event.type) {
          case "start":
            liveIds.current = {
              session: event.session_id,
              conversation: event.conversation_id,
            };
            if (!conversationId) setConversationId(event.conversation_id);
            if (!sessionId) setSessionId(event.session_id);
            next.push({ kind: "status", text: "Preparando el contexto...", ts: event.ts });
            break;
          case "status":
            for (let i = next.length - 1; i >= 0; i--) {
              if (next[i].kind === "status") {
                next[i] = { kind: "status", text: event.message, ts: event.ts };
                return next;
              }
            }
            next.push({ kind: "status", text: event.message, ts: event.ts });
            break;
          case "thinking":
            next.push({ kind: "thinking", text: event.text, ts: event.ts });
            break;
          case "assistant_partial":
            next.push({ kind: "assistant", text: event.content, ts: event.ts });
            break;
          case "tool_call":
            next.push({
              kind: "tool",
              callId: event.call_id,
              seq: event.seq,
              tool: event.tool,
              args: event.arguments,
              server: event.server,
              status: "running",
              ts: event.ts,
            });
            break;
          case "tool_result": {
            const index = next.findIndex((x) => x.kind === "tool" && x.callId === event.call_id);
            if (index >= 0) {
              next[index] = {
                ...(next[index] as any),
                status: event.ok ? "ok" : "error",
                result: event.text,
                truncated: !!event.truncated,
                resultChars: event.result_chars,
                mlflowArtifact: event.mlflow_artifact,
                structured: event.structured,
                error: event.error,
                latencyMs: event.latency_ms,
                frames: event.frames,
              };
            }
            break;
          }
          case "final": {
            const filtered = next.filter((x) => x.kind !== "status");
            if (event.content) {
              filtered.push({
                kind: "assistant",
                text: event.content,
                ts: event.ts,
                interactionId: event.interaction_id,
                metrics: event.metrics,
              });
            }
            return filtered;
          }
          case "error":
            next.push({ kind: "error", text: event.message, ts: event.ts });
            break;
        }
        return next;
      });
    };

    try {
      await streamChat(
        {
          message: text,
          session_id: sessionId,
          conversation_id: conversationId,
          mcp_conn_ids: selectedConns,
          provider,
          base_url: baseUrl,
          model,
          mlflow_experiment: experiment,
          api_key: apiKeys[provider] || null,
          temperature,
          thinking,
          system_prompt: systemPrompt,
          max_iterations: maxIterations,
        },
        apply,
        controller.signal,
      );
    } catch (e: any) {
      if (e.name !== "AbortError") {
        setItems((current) => [
          ...current.filter((x) => x.kind !== "status"),
          { kind: "error", text: e.message, ts: Date.now() / 1000 },
        ]);
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
      setRefreshKey((k) => k + 1);
      // La sesion puede haberla creado el backend en este turno, asi que se
      // refresca con el id que llego por el stream, no con el del estado.
      const activeSession = liveIds.current.session || sessionId;
      if (activeSession) {
        loadConversations(activeSession);
        refreshSessions();
      }
    }
  };

  const stop = () => {
    abortRef.current?.abort();
    setStreaming(false);
  };

  const toolsCount = connections
    .filter((c) => selectedConns.includes(c.conn_id))
    .reduce((total, c) => total + c.tools.length, 0);

  // ---------------------------------------------------------------- render -
  return (
    <div className="app">
      <TopBar
        health={health}
        config={config}
        models={models}
        model={model}
        onModelChange={setModel}
        onRefresh={loadSystem}
        busy={busy}
        onOpenModelTab={() => setTab("modelo")}
        mode={mode}
        onModeChange={setMode}
      />

      {globalError && <div className="error-box" style={{ margin: 12 }}>{globalError}</div>}

      <div className="body">
        <Sidebar
          sessions={sessions}
          currentSessionId={sessionId}
          onSelectSession={selectSession}
          onNewSession={newSession}
          onDeleteSession={deleteSession}
          onRenameSession={renameSession}
          onPruneSessions={pruneSessions}
          experiment={experiment}
          experiments={experiments}
          onExperimentChange={setExperiment}
          onExperimentApply={applyExperiment}
          mlflowAvailable={!!config?.mlflow?.available}
          conversations={conversations}
          currentConversationId={conversationId}
          onSelectConversation={openConversation}
          onNewConversation={newConversation}
          onDeleteConversation={deleteConversation}
        />

        <div className="center-slot" hidden={mode !== "chat"}>
          <Chat
            items={items}
            streaming={streaming}
            onSend={send}
            onStop={stop}
            onReset={resetConversation}
            toolsCount={toolsCount}
            serversCount={selectedConns.length}
            conversationId={conversationId}
          />
        </div>

        <div className="center-slot" hidden={mode !== "eval"}>
          <EvalWorkbench
            active={mode === "eval"}
            config={config}
            agent={{ provider, baseUrl, model, temperature, thinking, maxIterations, systemPrompt }}
            agentModels={models}
            apiKeys={apiKeys}
            mcp={{ connIds: selectedConns, servers: selectedConns.length, tools: toolsCount }}
            experiment={experiment}
            experiments={experiments}
            onFocusConversation={focusConversation}
            onOpenInChat={openInChat}
            onSessionsChanged={sessionsChanged}
          />
        </div>

        <aside className="inspector">
          <div className="tabs">
            {(["mcp", "trazas", "metricas", "modelo"] as Tab[]).map((t) => (
              <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
                {t === "mcp" ? "Servidor MCP" : t[0].toUpperCase() + t.slice(1)}
              </button>
            ))}
          </div>

          {tab === "mcp" && (
            <McpPanel
              connections={connections}
              selected={selectedConns}
              onToggle={(id) =>
                setSelectedConns((current) =>
                  current.includes(id) ? current.filter((x) => x !== id) : [...current, id],
                )
              }
              onConnected={refreshConnections}
              onDisconnect={disconnect}
            />
          )}

          {tab === "trazas" && (
            <TracesPanel
              sessionId={sessionId}
              conversationId={conversationId}
              refreshKey={refreshKey}
              mlflowUrl={config?.mlflow_url || ""}
            />
          )}

          {tab === "metricas" && (
            <MetricsPanel
              sessionId={sessionId}
              refreshKey={refreshKey}
              mlflow={config?.mlflow}
              mlflowUrl={
                experiments.find((e) => e.name === experiment)?.url || config?.mlflow_url || ""
              }
              experiment={experiment}
            />
          )}

          {tab === "modelo" && (
            <ModelPanel
              config={config}
              health={health}
              models={models}
              provider={provider}
              onProviderChange={setProvider}
              baseUrl={baseUrl}
              onBaseUrlChange={setBaseUrl}
              model={model}
              onModelChange={setModel}
              apiKeys={apiKeys}
              onApiKeyChange={(name, key) => setApiKeys((c) => ({ ...c, [name]: key }))}
              temperature={temperature}
              onTemperatureChange={setTemperature}
              thinking={thinking}
              onThinkingChange={setThinking}
              maxIterations={maxIterations}
              onMaxIterationsChange={setMaxIterations}
              systemPrompt={systemPrompt}
              onSystemPromptChange={setSystemPrompt}
              onModelsRefreshed={(list) => {
                setModels(list);
                api.get("/api/health").then(setHealth).catch(() => undefined);
              }}
            />
          )}
        </aside>
      </div>
    </div>
  );
}
