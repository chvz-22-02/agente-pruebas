export type ToolSpec = { name: string; description: string; input_schema: any };

export type McpConnection = {
  conn_id: string;
  config: { url: string; name: string; transport: string; headers: Record<string, string> };
  alive: boolean;
  transport: string | null;
  connected_at: number | null;
  server_info: {
    name?: string;
    title?: string;
    version?: string;
    protocol_version?: string;
    instructions?: string;
    capabilities?: any;
  };
  last_error: string | null;
  tools: ToolSpec[];
};

export type Frame = {
  direction: "agent->mcp" | "mcp->agent";
  payload: any;
  ts: number;
};

export type ServerRef = {
  conn_id: string;
  url: string;
  server_name: string;
  real_tool: string;
};

/** Evento SSE emitido por el backend durante una interaccion. */
export type AgentEvent = {
  type:
    | "start"
    | "status"
    | "thinking"
    | "assistant_partial"
    | "llm_usage"
    | "tool_call"
    | "tool_result"
    | "final"
    | "error";
  ts: number;
  session_id: string;
  conversation_id: string;
  interaction_id: string;
  [key: string]: any;
};

export type ChatItem =
  | { kind: "user"; text: string; ts: number }
  | { kind: "assistant"; text: string; ts: number; interactionId?: string; metrics?: any }
  | { kind: "thinking"; text: string; ts: number }
  | {
      kind: "tool";
      callId: string;
      seq: number;
      tool: string;
      args: any;
      server?: ServerRef;
      status: "running" | "ok" | "error";
      result?: string;
      /** El texto de la UI viene recortado; el integro esta en MLflow/SQLite. */
      truncated?: boolean;
      resultChars?: number;
      mlflowArtifact?: string;
      structured?: any;
      error?: string | null;
      latencyMs?: number;
      frames?: Frame[];
      ts: number;
    }
  | { kind: "status"; text: string; ts: number }
  | { kind: "error"; text: string; ts: number };

export type Session = {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  conversations: number;
  interactions: number;
  total_tokens: number;
  /** Experimento de MLflow donde se registra. Vacio = el del backend. */
  mlflow_experiment?: string;
  /** `kind: "evaluation"` en las sesiones creadas por una evaluacion. */
  metadata?: { kind?: string; eval_run_id?: string; [key: string]: any };
};

export type MlflowExperiment = {
  experiment_id: string;
  name: string;
  url: string;
  last_update_time?: number | null;
};

export type Conversation = {
  id: string;
  session_id: string;
  title: string;
  created_at: number;
  updated_at: number;
  provider: string;
  model: string;
  interactions?: number;
};

export type Interaction = {
  id: string;
  conversation_id: string;
  session_id: string;
  user_message: string;
  final_answer: string;
  status: string;
  error: string;
  started_at: number;
  latency_ms: number;
  llm_latency_ms: number;
  mcp_latency_ms: number;
  llm_calls: number;
  tool_calls: number;
  tool_errors: number;
  iterations: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  model: string;
  provider: string;
  mlflow_trace_id: string;
  mlflow_run_id: string;
};

export type ToolEvent = {
  id: string;
  interaction_id: string;
  conversation_id: string;
  session_id: string;
  seq: number;
  iteration: number;
  tool_name: string;
  real_tool_name: string;
  server_name: string;
  server_url: string;
  arguments: any;
  result_text: string;
  structured: any;
  ok: boolean;
  error: string;
  latency_ms: number;
  frames: Frame[];
  created_at: number;
};

export type BackendConfig = {
  providers: string[];
  defaults: {
    provider: string;
    base_url: string;
    model: string;
    temperature: number;
    max_tokens: number;
    num_ctx: number;
    thinking: boolean;
    max_iterations: number;
  };
  mlflow: {
    enabled: boolean;
    available: boolean;
    status: string;
    tracing_supported: boolean;
    experiment: string;
  };
  mlflow_url: string;
  /** Experimentos ya existentes en el servidor de MLflow. */
  mlflow_experiments: MlflowExperiment[];
  /** Proveedores que saben descargar modelos por su propia API (Ollama). */
  pull_capable: string[];
  suggested_models: { name: string; size: string; note: string }[];
  catalog: ProviderInfo[];
};

/** Respuesta de GET /api/llm/features: que sabe hacer el modelo elegido. */
export type ModelFeatures = {
  ok: boolean;
  provider?: string;
  model?: string;
  in_catalog?: boolean;
  thinking?: "adaptive" | "effort" | "toggle" | "always" | "none" | "unknown";
  thinking_supported?: boolean;
  can_disable_thinking?: boolean;
  tools?: boolean;
  capabilities?: string[];
  /** "engine" = preguntado al motor; "catalog" = del catalogo estatico. */
  source?: "engine" | "catalog" | "desconocido";
  error?: string;
};

export type CatalogModel = {
  id: string;
  label: string;
  context: string;
  notes: string;
  /** false = enviar temperature devuelve error (Claude 4.7+, GPT-5). */
  sampling: boolean;
  thinking: "adaptive" | "effort" | "toggle" | "always" | "none";
  recommended: boolean;
};

export type ProviderInfo = {
  name: string;
  label: string;
  kind: "local" | "cloud";
  needs_api_key: boolean;
  default_base_url: string;
  key_env: string;
  key_hint: string;
  console_url: string;
  /** Variable con el identificador de cuenta si la URL lo lleva (Cloudflare). */
  account_env: string;
  supports_pull: boolean;
  models: CatalogModel[];
};

// ------------------------------------------------------------ evaluaciones --

/** Respuesta de POST /api/eval/validate. */
export type EvalValidation = {
  ok: boolean;
  errors: string[];
  warnings: string[];
  personas: { id: string; nombre: string; descripcion: string }[];
  cases: {
    id: string;
    persona: string;
    goal: string;
    consulta_inicial: string;
    ambiguedad: string;
    resultado_esperado: string;
    valor_esperado: string;
  }[];
  matrix: number;
};

/** Modelo de un papel de la evaluacion (simulador o evaluador). */
export type RoleConfig = {
  /** true = mismo proveedor, URL y modelo que el agente bajo prueba. */
  sameAsAgent: boolean;
  provider: string;
  base_url: string;
  model: string;
  temperature: number;
  thinking: boolean;
};

export type EvalStatus = "pending" | "running" | "passed" | "failed" | "error" | "cancelled" | "interrupted";

/** Un criterio evaluado (del evaluador LLM o una comprobacion determinista). */
export type EvalItem = {
  id: string;
  origen: "evaluador" | "determinista";
  descripcion: string;
  cumple: boolean;
  detalle: string;
  peso: number;
  obligatorio: boolean;
  sin_juicio?: boolean;
  valor_reportado?: string | null;
};

export type EvalToolView = {
  call_id?: string;
  tool: string;
  arguments: any;
  ok?: boolean;
  error?: string | null;
  latency_ms?: number;
  preview?: string;
  result_chars?: number;
};

export type EvalTurn = {
  turn: number;
  role: "user" | "agent";
  text: string;
  source?: "simulador" | "guion";
  closing?: boolean;
  error?: string;
  tools: EvalToolView[];
  metrics?: any;
  interaction_id?: string;
};

/** Vista unificada de un caso x persona, en vivo o ya persistido. */
export type EvalResultView = {
  result_id: string;
  seq: number;
  case_id: string;
  case_title: string;
  persona_id: string;
  persona_name: string;
  repetition: number;
  status: EvalStatus;
  phase?: string;
  turn?: number;
  conversation_id?: string;
  score: number | null;
  passed: boolean | null;
  turns: number;
  end_reason: string;
  error: string;
  resumen?: string;
  items?: EvalItem[];
  failed_mandatory?: string[];
  transcript: EvalTurn[];
  metrics?: Record<string, number>;
  mlflow_run_id?: string;
};

export type EvalRunRow = {
  id: string;
  session_id: string;
  name: string;
  suite: string;
  status: "running" | "done" | "cancelled" | "error" | "interrupted";
  created_at: number;
  ended_at: number | null;
  summary: any;
  error: string;
  mlflow_experiment: string;
  mlflow_run_id: string;
  live?: boolean;
  config?: any;
};

export type EvalRunDetail = {
  run: EvalRunRow;
  results: any[];
  live: boolean;
  events: number;
  mlflow: {
    experiment: string;
    experiment_url: string;
    run_id: string;
    run_url: string;
    trace_filter: string;
  };
};

/** Evento SSE de GET /api/eval/runs/{id}/events. */
export type EvalEvent = {
  type:
    | "run_start"
    | "log"
    | "item_start"
    | "item_phase"
    | "turn_user"
    | "agent_tool_call"
    | "agent_tool_result"
    | "turn_agent"
    | "item_end"
    | "run_end";
  seq: number;
  ts: number;
  eval_run_id: string;
  [key: string]: any;
};

/** Respuesta de POST /api/llm/probe: valida la clave y lista modelos reales. */
export type ProbeResult = {
  ok: boolean;
  provider?: string;
  model?: string;
  model_available?: boolean;
  needs_api_key?: boolean;
  error?: string;
  models: string[];
  version?: string;
};
