import type { AgentEvent, EvalEvent } from "./types";

const STORAGE_KEY = "agente-pruebas:backend-url";

/**
 * URL del backend. Por defecto apunta al mismo host desde el que se sirve la
 * UI pero al puerto 8090, que es el caso habitual cuando frontend y backend
 * viven en la misma maquina de la red. Es editable desde la barra superior
 * para el caso de maquinas distintas.
 */
function defaultBackendUrl(): string {
  if (typeof window === "undefined") return "http://127.0.0.1:8090";
  const { protocol, hostname } = window.location;
  return `${protocol}//${hostname}:8090`;
}

export function getBackendUrl(): string {
  const stored = localStorage.getItem(STORAGE_KEY);
  return (stored || defaultBackendUrl()).replace(/\/+$/, "");
}

export function setBackendUrl(url: string): void {
  localStorage.setItem(STORAGE_KEY, url.replace(/\/+$/, ""));
}

const KEYS_STORAGE = "agente-pruebas:api-keys";

/**
 * Claves de API por proveedor.
 *
 * Se guardan en localStorage del navegador para no tener que repegarlas en
 * cada sesion. Es comodo pero no es un almacen de secretos: cualquier script
 * que corra en este origen puede leerlas. Si la clave es sensible, dejala
 * vacia aqui y ponla en `backend/.env` (ANTHROPIC_API_KEY, OPENAI_API_KEY,
 * GOOGLE_API_KEY); el backend la usa cuando la UI no manda ninguna.
 */
export function loadApiKeys(): Record<string, string> {
  try {
    return JSON.parse(localStorage.getItem(KEYS_STORAGE) || "{}");
  } catch {
    return {};
  }
}

export function saveApiKeys(keys: Record<string, string>): void {
  try {
    localStorage.setItem(KEYS_STORAGE, JSON.stringify(keys));
  } catch {
    /* modo privado o almacenamiento lleno: se sigue sin persistir */
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${getBackendUrl()}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* respuesta sin cuerpo JSON */
    }
    throw new Error(detail);
  }
  return response.status === 204 ? (undefined as T) : ((await response.json()) as T);
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) }),
  patch: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "PATCH", body: JSON.stringify(body) }),
  del: <T>(path: string) => request<T>(path, { method: "DELETE" }),
};

export type ChatPayload = {
  message: string;
  session_id?: string | null;
  conversation_id?: string | null;
  mcp_conn_ids: string[];
  provider?: string | null;
  base_url?: string | null;
  model?: string | null;
  temperature?: number | null;
  max_tokens?: number | null;
  thinking?: boolean | null;
  system_prompt?: string;
  max_iterations?: number | null;
  /** Experimento de MLflow donde registrar este turno; se crea si no existe. */
  mlflow_experiment?: string;
  /** Clave del proveedor de nube; el backend no la persiste. */
  api_key?: string | null;
};

/**
 * Consume un endpoint SSE del backend.
 *
 * EventSource no admite POST, asi que se usa fetch con lectura incremental del
 * cuerpo y un parser minimo del formato SSE (`event:` + `data:` + linea vacia).
 */
export async function streamSse<T>(
  path: string,
  payload: unknown,
  onEvent: (event: T) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${getBackendUrl()}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok || !response.body) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* sin cuerpo */
    }
    throw new Error(detail);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    // sse-starlette termina las lineas con CRLF: se normaliza antes de partir
    // por la linea en blanco que separa eventos.
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const chunk = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const dataLines = chunk
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart());
      if (dataLines.length) {
        try {
          onEvent(JSON.parse(dataLines.join("\n")) as T);
        } catch {
          /* trozo no parseable: se ignora en vez de romper el stream */
        }
      }
      boundary = buffer.indexOf("\n\n");
    }
  }
}

export const streamChat = (
  payload: ChatPayload,
  onEvent: (event: AgentEvent) => void,
  signal?: AbortSignal,
) => streamSse<AgentEvent>("/api/chat", payload, onEvent, signal);

export type PullEvent = {
  type: "progress" | "done" | "error";
  model?: string;
  status?: string;
  completed?: number;
  total?: number;
  percent?: number | null;
  installed?: boolean;
  models?: string[];
  message?: string;
};

/**
 * Sigue en vivo una evaluacion.
 *
 * Aqui si vale EventSource (el endpoint es GET): reconecta solo y reenvia
 * Last-Event-ID, asi que un corte de red no pierde eventos. Hay que cerrarlo a
 * mano al recibir `run_end`, o seguiria reconectando.
 */
export function followEval(
  evalRunId: string,
  since: number,
  onEvent: (event: EvalEvent) => void,
  onError?: () => void,
): () => void {
  const source = new EventSource(`${getBackendUrl()}/api/eval/runs/${evalRunId}/events?since=${since}`);
  const kinds: EvalEvent["type"][] = [
    "run_start",
    "log",
    "item_start",
    "item_phase",
    "turn_user",
    "agent_tool_call",
    "agent_tool_result",
    "turn_agent",
    "item_end",
    "run_end",
  ];
  for (const kind of kinds) {
    source.addEventListener(kind, (message) => {
      try {
        const event = JSON.parse((message as MessageEvent).data) as EvalEvent;
        onEvent(event);
        if (event.type === "run_end") source.close();
      } catch {
        /* evento no parseable: se ignora */
      }
    });
  }
  source.onerror = () => {
    // CLOSED = el servidor respondio con error (p.ej. 404 tras reiniciar el
    // backend): no tiene sentido seguir intentandolo.
    if (source.readyState === EventSource.CLOSED) onError?.();
  };
  return () => source.close();
}

/** Lanza la descarga de un modelo en el motor local y sigue su progreso. */
export const pullModel = (
  payload: { model: string; provider?: string | null; base_url?: string | null },
  onEvent: (event: PullEvent) => void,
  signal?: AbortSignal,
) => streamSse<PullEvent>("/api/llm/pull", payload, onEvent, signal);
