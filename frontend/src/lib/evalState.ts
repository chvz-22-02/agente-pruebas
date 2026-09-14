import type { EvalEvent, EvalResultView, EvalStatus, EvalTurn } from "./types";

/**
 * Estado de una evaluacion en la UI.
 *
 * Mientras corre se construye a partir de los eventos SSE; cuando un caso
 * termina se sustituye por lo persistido en SQLite, que trae la rubrica
 * evaluada completa. Las dos fuentes acaban en la misma forma
 * (`EvalResultView`) para que la vista no tenga que distinguirlas.
 */

export const FINISHED: EvalStatus[] = ["passed", "failed", "error", "cancelled", "interrupted"];

/** Resultado tal y como lo devuelve GET /api/eval/runs/{id}. */
export function fromPersisted(row: any): EvalResultView {
  const transcript: EvalTurn[] = (row.transcript || []).map((t: any) => ({
    turn: t.turn,
    role: t.role,
    text: t.text || "",
    source: t.source,
    closing: t.closing,
    error: t.error || "",
    metrics: t.metrics,
    interaction_id: t.interaction_id,
    tools: (t.tool_calls || []).map((c: any) => ({
      tool: c.tool,
      arguments: c.arguments,
      ok: c.ok,
      error: c.error,
      latency_ms: c.latency_ms,
      preview: (c.result || "").slice(0, 1500),
      result_chars: c.result_chars,
    })),
  }));
  return {
    result_id: row.id,
    seq: row.seq,
    case_id: row.case_id,
    case_title: row.case_title,
    persona_id: row.persona_id,
    persona_name: row.persona_name,
    repetition: row.repetition,
    status: row.status,
    conversation_id: row.conversation_id || undefined,
    score: row.score ?? null,
    passed: row.passed ?? null,
    turns: row.turns || 0,
    end_reason: row.end_reason || "",
    error: row.error || "",
    resumen: row.verdict?.resumen || "",
    items: row.items || [],
    failed_mandatory: row.verdict?.aggregate?.failed_mandatory || [],
    transcript,
    metrics: row.metrics || {},
    mlflow_run_id: row.mlflow_run_id || "",
  };
}

function blank(ref: any): EvalResultView {
  return {
    result_id: ref.result_id,
    seq: ref.seq,
    case_id: ref.case_id,
    case_title: ref.case_title,
    persona_id: ref.persona_id,
    persona_name: ref.persona_name,
    repetition: ref.repetition,
    status: "pending",
    score: null,
    passed: null,
    turns: 0,
    end_reason: "",
    error: "",
    transcript: [],
  };
}

/** Aplica un evento SSE. Devuelve un array nuevo (inmutable para React). */
export function applyEvent(results: EvalResultView[], event: EvalEvent): EvalResultView[] {
  if (event.type === "run_start") {
    const known = new Map(results.map((r) => [r.result_id, r]));
    return (event.items || []).map((ref: any) => known.get(ref.result_id) || blank(ref));
  }
  if (!event.result_id) return results;

  const index = results.findIndex((r) => r.result_id === event.result_id);
  if (index < 0) return results;
  const current = results[index];
  // Un caso ya cerrado no se vuelve a tocar: al reengancharse al flujo se
  // repiten desde el principio eventos de casos que ya vienen de SQLite, con
  // su rubrica completa, y no deben pisarse con la version resumida.
  if (FINISHED.includes(current.status)) return results;

  const next = { ...current, transcript: [...current.transcript] };
  const lastTurn = (role: "user" | "agent", turn: number) =>
    next.transcript.findIndex((t) => t.role === role && t.turn === turn);

  switch (event.type) {
    case "item_start":
      next.status = "running";
      next.conversation_id = event.conversation_id;
      break;
    case "item_phase":
      next.status = "running";
      next.phase = event.phase;
      next.turn = event.turn;
      break;
    case "turn_user":
      next.transcript.push({
        turn: event.turn,
        role: "user",
        text: event.text,
        source: event.source,
        closing: event.closing,
        tools: [],
      });
      break;
    case "agent_tool_call": {
      let at = lastTurn("agent", event.turn);
      if (at < 0) {
        next.transcript.push({ turn: event.turn, role: "agent", text: "", tools: [] });
        at = next.transcript.length - 1;
      }
      const turn = { ...next.transcript[at], tools: [...next.transcript[at].tools] };
      turn.tools.push({ call_id: event.call_id, tool: event.tool, arguments: event.arguments });
      next.transcript[at] = turn;
      break;
    }
    case "agent_tool_result": {
      const at = lastTurn("agent", event.turn);
      if (at < 0) break;
      const turn = { ...next.transcript[at] };
      turn.tools = turn.tools.map((t) =>
        t.call_id === event.call_id
          ? { ...t, ok: event.ok, error: event.error, latency_ms: event.latency_ms, preview: event.preview }
          : t,
      );
      next.transcript[at] = turn;
      break;
    }
    case "turn_agent": {
      const at = lastTurn("agent", event.turn);
      const base = at >= 0 ? next.transcript[at] : { turn: event.turn, role: "agent" as const, text: "", tools: [] };
      const turn = {
        ...base,
        text: event.text,
        error: event.error,
        metrics: event.metrics,
        interaction_id: event.interaction_id,
      };
      if (at >= 0) next.transcript[at] = turn;
      else next.transcript.push(turn);
      next.turns = Math.max(next.turns, event.turn);
      break;
    }
    case "item_end":
      next.status = event.status;
      next.phase = undefined;
      next.score = event.score ?? null;
      next.passed = event.passed ?? null;
      next.turns = event.turns ?? next.turns;
      next.end_reason = event.end_reason || "";
      next.resumen = event.resumen || "";
      next.error = event.error || "";
      next.failed_mandatory = event.failed_mandatory || [];
      next.metrics = event.metrics || {};
      next.mlflow_run_id = event.mlflow_run_id || "";
      break;
  }

  const copy = [...results];
  copy[index] = next;
  return copy;
}

/** Combina lo persistido con lo que llega en vivo: lo cerrado manda SQLite. */
export function mergePersisted(live: EvalResultView[], persisted: any[]): EvalResultView[] {
  const stored = new Map(persisted.map((row) => [row.id, row]));
  if (live.length === 0) return persisted.map(fromPersisted);
  return live.map((item) => {
    const row = stored.get(item.result_id);
    return row && FINISHED.includes(row.status) ? fromPersisted(row) : item;
  });
}

/** Agregados calculados en el cliente, para verlos crecer mientras corre. */
export function summarize(results: EvalResultView[]) {
  const judged = results.filter((r) => r.status === "passed" || r.status === "failed");
  const passed = judged.filter((r) => r.passed).length;
  const scores = judged.map((r) => r.score ?? 0);
  const sum = (key: string) => results.reduce((acc, r) => acc + (Number(r.metrics?.[key]) || 0), 0);
  const group = (key: "case_id" | "persona_id") => {
    const out = new Map<string, { label: string; runs: number; passed: number; scores: number[] }>();
    for (const r of judged) {
      const id = r[key];
      const label = key === "case_id" ? r.case_title || id : r.persona_name || id;
      const bucket = out.get(id) || { label, runs: 0, passed: 0, scores: [] };
      bucket.runs += 1;
      bucket.passed += r.passed ? 1 : 0;
      bucket.scores.push(r.score ?? 0);
      out.set(id, bucket);
    }
    return [...out.entries()].map(([id, b]) => ({
      id,
      label: b.label,
      runs: b.runs,
      passed: b.passed,
      avg: b.scores.reduce((a, s) => a + s, 0) / (b.scores.length || 1),
    }));
  };
  return {
    total: results.length,
    finished: results.filter((r) => FINISHED.includes(r.status)).length,
    judged: judged.length,
    passed,
    errors: results.filter((r) => r.status === "error").length,
    passRate: judged.length ? passed / judged.length : 0,
    avgScore: scores.length ? scores.reduce((a, s) => a + s, 0) / scores.length : 0,
    tokens: {
      agent: sum("agent_total_tokens"),
      simulator: sum("sim_total_tokens"),
      judge: sum("judge_total_tokens"),
    },
    toolCalls: sum("tool_calls"),
    byCase: group("case_id"),
    byPersona: group("persona_id"),
  };
}

export const STATUS_LABEL: Record<string, string> = {
  pending: "pendiente",
  running: "en curso",
  passed: "aprobado",
  failed: "suspendido",
  error: "error",
  cancelled: "cancelado",
  interrupted: "interrumpido",
  done: "terminada",
};

export const PHASE_LABEL: Record<string, string> = {
  simulando: "la persona escribe",
  agente: "el agente responde",
  evaluando: "evaluando",
};

export const END_REASON_LABEL: Record<string, string> = {
  persona_termina: "la persona cerro",
  max_turnos: "limite de turnos",
  error_agente: "error del agente",
  cancelada: "cancelada",
  error: "error",
};
