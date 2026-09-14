import { useEffect, useRef, useState } from "react";
import Json from "./Json";
import type { ChatItem, Frame } from "../lib/types";

type Props = {
  items: ChatItem[];
  streaming: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
  onReset: () => void;
  toolsCount: number;
  serversCount: number;
  conversationId: string | null;
};

const ms = (v?: number) => (v === undefined ? "-" : `${Math.round(v)} ms`);

function FrameList({ frames }: { frames: Frame[] }) {
  if (!frames?.length) return null;
  return (
    <details>
      <summary className="muted" style={{ cursor: "pointer", marginTop: 8 }}>
        JSON-RPC ({frames.length} tramas)
      </summary>
      {frames.map((f, i) => (
        <div key={i} className={`frame ${f.direction === "agent->mcp" ? "out" : "in"}`}>
          <div className="dir">{f.direction === "agent->mcp" ? "AGENTE -> MCP" : "MCP -> AGENTE"}</div>
          <Json value={f.payload} max={6000} />
        </div>
      ))}
    </details>
  );
}

function ToolBlock({ item }: { item: Extract<ChatItem, { kind: "tool" }> }) {
  return (
    <details className={`block tool ${item.status === "error" ? "err" : ""}`}>
      <summary>
        {item.status === "running" ? <span className="spin" /> : <span className={`dot ${item.status === "ok" ? "ok" : "err"}`} />}
        <span className="name">{item.tool}</span>
        {item.server?.server_name && <span className="badge">{item.server.server_name}</span>}
        <div style={{ flex: 1 }} />
        <span className="muted">{item.status === "running" ? "ejecutando..." : ms(item.latencyMs)}</span>
      </summary>
      <div className="inner">
        <div className="muted">Argumentos</div>
        <Json value={item.args} />
        {item.error && (
          <>
            <div className="muted" style={{ marginTop: 8, color: "var(--err)" }}>
              Error
            </div>
            <pre style={{ color: "#ffb3ae" }}>{item.error}</pre>
          </>
        )}
        {item.result !== undefined && item.status !== "running" && (
          <>
            <div className="muted" style={{ marginTop: 8 }}>
              Resultado
              {item.resultChars !== undefined && (
                <span> · {item.resultChars.toLocaleString()} caracteres</span>
              )}
            </div>
            <pre>{item.result || "(vacio)"}</pre>
            {item.truncated && (
              <div className="muted">
                Extracto para la UI. El resultado completo esta en la pestana{" "}
                <strong>Trazas</strong> y en MLflow
                {item.mlflowArtifact && (
                  <>
                    {" "}
                    (artifact <code>{item.mlflowArtifact}</code>)
                  </>
                )}
                .
              </div>
            )}
          </>
        )}
        {item.structured != null && (
          <>
            <div className="muted" style={{ marginTop: 8 }}>
              structuredContent
            </div>
            <Json value={item.structured} />
          </>
        )}
        {item.server?.url && (
          <div className="muted" style={{ marginTop: 8 }}>
            {item.server.url} · herramienta real: <code>{item.server.real_tool}</code>
          </div>
        )}
        <FrameList frames={item.frames || []} />
      </div>
    </details>
  );
}

export default function Chat({
  items,
  streaming,
  onSend,
  onStop,
  onReset,
  toolsCount,
  serversCount,
  conversationId,
}: Props) {
  const [text, setText] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [items.length, streaming]);

  const submit = () => {
    const value = text.trim();
    if (!value || streaming) return;
    onSend(value);
    setText("");
  };

  return (
    <section className="center">
      <div className="panel-head">
        Conversacion
        {conversationId && <span className="badge">{conversationId}</span>}
        <div style={{ flex: 1 }} />
        <span className="muted">
          {serversCount} MCP · {toolsCount} herramientas
        </span>
        <button className="tiny" onClick={onReset} disabled={streaming} title="Empieza un hilo nuevo sin memoria previa">
          Reiniciar conversacion
        </button>
      </div>

      <div className="chat-scroll">
        <div className="chat-inner">
          {items.length === 0 && (
            <p className="muted">
              Conecta un servidor MCP en el panel derecho y escribe un mensaje. El agente recuerda todo el hilo
              hasta que pulses <strong>Reiniciar conversacion</strong>.
            </p>
          )}

          {items.map((item, index) => {
            switch (item.kind) {
              case "user":
                return (
                  <div key={index} className="msg user">
                    <div className="who">Tu</div>
                    <div className="bubble">{item.text}</div>
                  </div>
                );
              case "assistant":
                return (
                  <div key={index} className="msg assistant">
                    <div className="who">Agente</div>
                    <div className="bubble">{item.text || "(respuesta vacia)"}</div>
                    {item.metrics && (
                      <div className="metrics">
                        <span>{ms(item.metrics.latency_ms)} total</span>
                        <span>{ms(item.metrics.llm_latency_ms)} LLM</span>
                        <span>{ms(item.metrics.mcp_latency_ms)} MCP</span>
                        <span>{item.metrics.total_tokens} tokens</span>
                        <span>{item.metrics.tool_calls} tool calls</span>
                        <span>{(item.metrics.output_tokens_per_second || 0).toFixed(1)} tok/s</span>
                      </div>
                    )}
                  </div>
                );
              case "thinking":
                return (
                  <details key={index} className="block thinking">
                    <summary>Razonamiento del modelo</summary>
                    <div className="inner">
                      <pre>{item.text}</pre>
                    </div>
                  </details>
                );
              case "tool":
                return <ToolBlock key={index} item={item} />;
              case "status":
                return (
                  <p key={index} className="muted" style={{ margin: "6px 0" }}>
                    {streaming && <span className="spin" style={{ display: "inline-block", marginRight: 6 }} />}
                    {item.text}
                  </p>
                );
              case "error":
                return (
                  <div key={index} className="msg error">
                    <div className="who">Error</div>
                    <div className="bubble">{item.text}</div>
                  </div>
                );
              default:
                return null;
            }
          })}
          <div ref={endRef} />
        </div>
      </div>

      <div className="composer">
        <div className="composer-inner">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder="Escribe un mensaje. Enter envia, Shift+Enter salta de linea."
            disabled={streaming}
          />
          {streaming ? (
            <button className="danger" onClick={onStop}>
              Detener
            </button>
          ) : (
            <button className="primary" onClick={submit} disabled={!text.trim()}>
              Enviar
            </button>
          )}
        </div>
        <div className="composer-hint">
          <span>La memoria se mantiene entre mensajes de esta conversacion.</span>
        </div>
      </div>
    </section>
  );
}
