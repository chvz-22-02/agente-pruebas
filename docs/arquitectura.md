# Arquitectura

## Flujo de una interacción

```
UI ──POST /api/chat (SSE)──► routes_chat
                                │  resuelve sesión + conversación
                                ▼
                            AgentRunner.run()
                                │
                ┌───────────────┴──────────────────────────────┐
                │  bucle, hasta AGENT_MAX_ITERATIONS           │
                │                                              │
                │  1. historial (SQLite) + system prompt       │
                │  2. LLMProvider.chat(messages, tools)        │
                │        └─ span LLM en MLflow                 │
                │  3. ¿tool_calls?                             │
                │       sí  → ToolRouter.call()                │
                │              └─ MCPConnection (task propia)  │
                │                   └─ tee JSON-RPC            │
                │              └─ span TOOL en MLflow          │
                │              → observación al historial      │
                │              → volver a 2                    │
                │       no  → respuesta final, salir           │
                └──────────────────────────────────────────────┘
                                │
                    SQLite + MLflow + eventos SSE a la UI
```

Cada paso emite un evento SSE (`start`, `status`, `thinking`, `llm_usage`, `tool_call`,
`tool_result`, `assistant_partial`, `final`, `error`), así que la UI muestra el progreso en
vivo aunque el modelo tarde en CPU.

## Por qué una task por conexión MCP

El SDK de MCP usa *task groups* de anyio. Un `ClientSession` abierto dentro de una petición
HTTP moriría al terminar esa petición, y una sesión no puede cruzar fronteras de tarea sin
romper los *cancel scopes*.

`MCPConnection` resuelve esto con una task supervisora de larga vida:

```
MCPConnection.start()
  └── asyncio.create_task(_run)
        └── async with transporte:            ← streamable_http | sse
              └── async with Client(...):     ← handshake, negociación de versión
                    ├── future `ready` resuelto  → start() puede devolver
                    └── _command_loop()          ← consume la cola indefinidamente

Cualquier petición HTTP  ──(op, payload, future)──►  cola  ──►  _command_loop  ──► future
```

Ventajas: la sesión sobrevive entre peticiones, conserva el `Mcp-Session-Id` y la caché del
cliente, y se cierra de forma limpia. Contrapartida: las operaciones contra un mismo
servidor se serializan.

## Captura del JSON-RPC

`Client` acepta cualquier objeto que cumpla el protocolo `Transport` (un context manager que
produce un par de streams). Se abre el transporte real, se envuelven sus streams con
`TeeReceiveStream` / `TeeSendStream` y se le pasan a `Client` ya envueltos mediante
`_PreopenedTransport`.

```
transporte real ──► TeeSendStream    ──► Client  (agente → MCP)
                └── TeeReceiveStream ◄──          (MCP → agente)
                         │
                         └──► buffer circular (500 tramas) + colectores por llamada
```

Los *tee* delegan todo lo que no interceptan vía `__getattr__`, y el volcado a dict va
dentro de un `try/except` amplio: la observabilidad nunca puede romper el transporte.

## Modelo de datos

```
sessions ──1:N──► conversations ──1:N──► interactions ──1:N──► tool_events
                        │                                          │
                        └──1:N──► messages                    frames (JSON-RPC)
```

* `messages` es la **memoria** del agente: se relee entera en cada turno.
* `interactions` es la unidad de **medida**: una fila por turno con todas sus métricas.
* `tool_events` es el **detalle** de cada llamada al MCP, con sus tramas.

`interaction_id` está presente en `messages` y en `tool_events`, así que se puede
reconstruir exactamente qué hizo el agente en un turno concreto.

Las evaluaciones cuelgan de la misma jerarquía en vez de inventar otra:

```
eval_runs ──1:1──► sessions            (metadata.kind = "evaluation")
    │
    └──1:N──► eval_results ──1:1──► conversations ──► interactions ──► tool_events
              (consulta × repetición: nota, rúbrica evaluada, transcripción, métricas)
```

Así todo lo que ya existía (pestaña *Trazas*, *Métricas*, exportación, borrado en cascada
con purga de MLflow) funciona igual sobre una conversación de evaluación.

## Puntos de extensión

| Quiero… | Toco |
|---|---|
| Otro motor de inferencia | `llm/`: implementar `LLMProvider` + registrar en `PROVIDERS` |
| Otro system prompt | `agent/prompts.py`, o el campo de la pestaña *Modelo* |
| Otras métricas | `RunMetrics.as_mlflow()` en `agent/loop.py` |
| Otro almacén | `store/db.py` y `store/repository.py` |
| Transporte MCP por stdio | `mcpclient/connection.py::_open_transport` (`StdioServerParameters`) |
| Otro endpoint HTTP | `api/routes_*.py` y registrarlo en `main.py` |
| Un campo nuevo en `personas.json` / `consultas.json` | `evals/spec.py` (modelo pydantic) y quien lo use: `simulator.py`, `judge.py` o `checks.py` |
| Otra comprobación determinista | `evals/checks.py::deterministic_checks` (entra sola en la nota con su peso) |
| Otro prompt de simulador o evaluador | `evals/simulator.py::build_system_prompt`, `evals/judge.py::SYSTEM_PROMPT` |

## Decisiones y sus motivos

**SSE en vez de WebSocket.** El flujo es unidireccional (servidor → cliente) y SSE atraviesa
proxies sin configuración extra. Como `EventSource` no admite POST, la UI usa `fetch` con
lectura incremental del cuerpo y un parser SSE mínimo en `lib/api.ts`.

**`start_span_no_context` en MLflow.** La API fluida (`mlflow.start_span`) se apoya en
`contextvars`; con peticiones concurrentes y saltos a hilos el contexto se vuelve frágil.
La variante sin contexto recibe el span padre explícitamente. Las llamadas a MLflow son
HTTP bloqueante, así que van a un hilo con `asyncio.to_thread`.

**MLflow con `MlflowClient` y `run_id` explícito.** La API fluida mantiene un run activo
global, incompatible con varias conversaciones simultáneas.

**Claves estándar de MLflow.** Los tokens se publican como `mlflow.chat.tokenUsage` y el
modelo como `mlflow.llm.model`, de modo que MLflow los agrega y los muestra de forma nativa
en lugar de quedar como atributos sueltos.

**Fail-soft en observabilidad.** Si MLflow no está disponible, el agente funciona igual y lo
reporta en `/api/health`. Un banco de pruebas que se cae porque el servidor de métricas no
está levantado no sirve de mucho.

**Las sesiones se crean bajo demanda.** Una sesión nace al pulsar «+ Sesión» o al enviar el
primer mensaje sin sesión abierta; nada las crea de forma periódica ni al recargar la UI.
Su run de MLflow tampoco se crea al crearla, sino con la primera interacción: una sesión que
se abre y no se usa no deja rastro en el experimento.

**El nombre de la sesión no necesita ser único.** Se edita desde la barra lateral y se
propaga al run de MLflow (`mlflow.runName`), pero nada depende de él: los runs se identifican
por `run_id` y siempre se buscan por `tags.session_id`. Por eso renombrar es seguro y dos
sesiones homónimas no se pisan. El nombre por defecto es un sello corto (`10/09 10:49`).
Ojo con un detalle fácil de perder: el run padre se crea en la primera interacción, así que
hay que pasarle el título de la sesión hasta `ensure_session_run`; sin eso el run se llamaba
`session::ses_<id>`.

**Borrar una sesión la borra también en MLflow.** `DELETE /api/sessions/{id}` limpia SQLite
(conversaciones, mensajes, interacciones y eventos de herramienta) y busca en todos los
experimentos los runs y las trazas con `tags.session_id = <id>` para eliminarlos. Se recorren
todos porque una sesión puede haber registrado en varios si se cambió el experimento por el
camino. Con `?purge_mlflow=false` se conserva el registro. Lo mismo aplica a las
conversaciones con `tags.conversation_id`.

Matiz de MLflow: las trazas se borran de verdad, pero `delete_run` es un borrado lógico —
el run desaparece de la vista normal y queda marcado como eliminado. MLflow no expone borrado
físico por API; para liberar los artifacts del disco hay que pasar `mlflow gc` por el
servidor de tracking.

**El experimento de MLflow se elige desde la UI.** Viaja con cada petición de chat, se guarda
en la columna `sessions.mlflow_experiment` y se crea si el nombre no existe
(`ensure_experiment`). Las cachés de runs se indexan por `(experiment_id, id)`, de modo que
cambiar de experimento a media sesión abre runs nuevos en el destino en vez de reutilizar los
viejos. Lo ya registrado no se mueve.

**Lo que se registra no se trunca.** El recorte solo existe en dos sitios y por motivos
distintos: `MAX_TOOL_RESULT_CHARS` limita lo que se le devuelve al modelo (presupuesto de
contexto) y `MAX_UI_RESULT_CHARS` lo que viaja por SSE (la UI no necesita 200 KB de JSON). En
MLflow el resultado de cada llamada MCP va entero al span **y** como artifact JSON en
`mcp_tool_calls/<interaccion>/<seq>_<tool>.json` del run de la conversación, con argumentos,
`structuredContent` y los frames JSON-RPC en crudo. El span guarda la ruta del artifact en
`result_artifact`, y la tabla `mcp_tool_calls.json` la repite como índice. La respuesta final
del agente también se guarda completa.

**Las capacidades del modelo se preguntan al motor.** El catálogo estático solo cubre los
modelos que venían de fábrica, así que para uno descargado después la fuente de verdad es
`/api/show` de Ollama, que declara `capabilities` (`thinking`, `tools`, `vision`...).
`GET /api/llm/features` lo expone y la UI decide con eso si el interruptor de razonamiento
aplica. Además, a Ollama se le manda `think` explícitamente también cuando vale `false`:
omitirlo no desactiva el razonamiento, porque modelos como Qwen3 razonan por defecto, y el
interruptor solo funcionaría en un sentido.

**Los campos que el proveedor exige de vuelta viajan con la llamada.** Claude pide sus
bloques de razonamiento firmados y Gemini 3 su `thought_signature` por cada `functionCall`.
Claude lo resuelve con `raw_blocks`, que solo vive en memoria porque a él le basta con el
turno en curso; Gemini lo necesita también en turnos posteriores, así que su firma se guarda
en `ToolCall.extra` y se serializa dentro de `tool_calls` — es decir, se persiste en SQLite y
sobrevive a reconstruir el historial. El reenvío está atado a `extra_content_key`, para que
un `extra_content` de Gemini no acabe en una petición a OpenAI si se cambia de modelo a mitad
de conversación.

**Errores de herramienta como observación.** Un `isError` del MCP se le devuelve al modelo
como resultado, no como excepción: así se puede observar cómo reacciona el agente a un MCP
que falla, que es justo uno de los casos que interesa testear.

**El `.env` se carga también en el entorno del proceso.** `Settings` solo mapea los campos que
declara (`LLM_*`, `MLFLOW_*`...), pero las claves de nube se buscan por el nombre que declara el
catálogo (`GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`...) en `os.environ`. Sin un `load_dotenv`
explícito en `config.py`, una clave escrita en `backend/.env` se ignoraba y solo funcionaba la
que se pegase en la UI, justo al revés de lo que promete el `.env.example`.

**Evaluaciones: el agente bajo prueba es el mismo del chat.** El orquestador
(`evals/runner.py`) no reimplementa nada: cada mensaje de la persona simulada entra por
`AgentRunner`, igual que uno escrito a mano, así que se evalúa exactamente el agente que se
usa en el chat, con su captura JSON-RPC y sus trazas. Las trazas llevan etiquetas extra
(`eval_run_id`, `eval_case_id`, `eval_persona_id`, `eval_role`) vía `RunConfig.trace_tags`.

**El evaluador juzga, el código puntúa.** El LLM evaluador solo emite un booleano y una
justificación por criterio; pesos, umbral y criterios obligatorios se aplican en
`checks.aggregate`. Cambiar de modelo evaluador cambia los juicios, nunca la aritmética, y la
nota es reproducible a partir de `evaluation/result.json`.

**El simulador ve la conversación entera en un solo mensaje.** Darle el historial con los roles
invertidos (sus mensajes como `assistant`) es lo obvio, pero en las pruebas con qwen3:4b el
simulador se creía al principio de la conversación y repetía la primera pregunta. Con la
transcripción y el estado explícito («tu mensaje 2 de 4») en un único mensaje no pasa. El cierre
es una marca en texto (`<<FIN>>`), no JSON: los modelos pequeños rompen el JSON con facilidad.

**Razonamiento sin etiqueta de apertura.** Los modelos que razonan siempre (qwen3:4b en Ollama,
variantes `-thinking-2507`) llevan el `<think>` en la plantilla del prompt, así que la salida
trae el razonamiento y solo el `</think>`, incluso con `think=false`. `split_thinking` lo trata;
sin eso el razonamiento aparecía como respuesta del agente y la marca `<<FIN>>` que el simulador
mencionaba al razonar cerraba la conversación.

**Las evaluaciones corren en segundo plano.** En CPU un caso tarda minutos, así que la
ejecución es una `asyncio.Task` desacoplada de la petición HTTP. La UI la sigue por SSE con
`EventSource` (GET, reconexión automática con `Last-Event-ID`) y al reengancharse repite los
eventos desde el principio; lo cerrado se toma de SQLite. Un reinicio del backend marca las
que estaban en marcha como `interrupted`.
