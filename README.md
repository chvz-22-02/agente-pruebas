# Agente de pruebas MCP

Banco de pruebas local para servidores **MCP** (Model Context Protocol): un agente
conversacional con LLM local que se conecta al MCP que le indiques desde la interfaz,
captura toda la interacción agente ↔ MCP (incluidas las tramas JSON-RPC reales), la
muestra en la UI, la persiste y la registra en **MLflow** para poder comparar experimentos.

```
frontend (React/Vite)  ──HTTP+SSE──►  backend (FastAPI)  ──MCP──►  tu servidor MCP
                                            │
                                            ├──►  Ollama / llama.cpp   (LLM local)
                                            ├──►  SQLite              (histórico)
                                            └──►  MLflow              (experimentos)
```

---

## 1. Puesta en marcha

```powershell
# 1) dependencias
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

# 2) motor de inferencia local
#    (el modelo tambien se puede descargar despues desde la propia UI)
winget install Ollama.Ollama
ollama pull qwen3:8b

# 3) arrancar todo (añade -WithDemoMcp para levantar un MCP de ejemplo)
powershell -ExecutionPolicy Bypass -File scripts\start-all.ps1 -WithDemoMcp
```

| Servicio | URL | Qué es |
|---|---|---|
| UI | http://localhost:5173 | La interfaz |
| Backend | http://localhost:8090/docs | API y OpenAPI |
| MLflow | http://localhost:5000 | Experimentos y trazas |
| MCP demo | http://localhost:3333/mcp | Servidor MCP de juguete |

> El backend usa el puerto **8090** porque Docker Desktop ocupa el 8000 en esta máquina.

En la UI: pega el enlace de tu MCP en **Servidor MCP → Enlace del servidor MCP**, pulsa
*Conectar* y escribe. Nada de URLs de MCP en el `.env` ni en el código.

---

## 2. Elección del modelo

La máquina objetivo es un **i7-12700 (12 núcleos / 20 hilos), 32 GB de RAM y sin GPU**.
Sin GPU la inferencia va por CPU: la velocidad la marca el ancho de banda de memoria, así
que un modelo de 7–9 B cuantizado a Q4 es el punto dulce.

**Por defecto: `qwen3:8b`** (~5,2 GB en Q4_K_M)

* Razonamiento explícito (modo *thinking*), que es lo que pide un agente que decide qué
  herramienta llamar.
* Ventana de contexto de 32K nativos, más que suficiente cuando un MCP devuelve respuestas
  largas. Se puede subir con `LLM_NUM_CTX`.
* *Tool calling* nativo y fiable, que es el requisito crítico aquí: un modelo que no
  formatea bien las llamadas a herramientas no sirve para testear un MCP.
* Rendimiento **medido** en este equipo: ~8 tokens/s de generación. Un turno con dos
  llamadas a herramientas y razonamiento activo tarda ~110 s; uno sin herramientas, ~30 s.

Si eso te resulta lento, en orden de impacto: desmarca *Modo razonamiento* en la pestaña
**Modelo** (el *thinking* de Qwen3 gasta cientos de tokens antes de responder), baja
`LLM_NUM_CTX`, o pasa a `qwen3:4b`.

Alternativas ya soportadas — se cambian sin tocar código:

| Modelo | RAM aprox. | Cuándo |
|---|---|---|
| `qwen3:4b` | ~2,6 GB | Iteración rápida; el doble de velocidad, algo menos de precisión |
| `qwen3:14b` | ~9 GB | Casos difíciles; ~2× más lento |
| `granite3.3:8b` | ~4,9 GB | Alternativa con muy buen *tool calling* |
| `gpt-oss:20b` | ~14 GB | Razonamiento fuerte; entra en 32 GB pero va lento en CPU |

**Cómo cambiarlo:** en la pestaña **Modelo** del panel derecho, o con `LLM_MODEL` en
`backend/.env`. El desplegable de la barra superior lista lo que ya tengas instalado.

**Descargar un modelo nuevo sin salir de la UI:** pestaña **Modelo** → *Descargar un modelo*.
Escribe la etiqueta (`qwen3:4b`) o pulsa uno de los atajos de la lista de sugeridos, y verás la
barra de progreso en vivo. Si el modelo escrito no está instalado, aparece un aviso ahí mismo y
un botón *Descargar* en la barra superior que te lleva directo.

La descarga ocurre en el motor, no en el navegador: puedes cerrar la pestaña y sigue. Si la
cancelas y la relanzas, Ollama reaprovecha las capas ya bajadas. El botón solo aparece con
proveedores que exponen esa API (`pull_capable` en `/api/config`); con `openai_compat` tienes
que cargar el modelo con las herramientas de tu motor.

### El interruptor de razonamiento con modelos que no están en el catálogo

El catálogo estático solo conoce los modelos que venían de fábrica, así que para uno
descargado después se le pregunta al motor: Ollama publica las capacidades reales de cada
modelo en `/api/show` (`thinking`, `tools`, `vision`...) y el backend las expone en
`GET /api/llm/features?provider=&model=`. Con eso la pestaña **Modelo** decide si el
interruptor aplica, lo etiqueta («razonamiento conmutable», «sin razonamiento», «según el
motor») y avisa si el modelo no admite herramientas —en cuyo caso no podrá llamar al MCP.

Además, a Ollama se le manda `think` explícitamente **también cuando vale `false`**.
Omitirlo no desactiva el razonamiento: Qwen3 y compañía razonan por defecto, así que sin ese
`false` explícito el interruptor solo funcionaba en un sentido. Si el modelo rechaza el
parámetro, el backend lo recuerda y reintenta sin él.

Comprobado con los modelos instalados en este equipo:

| Modelo | Capacidades según el motor | Interruptor |
|---|---|---|
| `qwen3:4b`, `qwen3:8b`, `qwen3:14b` | `completion, tools, thinking` | activo en ambos sentidos |
| `qwen3:30b-a3b-instruct-2507` | `completion, tools, thinking` | activo (la variante *instruct* no razona en la práctica) |
| `llama3.1:8b` | `completion, tools` | deshabilitado, `think` no se envía |
| `mistral-small3.2:24b` | `completion, vision, tools` | deshabilitado, `think` no se envía |

Con `openai_compat` (llama.cpp, vLLM, LM Studio) el interruptor viaja como
`chat_template_kwargs.enable_thinking`, que es la variable que usan las plantillas de los
modelos híbridos; si el motor la rechaza, se retira y se sigue sin controlarlo.

### Modelos en la nube con tu clave de API

Además del modelo local, la pestaña **Modelo** permite usar proveedores de nube
introduciendo la clave desde la interfaz. No hace falta reiniciar el backend ni tocar el `.env`.

| Proveedor | Modelos incluidos en el selector |
|---|---|
| **Anthropic** | Claude Opus 5 · Sonnet 5 · Opus 4.8 · Sonnet 4.6 · Haiku 4.5 · Fable 5.1 |
| **OpenAI** | GPT-5 · GPT-5 mini · GPT-5 nano · GPT-4.1 · GPT-4.1 mini · o4-mini |
| **Google** | Familias Gemini **pro**, **flash** y **flash-lite**, incluidos los `-preview` |
| **Cloudflare Workers AI** | GLM-4.7 Flash · gpt-oss 120B / 20B · Qwen3 30B · Mistral Small 3.1 · Llama 4 Scout · Llama 3.3 70B (solo modelos con *function calling*) |
| **NVIDIA (build.nvidia.com)** | Nemotron 3 Super · GLM-5.3 / 5.3 Flash · DeepSeek V4 Flash · gpt-oss 20B · Mistral Nemotron |
| **AWS (Amazon Bedrock)** | Claude Sonnet 4.6 · Claude Haiku 4.5 · Amazon Nova 2 Lite · gpt-oss 120B · DeepSeek V3.2 · Qwen3 Coder Next |

Flujo: elige el proveedor → pega la clave → **Validar**. Si la clave es buena, el desplegable
de modelos se rellena con **los modelos reales de tu cuenta** (consultados a la API), no solo
con la lista del catálogo. El campo de identificador exacto acepta cualquier modelo aunque no
esté en la lista, para probar uno recién salido.

El selector avisa de las particularidades de cada modelo: los que **ignoran `temperature`**
(Claude 4.7+, GPT-5, o-series) desactivan el deslizador, y los que **razonan siempre**
(Claude Fable) desactivan la casilla de razonamiento.

#### Dónde vive la clave

Dos opciones, y el backend acepta las dos:

1. **En la UI.** Se guarda en el `localStorage` del navegador para no repegarla cada vez.
   Es cómodo, pero no es un almacén de secretos: cualquier script que corra en ese origen
   puede leerla.
2. **En el backend.** Deja el campo vacío y define `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
   `GOOGLE_API_KEY`, `CLOUDFLARE_API_TOKEN`, `NVIDIA_API_KEY` o `AWS_BEARER_TOKEN_BEDROCK` en `backend/.env`. El backend la usa cuando la UI
   no manda ninguna.

En ambos casos la clave se usa solo para la petición en curso: **no se persiste en SQLite, no
se registra en MLflow y no aparece en los logs**. La caché de clientes se indexa por un hash
de la clave, no por la clave.

#### Cloudflare Workers AI (free tier)

Workers AI regala **10.000 neuronas al día** (se reinicia a las 00:00 UTC; al agotarlas las
peticiones fallan hasta el reinicio, salvo que tengas el plan de pago). Necesita dos datos:

1. **API token** con el permiso *Workers AI* (Read basta para listar modelos; para inferencia,
   la plantilla *Workers AI* del panel). Se crea en
   [dash.cloudflare.com/profile/api-tokens](https://dash.cloudflare.com/profile/api-tokens).
2. **Account ID** de la misma cuenta (panel de Cloudflare → *Workers AI*).

```ini
# backend/.env
CLOUDFLARE_API_TOKEN=...
CLOUDFLARE_ACCOUNT_ID=...
```

Los dos se pueden escribir también en la pestaña **Modelo** (el Account ID tiene su propio
campo, porque va dentro de la URL `.../accounts/<id>/ai/v1`). Detalles:

* Se usa el **endpoint compatible con OpenAI** de Workers AI, así que el agente, el simulador y
  el evaluador funcionan igual que con cualquier otro proveedor.
* **Validar** lista los modelos reales de la cuenta con la API propia de Workers AI
  (`/ai/models/search`, porque `/v1/models` no existe) y deja solo los de **generación de texto
  con function calling**: sin herramientas un modelo no sirve para probar un MCP.
* Los modelos de razonamiento (GLM-4.7 Flash, gpt-oss, Qwen3) reciben `reasoning_effort`; si
  alguno lo rechaza, se retira y se recuerda.
* **Qué modelo elegir para estirar el free tier:** el coste va por tokens, y los MCP con
  descripciones largas gastan mucho en entrada (con SIRTOD, 35k–130k tokens por caso de
  evaluación). `@cf/zai-org/glm-4.7-flash` es de los más baratos por token y tiene 131K de
  contexto; `gpt-oss-120b` es más capaz pero agota el cupo varias veces antes. Evita los de
  ventana corta (Llama 3.3 70B, 24K) con MCP de muchas herramientas.
* Al agotar el cupo, el error lo dice explícitamente (código `4006`) en lugar de un 429 mudo.

#### NVIDIA build.nvidia.com (endpoints gratuitos)

Los modelos alojados de [build.nvidia.com](https://build.nvidia.com) son gratuitos para
prototipar con el NVIDIA Developer Program. **No hay cupo de tokens, pero sí un límite de 40
peticiones por minuto** por cuenta (se puede solicitar subirlo a 200).

```ini
# backend/.env — clave en build.nvidia.com/settings/api-keys
NVIDIA_API_KEY=nvapi-...
```

* Endpoint compatible con OpenAI en `https://integrate.api.nvidia.com/v1`. **Validar** lista
  los modelos del catálogo quitando los que no conversan (embeddings, rerankers,
  guardarraíles, *parsers*, traducción, recompensa).
* **Razonamiento:** en estos modelos no hay `reasoning_effort`; lo lee la plantilla de chat de
  cada uno. Nemotron usa `enable_thinking` y Kimi, GLM o DeepSeek usan `thinking`, así que
  el interruptor manda ambas en `chat_template_kwargs`. Si el endpoint rechaza el parámetro,
  se retira y se recuerda. `gpt-oss` razona siempre y `mistral-nemotron` nunca.
* **Límite de 40 RPM y fallos intermitentes:** una evaluación encadena muchas peticiones
  (varias por turno del agente, más el simulador y el evaluador), y los endpoints gratuitos
  devuelven a veces `500 Internal server error` o `503 Service temporarily overloaded` a una
  petición que al reintentarla funciona. Ante `429`, `500`, `502`, `503` o `504` el backend
  espera lo que indique `Retry-After` (o 2 s, 4 s, 8 s… hasta 30 s) y reintenta hasta 5 veces
  antes de dar error.
* **Modelos comprobados con tool calling en el free tier** (2026-09-17): Nemotron 3 Super
  responde en ~1 s; GLM-5.3 en ~15 s; gpt-oss 20B en ~1 min; GLM-5.3 Flash y DeepSeek V4
  Flash tienen colas de varios minutos. `/v1/models` lista también modelos que una cuenta
  gratuita no puede usar (`404 Function not found for account`), como Kimi K2.6 o Nemotron
  Nano 3, por eso no están en el selector.

#### AWS (Amazon Bedrock)

Bedrock publica **también** un endpoint compatible con OpenAI, que habría encajado sin escribir
código, igual que Gemini. No sirve: su matriz de compatibilidad deja fuera a **Claude, Nova y
Llama**, justo las familias por las que se usa Bedrock. Así que aquí se habla su **API nativa
Converse**, la única con una forma única de *tool calling* para todos los modelos de chat del
servicio — y sin herramientas no se puede probar un MCP.

```ini
# backend/.env — clave en la consola de Bedrock -> API keys
AWS_BEARER_TOKEN_BEDROCK=...
AWS_REGION=us-east-1
```

* **Sin boto3 y sin firmar nada.** La clave de API de Bedrock viaja como
  `Authorization: Bearer`, así que basta `httpx`. No hace falta SigV4, ni access key, ni secret.
  Las claves son de dos tipos: las de corta duración (≤12 h, heredan los permisos de tu rol) y
  las de larga duración, que AWS marca como "solo para explorar".
* **La región va dentro del host** (`bedrock-runtime.<region>.amazonaws.com`), así que se pide
  en su propio campo de la pestaña **Modelo**, como el Account ID de Cloudflare. Tiene que ser
  la misma región en la que generaste la clave.
* **Perfiles de inferencia.** Muchos modelos no se invocan por su identificador pelado y exigen
  el del perfil, con prefijo de región (`us.anthropic.…`). Pero **no es universal**: varios
  modelos abiertos solo existen por su identificador base y solo en algunas regiones. Por eso
  *Validar* consulta las dos listas de la cuenta (`ListInferenceProfiles` y
  `ListFoundationModels`) y descarta los identificadores retirados.
* **Si *Validar* falla con 403**, puede ser la clave o pueden ser los permisos: la política que
  AWS engancha por defecto a las claves no siempre incluye `bedrock:ListInferenceProfiles` y
  `bedrock:ListFoundationModels`. En ese caso la inferencia funciona igual; escribe el
  identificador del modelo a mano.
* **Acceso a los modelos.** Está habilitado por defecto en las regiones comerciales y se
  suscribe solo en la primera invocación, pero los modelos de **Anthropic piden además rellenar
  una vez su formulario de acceso** en la consola. Si falta, sale un `AccessDeniedException`.
* **Razonamiento.** Converse no tiene un campo común: viaja en `additionalModelRequestFields` y
  el esquema lo valida contra cada modelo, así que una clave inventada es un 400 duro. Solo se
  manda donde AWS lo documenta — `thinking` en Claude, `reasoningConfig` en Nova — y a DeepSeek
  (que razona siempre), gpt-oss y Qwen no se les manda nada. Si aun así lo rechazan, el backend
  lo retira y lo recuerda, igual que con los motores locales.

#### Detalles de cada API que el código tiene en cuenta

* **Anthropic** usa el SDK oficial. `temperature` ya no existe en esa API (devuelve 400), el
  razonamiento va con `thinking: {type: "adaptive"}` y los `tool_result` de un turno tienen
  que ir agrupados en un único mensaje de usuario — repartirlos le enseña al modelo a dejar
  de pedir herramientas en paralelo. En Opus 5 y Fable se activa el *fallback* por rechazo en
  servidor, para que un rechazo del clasificador no rompa la prueba.
* Desmarcar *Modo razonamiento* con Claude **no lo desactiva**: baja el esfuerzo a `low`
  manteniendo el razonamiento adaptativo. Desactivarlo del todo hace que el modelo escriba a
  veces la llamada a la herramienta como texto en lugar de emitir un `tool_use`, y en un banco
  de pruebas de MCP eso es justo el fallo que no queremos.
* **OpenAI** exige `max_completion_tokens` y rechaza `temperature` en sus modelos de
  razonamiento. El código lo deduce del catálogo y, si la API se queja igualmente, reintenta
  una vez sin el parámetro y recuerda el ajuste.
* **Google** va por la capa compatible con OpenAI de Gemini
  (`generativelanguage.googleapis.com/v1beta/openai`), que sí acepta `max_tokens`. Para
  Vertex AI harían falta credenciales de GCP, que es otro mecanismo.

#### Gemini 3: firmas de razonamiento (`thought_signature`)

Desde Gemini 3, cada llamada a herramienta vuelve **firmada**: la respuesta trae
`tool_calls[].extra_content.google.thought_signature`, una instantánea cifrada del
razonamiento del modelo. Si el turno siguiente no la devuelve, la API corta con un 400:

```
Function call is missing a thought_signature in functionCall parts.
Additional data, function call `default_api:<herramienta>`, position 2.
```

El backend la guarda en `ToolCall.extra` y la reenvía tal cual. Detalles que importan:

* **Cada llamada lleva la suya.** Con herramientas en paralelo hay que devolver las N,
  no solo la primera.
* **Sobrevive al historial.** Viaja dentro de `tool_calls` en SQLite, así que un segundo
  turno del mismo hilo —que reconstruye el historial desde la base de datos— también la
  devuelve. Si solo viviera en memoria, fallaría en el turno 2.
* **Solo la recibe Gemini.** El reenvío está atado a `extra_content_key` en el proveedor;
  para OpenAI o un motor local `extra_content` sería un campo desconocido y devolverían 400.
  Esto importa al cambiar de modelo a mitad de conversación.

Es el mismo problema que los bloques de razonamiento firmados de Claude, que se resuelven
con `raw_blocks` / `_provider_blocks`.

#### Las manías de Gemini

Es el proveedor que más se aparta del dialecto de OpenAI, y cada desviación se manifiesta
como un `400 Bad Request` sin explicación. Todas están contempladas:

| Síntoma | Causa |
|---|---|
| Falla en el **primer mensaje** | Herramienta MCP **sin argumentos**: Gemini rechaza `properties: {}`, hay que omitir `parameters` entero |
| Falla en el **primer mensaje** | El `inputSchema` del MCP trae `$schema`, `$ref`, `additionalProperties`, `const`, `exclusiveMinimum` o `format: uuid` — Gemini solo admite un subconjunto de OpenAPI 3.0 |
| Falla en el **primer mensaje** | `reasoning_effort: "minimal"`, que solo existe en OpenAI (aquí se usa `low`) |
| Falla **tras la primera herramienta** | `"content": null` en el turno del asistente con `tool_calls`, y el campo `name` en los mensajes de rol `tool` |
| `404` | El identificador de modelo no existe en esa versión de la API |

Los esquemas de las herramientas se traducen en [schema_utils.py](backend/app/llm/schema_utils.py):
resuelve `$ref`, fusiona `allOf`, pasa `oneOf` a `anyOf`, convierte `const` en `enum`, mapea
`anyOf: [X, null]` a `nullable` y descarta lo demás con una **lista de permitidos** — ante un
esquema raro es preferible perder una restricción que tumbar la petición entera.

**El selector de modelos se rellena desde tu cuenta.** El catálogo estático solo tiene tres
entradas de partida (`gemini-2.5-pro`, `-flash`, `-flash-lite`) porque los identificadores con
versión caducan y devuelven 404. Al pulsar *Validar*, el desplegable pasa a mostrar los modelos
reales que tu clave puede usar, filtrados a las familias pro / flash / flash-lite e incluyendo
los `-preview`; se descartan embeddings, imagen, veo, tts y demás, que no sirven para un agente.

#### Cuando algo falle

Los rechazos del proveedor ya no se pierden: el mensaje que llega a la UI incluye **el motivo
real que devuelve la API** (no solo el código HTTP), y el backend registra además el modelo,
los roles del historial, las herramientas ofrecidas y los parámetros enviados:

```
WARNING | google rechazo la peticion (400) | modelo=gemini-2.5-flash
        | roles=['system','user','assistant','tool'] herramientas=['consultar_inventario']
        | params=['max_tokens','model','reasoning_effort','stream','temperature','tool_choice','top_p']
```

### Cambiar de motor de inferencia

El backend no habla con Ollama, habla con una interfaz (`LLMProvider`). Vienen dos
implementaciones:

* `ollama` — Ollama (por defecto).
* `anthropic` — Claude, sobre el SDK oficial.
* `openai` / `google` / `cloudflare` / `nvidia` — OpenAI, Gemini, Workers AI y build.nvidia.com.
* `aws` — Amazon Bedrock, por su API nativa Converse.
* `openai_compat` — cualquier servidor con API compatible OpenAI: `llama-server` de
  llama.cpp, LM Studio, vLLM, TGI, LocalAI…

```env
LLM_PROVIDER=openai_compat
LLM_BASE_URL=http://127.0.0.1:8080
LLM_MODEL=qwen3-8b
```

Para añadir un motor nuevo: implementa `LLMProvider` en `backend/app/llm/` y añádelo al
diccionario `PROVIDERS` de [registry.py](backend/app/llm/registry.py). Nada más.

---

## 3. Memoria de la conversación

Cada conversación es un hilo con memoria: el backend reconstruye el historial completo
(mensajes de usuario, respuestas, llamadas a herramientas y sus resultados) desde SQLite
en cada turno.

El botón **Reiniciar conversación** abre un hilo nuevo y limpio dentro de la misma sesión;
el hilo anterior y sus trazas se conservan para poder compararlos. Si prefieres borrar la
memoria manteniendo el mismo identificador:

```
POST /api/conversations/{id}/reset?new_thread=false
```

La ventana de historial se recorta con `AGENT_HISTORY_MAX_MESSAGES` (60 por defecto),
conservando siempre el *system prompt*.

### Sesiones

Una sesión agrupa N conversaciones. **No se crean solas**: nacen al pulsar *+ Sesión* o al
enviar el primer mensaje sin sesión abierta. Al arrancar, la UI abre la sesión más reciente
si la hay, y si no hay ninguna espera a que escribas.

**Nombre.** El automático es corto y ordenable (`10/09 10:49`). Para cambiarlo, doble clic
sobre el nombre en la barra lateral o el botón `✎`; *Enter* guarda, *Escape* cancela. El
nombre viaja al run de MLflow (`session::<nombre>`): si la sesión ya tenía run, se repinta;
si aún no lo tenía, nace con el nombre nuevo.

No hace falta que los nombres sean únicos. MLflow no lo exige —los runs se identifican por
`run_id`— y aquí siempre se buscan por `tags.session_id`, que sí lo es. Dos sesiones que se
llamen igual conviven sin conflicto, y borrar una no toca la otra.

Cada sesión se borra desde la papelera (`x`) de la barra lateral. El borrado arrastra sus
conversaciones, mensajes, interacciones y eventos de herramienta en SQLite **y** sus runs y
trazas en MLflow. El botón *Limpiar vacías* borra de golpe todas las que no llegaron a
registrar ninguna interacción, conservando la abierta.

```
DELETE /api/sessions/{id}                  # borra tambien en MLflow
DELETE /api/sessions/{id}?purge_mlflow=false   # conserva el registro
POST   /api/sessions/prune?keep={id}       # borra las sesiones vacias
PATCH  /api/sessions/{id}                  # {"title": "..."} renombra (tambien en MLflow)
DELETE /api/conversations/{id}             # mismo criterio, por conversacion
```

Matiz de MLflow: las trazas se eliminan de verdad; los runs se marcan como eliminados
(`delete_run` es un borrado lógico y desaparecen de la vista normal). MLflow no ofrece
borrado físico por API — para liberar el disco, `mlflow gc` en el servidor de tracking.

---

## 4. Frontend y backend en máquinas distintas

Todo está preparado para ello:

* El backend escucha en `0.0.0.0` y trae CORS abierto (`CORS_ORIGINS=*`; puedes limitarlo
  a orígenes concretos separados por comas).
* El dev server de Vite arranca con `--host 0.0.0.0`.
* La UI tiene un campo **Backend** en la barra superior. Escribe ahí
  `http://<ip-del-backend>:8090`, pulsa *Aplicar* y queda guardado en el navegador.

En la máquina del backend hay que abrir los puertos del firewall (una vez, como admin):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\open-firewall.ps1
```

Solo abre los perfiles *Private* y *Domain*, nunca redes públicas.

---

## 5. Captura de la interacción con el MCP

Se registra en tres niveles, del más fino al más grueso:

1. **Tramas JSON-RPC.** El transporte del SDK de MCP se envuelve con un *tee*
   ([tee.py](backend/app/mcpclient/tee.py)) que copia cada mensaje en ambos sentidos sin
   tocar la librería ni el servidor remoto. Ves el `tools/call` exacto que sale y la
   respuesta exacta que entra.
2. **Llamadas a herramienta.** Nombre, argumentos, resultado, `structuredContent`, error,
   latencia, servidor de origen y las tramas asociadas.
3. **Interacción.** El turno completo: mensaje, razonamiento, N llamadas al LLM, M llamadas
   al MCP y la respuesta final, con sus métricas agregadas.

Dónde verlo:

* **En la UI**, en línea con la conversación (cada herramienta es un bloque desplegable con
  argumentos, resultado y tramas) y en la pestaña **Trazas** para el histórico persistido.
* **En SQLite**, en `backend/data/agente.sqlite3` (tablas `interactions`, `tool_events`,
  `messages`).
* **Exportado** a un JSON completo de la sesión: `GET /api/traces/export?session_id=...`
  (botón *Exportar* de la pestaña Trazas).

La pestaña *Servidor MCP* incluye además un **ejecutor manual de herramientas**: llamas a
una herramienta con los argumentos que quieras sin pasar por el LLM, útil para aislar si un
fallo es del MCP o del modelo.

---

## 6. MLflow

### Elegir el experimento desde la UI

Arriba de la barra lateral hay un campo **Experimento de MLflow**: se elige uno de los que
ya existen (desplegable) o se escribe un nombre nuevo, que se **crea al aplicarlo**. Lo
elegido se guarda en la sesión abierta (`sessions.mlflow_experiment`), así que cada sesión
puede registrar en un experimento distinto y al reabrirla vuelve al suyo.
`MLFLOW_EXPERIMENT` del `.env` pasa a ser solo el valor por defecto.

Cambiar de experimento a media sesión no mueve lo ya registrado: a partir de la siguiente
interacción se abren runs nuevos en el experimento de destino.

```
GET  /api/mlflow/experiments        # los que existen
POST /api/mlflow/experiments        # {"name": "..."} -> lo crea si no existe
PATCH /api/sessions/{id}            # {"mlflow_experiment": "..."}
```

### Jerarquía

Los tres niveles que pediste se mapean así:

```
Experimento  <el elegido en la UI>
└── Run padre        SESIÓN          tag: session_id
    └── Run hijo     CONVERSACIÓN    tags: session_id, conversation_id
        └── Trace    INTERACCIÓN     tags: session_id, conversation_id, interaction_id
            ├── span agent_interaction   (AGENT)  el turno completo
            ├── span llm_call_N          (LLM)    tokens, latencia, modelo
            └── span mcp_tool::<nombre>  (TOOL)   args, resultado, servidor, JSON-RPC
```

### Qué se registra

**Métricas** (por interacción, con `step` = índice dentro de la conversación, para ver la
evolución del hilo):

`latency_ms`, `llm_latency_ms`, `mcp_latency_ms`, `overhead_ms`, `prompt_tokens`,
`completion_tokens`, `total_tokens`, `llm_calls`, `tool_calls`, `tool_errors`,
`tool_success_rate`, `iterations`, `output_tokens_per_second`, `total_tokens_per_second`.

**Parámetros** (por conversación): proveedor, modelo, `base_url`, temperatura, `num_ctx`,
`max_tokens`, modo razonamiento, `max_iterations` y las URLs de los MCP conectados.

**Artefactos** (por conversación):
* `interactions/<interaction_id>.json` — transcripción completa, incluidas las tramas JSON-RPC.
* `mcp_tool_calls/<interaction_id>/<seq>_<herramienta>.json` — una por llamada al MCP, con
  argumentos, resultado **íntegro**, `structuredContent` y las tramas JSON-RPC en crudo.
* `mcp_tool_calls.json` — tabla índice de todas las llamadas, con `result_chars` y
  `result_artifact` apuntando al fichero anterior.

### Nada de lo que se registra está truncado

Hay recorte en dos sitios, y por motivos distintos:

| Dónde | Límite | Por qué |
| --- | --- | --- |
| Observación que se le devuelve al modelo | `MAX_TOOL_RESULT_CHARS` (12 000) | presupuesto de contexto |
| Texto que viaja por SSE a la UI | `MAX_UI_RESULT_CHARS` (4 000) | la UI no necesita 200 KB de JSON |

El registro no se recorta en ningún caso. El resultado completo de cada llamada al MCP está
en tres sitios: el span `mcp_tool::<nombre>` de la traza, el artifact por llamada, y la
columna `result_text` de `tool_events` en SQLite (que es lo que muestra la pestaña
*Trazas*). Cuando la UI recorta, lo dice e indica la ruta del artifact.

El consumo de tokens se publica con la clave estándar `mlflow.chat.tokenUsage` en los spans
LLM, así que MLflow lo **agrega solo** a nivel de traza y lo muestra en su UI sin
configuración extra.

### Consultas útiles

En la UI de MLflow, en la pestaña *Traces*:

```
tags.session_id = 'ses_xxxxxxxx'                     # todo el trabajo de una sesión
tags.conversation_id = 'conv_xxxxxxxx'               # un hilo concreto
tags.interaction_id = 'int_xxxxxxxx'                 # un turno concreto
```

Desde Python:

```python
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
exp = MlflowClient().get_experiment_by_name("agente-pruebas-mcp")

df = mlflow.search_traces(
    locations=[exp.experiment_id],
    filter_string="tags.session_id = 'ses_xxxxxxxx'",
)
```

### Si MLflow no está levantado

El backend sigue funcionando: lo indica en `/api/health` y en la UI, y todo se guarda igual
en SQLite. Es deliberado — la observabilidad no debe tumbar el banco de pruebas. Para
desactivarla del todo: `MLFLOW_ENABLED=false`.

---

## 7. Evaluación con usuarios simulados

El modo **Evaluación** (selector *Chat / Evaluación* de la barra superior) ejecuta baterías de
casos sin intervención manual: un agente **simulador** interpreta a una persona y conversa
con tu agente conectado al MCP, y al terminar un agente **evaluador** juzga la conversación
con la rúbrica del caso.

```
personas.json ─┐                          ┌─► comprobación determinista (las cifras esperadas)
               ├─► consulta × repetición ─┤
consultas.json ┘   simulador ⇄ agente+MCP └─► evaluador LLM (un juicio por criterio)
                   (N turnos, hasta <<FIN>>)          │
                                                      ▼
                                     nota = Σ pesos cumplidos / Σ pesos  (la calcula el código)
```

### Los dos ficheros

Son independientes para poder combinar el mismo juego de personas con distintos bancos de
consultas. Las plantillas están en
[backend/app/evals/templates/](backend/app/evals/templates/) y se cargan desde la UI con el
botón **Plantillas**. Las claves desconocidas son un error (no un aviso): en un fichero escrito
a mano, `valor-esperado` en lugar de `valor_esperado` no debe dejar la consulta sin dato que
comprobar en silencio.

**`personas.json`** — los perfiles que se simulan, indexados por su identificador:

```json
{
  "estudiante": {
    "nombre": "Estudiante universitario",
    "descripcion": "Eres un estudiante de ciencias sociales que necesita datos para un trabajo academico..."
  }
}
```

| Campo | Para qué |
|---|---|
| clave | Identificador (letras, números, `_ . -`). Lo referencian las consultas y se usa en MLflow. |
| `nombre` | Cómo se presenta la persona. |
| `descripcion` | Quién es, qué sabe y cómo conversa. Va **tal cual** al prompt del simulador. |

**`consultas.json`** — el banco de pruebas, una lista donde cada consulta trae su persona:

```json
[
  {
    "id": "PG-01",
    "persona": "estudiante",
    "goal": "Obtener la tasa de pobreza de Ayacucho en el anio mas reciente disponible.",
    "consulta_inicial": "Hola, necesito saber cuanta pobreza hay en Ayacucho para mi tesis.",
    "ambiguedad": "alta",
    "resultado_esperado": "Existe un valor y se devuelve correctamente a pesar de la ambiguedad",
    "valor_esperado": "30.0% - 33.8% (intervalo de confianza)"
  }
]
```

| Campo | Para qué |
|---|---|
| `id` | Identificador de la consulta. Es la etiqueta `eval_case_id` en MLflow. |
| `persona` | Quién la formula. Tiene que existir en `personas.json`. |
| `goal` | Lo que la persona quiere conseguir. Lo recibe el simulador, **no** el agente. |
| `consulta_inicial` | Primer mensaje literal. Si falta, lo redacta el simulador. |
| `ambiguedad` | `alta`, `media` o `baja`: cuánto concreta la persona al preguntar. Modula el prompt del simulador. |
| `resultado_esperado` | Qué debería pasar, en texto libre (*"Existe un valor y se devuelve correctamente"*, *"No se encuentra disponible el valor o no existe"*). |
| `valor_esperado` | El dato esperado, en texto libre: una cifra (`330481.79`), una serie (`863, 933, 4907`), un rango (`30.0% - 33.8%`), un ámbito temático (`empleo o mercado laboral`) o `No aplica`. |

**No hay producto cartesiano**: cada consulta se ejecuta con su persona y nada más. Descartar
una persona en la UI descarta sus consultas.

### La rúbrica no está en los ficheros

El banco de pruebas dice qué se pide y qué debería salir; cómo se puntúa lo fija el código
(`CRITERIOS` / `criterios_de` en [spec.py](backend/app/evals/spec.py)), igual para todas las
consultas:

| Elemento | Peso | Obligatorio | Quién lo decide |
|---|---|---|---|
| `resultado_esperado` | 3 | sí | evaluador LLM |
| `valor_esperado` (si lo hay) | 3 | no | evaluador LLM |
| `sin_invenciones` — toda cifra que afirma viene de una herramienta | 2 | sí | evaluador LLM |
| `respuesta_util` — contesta a lo que se pide y al nivel de la persona | 1 | no | evaluador LLM |
| `valor_mencionado` (si `valor_esperado` trae cifras) | 1 | no | código |

`max_turnos` y el `umbral` de aprobación tampoco están en los ficheros: son de la ejecución y
se ajustan en la UI (6 y 0,7 por defecto).

### Cómo se puntúa

* El evaluador LLM solo decide, criterio a criterio, si se cumple y por qué. Un criterio sobre
  el que no se pronuncia cuenta como no cumplido (y se marca).
* La **comprobación determinista** no pasa por ningún LLM: si `valor_esperado` trae cifras,
  comprueba que todas aparezcan en las respuestas del agente (acepta `4.907` y `4907`, `24,9` y
  `24.9`). Un `valor_esperado` sin cifras (`empleo o mercado laboral`, `No aplica`) no genera
  comprobación: lo juzga el evaluador leyendo el texto.
* La **nota** la calcula el código con los pesos de la rúbrica. La consulta se **aprueba** si la
  nota llega al umbral *y* se cumplen todos los elementos obligatorios. Cambiar de evaluador
  cambia los juicios, nunca la aritmética.
* Si el evaluador no devuelve un JSON válido tras un reintento, la consulta queda en `error`
  (no suspende): es un fallo del instrumento, no del agente.

### En la UI

1. **Ficheros**: carga o edita los dos JSON; se validan al vuelo con errores legibles
   (`consultas.json · [0] (PG-01).valor-esperado: clave no reconocida`). El nombre del fichero
   de consultas se usa como nombre del banco (`eval_suite` en MLflow).
2. **Qué ejecutar**: marca consultas y personas, repeticiones (para medir la variabilidad del
   agente), el límite de turnos y el umbral de aprobación.
3. **Modelos**: el agente bajo prueba usa lo configurado en *Modelo* y *Servidor MCP*, igual
   que el chat. El simulador y el evaluador pueden usar el mismo modelo o cualquier otro del
   catálogo (también de nube, con la clave guardada en *Modelo*). Por defecto el simulador va
   sin razonamiento (más rápido y natural) y el evaluador con temperatura 0.
4. **MLflow**: experimento (se crea si no existe) y nombre de la ejecución.

Mientras corre se ve la conversación en vivo (persona, agente y cada llamada al MCP); al cerrar
cada caso aparece la rúbrica evaluada con sus justificaciones. Las pestañas *Trazas* y
*Métricas* del inspector siguen al caso seleccionado, y **Ver en el chat** abre la conversación
con el razonamiento y las tramas JSON-RPC.

La ejecución corre **en el backend**, no en el navegador: recargar la página no la detiene y la
UI se reengancha sola. Se puede cancelar (lo ya evaluado se conserva). Si se reinicia el
backend, las que estaban en marcha quedan como `interrumpida`. Si el **servidor MCP se cae** a
media batería, la ejecución se detiene con estado `error` y el motivo a la vista, en vez de
seguir con los casos restantes sin herramientas: reconecta el MCP y relanza lo que falte
marcando solo esas consultas. Orientativo en CPU: cada turno cuesta una o dos llamadas al
modelo del agente más una del simulador, y cada caso una del evaluador, así que un caso de 3
turnos con modelos de 4–8 B tarda varios minutos.

### Qué queda en MLflow

Se reutiliza la jerarquía del chat, con etiquetas extra:

```
Experimento
  Run padre   -> la ejecución          (tags: kind=evaluation, eval_run_id, eval_suite)
     params:     modelos de agente/simulador/evaluador, MCP, hashes de los dos ficheros
     metrics:    eval.pass_rate, eval.avg_score, eval.passed/failed/errors, tokens.agent/simulator/judge
     artifacts:  evaluation/personas.json, consultas.json, summary.json, results.json (tabla)
    Run hijo  -> consulta × repetición  (tags: eval_case_id, eval_persona_id, eval.status)
     metrics:    eval.score, eval.passed, eval.turns, sim.*, judge.*, agent.*
     artifacts:  evaluation/result.json (transcripción + rúbrica), judge.json (prompt y respuesta),
                 criteria.json (tabla)
      Traza por turno del agente        (tags: eval_role=agent, eval_turn)
        └ assessments en la última:      un feedback por criterio (LLM_JUDGE o CODE),
                                         eval_score, eval_passed y la expectation resultado_esperado
      Traza del evaluador               (tags: eval_role=judge)
```

Los dos ficheros se guardan tal cual en el run padre, así que cualquier ejecución es
reproducible desde MLflow. Consultas útiles:

```python
# Comparar ejecuciones de una misma batería
mlflow.search_runs(experiment_names=["agente-pruebas-mcp"],
                   filter_string="tags.kind = 'evaluation' and tags.level = 'session'")

# Todas las trazas de una consulta concreta, en todas las ejecuciones
mlflow.search_traces(locations=[exp_id], filter_string="tags.eval_case_id = 'PG-01'")
```

Borrar la ejecución desde la UI borra su sesión, sus conversaciones y todo su rastro en MLflow
(runs, trazas del agente y del evaluador).

---

## 8. Estructura

```
backend/app/
  main.py                  arranque de FastAPI, CORS, ciclo de vida
  config.py                configuración por entorno
  llm/
    base.py                contrato LLMProvider + normalización de mensajes
    catalog.py             proveedores y modelos que ofrece la UI
    ollama_provider.py     Ollama
    anthropic_provider.py  Claude (SDK oficial)
    cloud_openai_providers.py  OpenAI, Gemini, Cloudflare Workers AI y NVIDIA
    bedrock_provider.py    Amazon Bedrock (API Converse)
    openai_compat_provider.py  llama.cpp / LM Studio / vLLM / ...
    registry.py            fábrica y caché de proveedores (clave por hash)
  mcpclient/
    connection.py          conexión persistente (task supervisora + cola de comandos)
    tee.py                 intercepción de las tramas JSON-RPC
    manager.py             N servidores a la vez + enrutado de herramientas
  agent/
    loop.py                bucle LLM ↔ herramientas, emisión de eventos, métricas
    prompts.py             system prompt del agente de pruebas
  evals/
    spec.py                estructura y validación de personas.json / consultas.json, y la rúbrica
    simulator.py           agente que interpreta a la persona (cierra con <<FIN>>)
    judge.py               agente evaluador (JSON con un juicio por criterio)
    checks.py              comprobaciones deterministas, nota y parseo del JSON
    runner.py              ejecución en segundo plano, eventos SSE, SQLite y MLflow
    templates/             plantillas de los dos ficheros
  store/                   esquema SQLite y repositorio
  observability/
    mlflow_tracker.py      runs, traces, spans, métricas, artefactos y assessments
  api/                     rutas HTTP (sistema, MCP, chat SSE, datos/trazas, evaluación)

frontend/src/
  App.tsx                  estado global y orquestación
  lib/api.ts               cliente HTTP + parser SSE
  lib/evalState.ts         estado de una evaluación (eventos en vivo + SQLite)
  components/              TopBar, Sidebar, Chat, McpPanel, TracesPanel, MetricsPanel,
                           EvalWorkbench (+ eval/EvalSetup, eval/EvalRunView)

scripts/
  setup.ps1                instalación
  start-all.ps1            arranca MLflow + backend + frontend
  open-firewall.ps1        acceso desde otra máquina
  demo_mcp_server.py       servidor MCP de ejemplo
```

---

## 9. Detalles de implementación que conviene conocer

* **Una task por conexión MCP.** El SDK de MCP se apoya en *task groups* de anyio, que no
  toleran que una sesión cruce fronteras de tarea. Cada conexión vive dentro de una task
  supervisora y el resto del backend le habla por una cola de comandos. Efecto secundario:
  las llamadas a un mismo servidor MCP se serializan.
* **SDK de MCP 2.x.** Se usa el cliente de alto nivel `mcp.Client` con campos en
  `snake_case` (`input_schema`, `is_error`, `structured_content`). El método `ping`
  desapareció del protocolo `2026-07-28`; la prueba de vida es un `tools/list`.
* **Transporte automático.** Se intenta *streamable HTTP* y se cae a *SSE*; si la URL
  termina en `/sse` se prueba en orden inverso. También se puede forzar desde la UI.
* **Spans sin contexto implícito.** MLflow propaga la traza activa por `contextvars`; con
  peticiones concurrentes eso es frágil, así que se usa `start_span_no_context` con el
  span padre pasado explícitamente y las llamadas bloqueantes van a un hilo aparte.
* **Errores de herramienta como observación.** Si un MCP falla, el error no rompe el turno:
  se le devuelve al modelo como resultado de la herramienta para que reaccione, y queda
  marcado en rojo en la UI y contabilizado en `tool_errors`.
* **Una conexión MCP que se cae no es una cancelación.** Cuando el servidor se cierra con una
  llamada en vuelo, la tarea supervisora de la conexión **falla** el *future* de quien espera
  en vez de cancelarlo. La diferencia importa: `CancelledError` es `BaseException`, así que se
  cuela por los `except Exception` del bucle del agente —dejando la traza a medias y las
  métricas sin escribir— y más arriba una batería de evaluación lo confunde con un *Detener*
  del usuario. Lo que quedara encolado también falla en el acto, en vez de esperar el timeout
  completo a una conexión que ya no atiende a nadie.
* **La evaluación distingue quién canceló.** `Task.cancelling()` separa una cancelación real de
  la batería de un `CancelledError` que sube desde más abajo; lo segundo se registra como
  **error**, con su motivo, en lugar de hacerlo pasar por una parada voluntaria. Y si se
  declararon servidores MCP pero ya no queda ninguno vivo, la batería se detiene ahí: los casos
  restantes se ejecutarían sin herramientas y "aprobarían" sin haber probado nada.
* **El run de MLflow se crea con la primera interacción**, no al crear la sesión. Una sesión
  que se abre y no se usa no deja rastro en el experimento.
* **Vaciado antes de purgar.** MLflow exporta las trazas en segundo plano; al borrar una
  sesión se fuerza `flush_trace_async_logging()` antes de buscarlas, o la última traza
  aterrizaría después de la purga y quedaría huérfana.
* **MLflow falla rápido.** Su cliente reintenta 7 veces con *backoff* por defecto, así que
  una caída del servidor a media sesión dejaba colgada cualquier petición un par de minutos.
  `MLFLOW_HTTP_TIMEOUT` y `MLFLOW_HTTP_RETRIES` lo acotan (10 s / 2 intentos).

---

## 10. Prueba de humo

Con el MCP de ejemplo levantado, valida el circuito entero (agente, memoria, captura
JSON-RPC, SQLite y MLflow) sin necesidad de tener un modelo descargado:

```powershell
cd backend
.venv\Scripts\python.exe tests\test_agent_loop.py       # agente + MCP + MLflow (LLM simulado)
.venv\Scripts\python.exe tests\test_cloud_providers.py  # traducción al dialecto de cada nube
.venv\Scripts\python.exe tests\test_http_smoke.py       # circuito completo con el modelo real
.venv\Scripts\python.exe -m pytest tests\test_evals.py  # evaluaciones: ficheros, nota y batería completa
.venv\Scripts\python.exe -m pytest tests\test_mcp_connection.py  # qué pasa si el MCP se cae a media prueba
```

`test_evals.py` tampoco necesita Ollama, MCP ni MLflow: ejecuta una batería entera con tres
modelos guionizados (agente, simulador y evaluador) sobre una SQLite temporal.

`test_cloud_providers.py` no hace red ni necesita claves: comprueba cómo se construye el
payload de cada proveedor, que es donde están las diferencias que rompen.
