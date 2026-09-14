"""Integracion con MLflow.

Modelo de datos que se replica en MLflow:

  Experimento           -> el banco de pruebas (se elige desde la UI)
    Run (padre)         -> una SESION       (tag: session_id)
      Run (hijo)        -> una CONVERSACION (tags: session_id, conversation_id)
        Trace           -> una INTERACCION  (tag: interaction_id)
          span AGENT    -> el bucle completo
          span LLM      -> cada llamada al modelo (tokens, latencia)
          span TOOL     -> cada llamada al servidor MCP (args, resultado, JSON-RPC)

Las evaluaciones reutilizan la misma jerarquia con etiquetas extra
(`kind=evaluation`, `eval_run_id`, `eval_case_id`, `eval_persona_id`): la
ejecucion de la bateria es el run padre, cada caso x persona un run hijo, y
los juicios del evaluador se cuelgan como *assessments* (feedback) de la
ultima traza de la conversacion, que es como MLflow los muestra de forma
nativa. El evaluador deja ademas su propia traza (`eval_role=judge`).

Decisiones importantes:

* Se usa `MlflowClient` con `run_id` explicito en lugar de la API fluida, para
  no depender del run activo global (que no es seguro con peticiones async
  concurrentes).
* Para los spans se usa `start_span_no_context`, que no depende del contexto
  implicito y por tanto funciona bien con asyncio.
* El experimento es *runtime*: cada sesion puede registrar en uno distinto y
  se crea si el nombre no existe todavia. Las caches de runs se indexan por
  (experimento, id) para que cambiar de experimento no reutilice runs viejos.
* Nada de lo que se registra en MLflow se trunca: los resultados de las
  herramientas MCP van completos al span y ademas como artifact, para poder
  auditarlos enteros aunque la UI muestre solo un extracto.
* Todo es *fail soft*: si MLflow no esta disponible el agente sigue
  funcionando y la UI lo indica.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..config import settings

logger = logging.getLogger(__name__)

MLFLOW_PARENT_RUN_ID = "mlflow.parentRunId"
MLFLOW_RUN_NAME = "mlflow.runName"

# Claves estandar de MLflow. Usarlas (en vez de nombres propios) hace que la UI
# agregue tokens, modelo y sesion de forma nativa en la vista de traces.
META_SOURCE_RUN = "mlflow.sourceRun"
META_TRACE_SESSION = "mlflow.trace.session"
ATTR_CHAT_USAGE = "mlflow.chat.tokenUsage"
ATTR_CHAT_TOOLS = "mlflow.chat.tools"
ATTR_LLM_MODEL = "mlflow.llm.model"
ATTR_LLM_PROVIDER = "mlflow.llm.provider"

# Limite real de MLflow para el valor de un parametro de run.
MAX_PARAM_CHARS = 490
# `delete_traces` no acepta mas ids por llamada.
MAX_TRACE_IDS_PER_DELETE = 100


def _truncate(value: Any, limit: int = 4000) -> str:
    """Recorta un texto. Solo para *params* y *tags*, que MLflow limita.

    Los resultados de herramientas y las respuestas del modelo nunca pasan por
    aqui: van completos a spans y artifacts.
    """
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} chars mas]"


@dataclass
class SpanHandle:
    """Referencia a un span abierto (o a nada, si el tracing no esta activo)."""

    span: Any = None
    started: float = field(default_factory=time.perf_counter)

    @property
    def active(self) -> bool:
        return self.span is not None


@dataclass
class InteractionTrace:
    """Estado de la traza de una interaccion concreta."""

    session_id: str
    conversation_id: str
    interaction_id: str
    run_id: str = ""
    trace_id: str = ""
    experiment_id: str = ""
    experiment: str = ""
    root: SpanHandle = field(default_factory=SpanHandle)
    step: int = 0
    tool_rows: list[dict[str, Any]] = field(default_factory=list)


class MLflowTracker:
    """Fachada unica hacia MLflow. Nunca propaga excepciones al agente."""

    def __init__(self) -> None:
        self.enabled = settings.mlflow_enabled
        self.available = False
        self.status: str = "sin inicializar"
        self.default_experiment: str = settings.mlflow_experiment
        self.experiment_id: str | None = None
        self._client: Any = None
        self._mlflow: Any = None
        self._start_span: Callable[..., Any] | None = None
        self._span_type: Any = None
        # nombre -> experiment_id
        self._experiments: dict[str, str] = {}
        # (experiment_id, session_id) -> run_id; (experiment_id, conv_id) -> run_id
        self._session_runs: dict[tuple[str, str], str] = {}
        self._conversation_runs: dict[tuple[str, str], str] = {}
        self._steps: dict[str, int] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------- arranque -
    def _init_sync(self) -> None:
        import mlflow
        from mlflow.tracking import MlflowClient

        # El cliente de MLflow lee estos limites del entorno en cada peticion.
        # Sin ellos reintenta 7 veces con backoff exponencial, asi que si el
        # servidor se cae a media sesion cualquier llamada (incluida la que
        # hace la UI al listar experimentos) se queda colgada un par de
        # minutos en vez de fallar y seguir.
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", str(int(settings.mlflow_http_timeout)))
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", str(settings.mlflow_http_retries))

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

        experiment = client.get_experiment_by_name(settings.mlflow_experiment)
        if experiment is None:
            experiment_id = client.create_experiment(settings.mlflow_experiment)
        else:
            experiment_id = experiment.experiment_id
        mlflow.set_experiment(experiment_id=experiment_id)

        # `start_span_no_context` crea spans sin apoyarse en el contexto implicito:
        # es la unica variante segura bajo asyncio con interacciones concurrentes.
        start_span = getattr(mlflow, "start_span_no_context", None)
        if start_span is None:
            start_span = getattr(getattr(mlflow, "tracing", None), "start_span_no_context", None)

        span_type = None
        try:
            from mlflow.entities import SpanType

            span_type = SpanType
        except Exception:  # noqa: BLE001
            span_type = None

        self._mlflow = mlflow
        self._client = client
        self.experiment_id = experiment_id
        self._experiments[settings.mlflow_experiment] = experiment_id
        self._start_span = start_span
        self._span_type = span_type
        self.available = True
        self.status = (
            f"conectado a {settings.mlflow_tracking_uri} (experimento '{settings.mlflow_experiment}')"
            + ("" if start_span else " — sin soporte de traces, solo metricas")
        )

    async def init(self) -> None:
        if not self.enabled:
            self.status = "deshabilitado por configuracion (MLFLOW_ENABLED=false)"
            return
        try:
            await asyncio.to_thread(self._init_sync)
            logger.info("MLflow %s", self.status)
        except Exception as exc:  # noqa: BLE001
            self.available = False
            self.status = f"no disponible: {type(exc).__name__}: {exc}"
            if settings.mlflow_fail_soft:
                logger.warning("MLflow %s (se continua sin trazabilidad)", self.status)
            else:
                raise

    def info(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "status": self.status,
            "tracking_uri": settings.mlflow_tracking_uri,
            "experiment": self.default_experiment,
            "experiment_id": self.experiment_id,
            "tracing_supported": self._start_span is not None,
        }

    def experiment_url(self, experiment_id: str | None = None) -> str:
        uri = settings.mlflow_tracking_uri.rstrip("/")
        eid = experiment_id or self.experiment_id
        if uri.startswith("http") and eid:
            return f"{uri}/#/experiments/{eid}"
        return uri

    # --------------------------------------------------------- experimentos -
    def _ensure_experiment_sync(self, name: str) -> str:
        """Devuelve el id del experimento, creandolo si el nombre no existe."""
        experiment = self._client.get_experiment_by_name(name)
        if experiment is not None:
            # Un experimento borrado conserva el nombre reservado: hay que
            # restaurarlo antes de poder escribir runs en el.
            if getattr(experiment, "lifecycle_stage", "active") == "deleted":
                self._client.restore_experiment(experiment.experiment_id)
            return experiment.experiment_id
        return self._client.create_experiment(name)

    async def ensure_experiment(self, name: str | None) -> str:
        """Resuelve un nombre de experimento a su id, creandolo si hace falta.

        Un nombre vacio significa "el del backend". El resultado se cachea para
        no consultar el tracking server en cada interaccion.
        """
        if not self.available:
            return ""
        wanted = (name or "").strip() or self.default_experiment
        async with self._lock:
            cached = self._experiments.get(wanted)
        if cached:
            return cached
        try:
            experiment_id = await asyncio.to_thread(self._ensure_experiment_sync, wanted)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow: no se pudo resolver el experimento '%s': %s", wanted, exc)
            return self.experiment_id or ""
        async with self._lock:
            self._experiments[wanted] = experiment_id
        return experiment_id

    def _list_experiments_sync(self) -> list[dict[str, Any]]:
        experiments = self._client.search_experiments(max_results=1000)
        rows = [
            {
                "experiment_id": e.experiment_id,
                "name": e.name,
                "url": self.experiment_url(e.experiment_id),
                "last_update_time": getattr(e, "last_update_time", None),
            }
            for e in experiments
        ]
        rows.sort(key=lambda r: r["name"].lower())
        return rows

    async def list_experiments(self) -> list[dict[str, Any]]:
        if not self.available:
            return []
        try:
            return await asyncio.to_thread(self._list_experiments_sync)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow: no se pudieron listar los experimentos: %s", exc)
            return []

    async def experiment_info(self, name: str | None) -> dict[str, Any]:
        """Datos del experimento activo de una sesion, para pintarlos en la UI."""
        wanted = (name or "").strip() or self.default_experiment
        experiment_id = await self.ensure_experiment(wanted)
        return {
            "name": wanted,
            "experiment_id": experiment_id,
            "url": self.experiment_url(experiment_id),
            "available": self.available,
        }

    # ----------------------------------------------------------------- runs -
    def _create_run(
        self, experiment_id: str, name: str, tags: dict[str, str], parent: str | None = None
    ) -> str:
        run_tags = dict(tags)
        run_tags[MLFLOW_RUN_NAME] = name
        if parent:
            run_tags[MLFLOW_PARENT_RUN_ID] = parent
        run = self._client.create_run(experiment_id=experiment_id, tags=run_tags)
        return run.info.run_id

    async def ensure_session_run(
        self,
        session_id: str,
        title: str = "",
        experiment: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Run padre de la sesion.

        Se crea con la primera interaccion, nunca antes: asi el experimento no
        se llena de runs de sesiones que no llegaron a usarse. La excepcion son
        las evaluaciones, que lo crean al arrancar para etiquetarlo (`tags`).
        """
        if not self.available:
            return ""
        experiment_id = await self.ensure_experiment(experiment)
        if not experiment_id:
            return ""
        cache_key = (experiment_id, session_id)
        async with self._lock:
            if cache_key in self._session_runs:
                return self._session_runs[cache_key]
        try:
            run_id = await asyncio.to_thread(
                self._create_run,
                experiment_id,
                f"session::{title or session_id}",
                {
                    **(tags or {}),
                    "session_id": session_id,
                    "level": "session",
                    "app": "agente-pruebas-mcp",
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow: no se pudo crear el run de sesion: %s", exc)
            return ""
        async with self._lock:
            self._session_runs[cache_key] = run_id
        return run_id

    async def ensure_conversation_run(
        self,
        session_id: str,
        conversation_id: str,
        title: str = "",
        params: dict[str, Any] | None = None,
        experiment: str | None = None,
        session_title: str = "",
        tags: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        """Devuelve `(run_id de la conversacion, experiment_id)`.

        `tags` solo se aplica al crear el run; si ya existe se devuelve tal cual.
        """
        if not self.available:
            return "", ""
        experiment_id = await self.ensure_experiment(experiment)
        if not experiment_id:
            return "", ""
        cache_key = (experiment_id, conversation_id)
        async with self._lock:
            existing = self._conversation_runs.get(cache_key)
        if existing:
            return existing, experiment_id

        parent = await self.ensure_session_run(session_id, session_title, experiment)
        try:
            run_id = await asyncio.to_thread(
                self._create_run,
                experiment_id,
                f"conv::{title or conversation_id}",
                {
                    **(tags or {}),
                    "session_id": session_id,
                    "conversation_id": conversation_id,
                    "level": "conversation",
                    "app": "agente-pruebas-mcp",
                },
                parent or None,
            )
            if params:
                await asyncio.to_thread(self._log_params, run_id, params)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow: no se pudo crear el run de conversacion: %s", exc)
            return "", experiment_id
        async with self._lock:
            self._conversation_runs[cache_key] = run_id
        return run_id, experiment_id

    def _log_params(self, run_id: str, params: dict[str, Any]) -> None:
        for key, value in params.items():
            try:
                self._client.log_param(run_id, key, _truncate(value, MAX_PARAM_CHARS))
            except Exception:  # noqa: BLE001 - un param repetido no debe romper nada
                continue

    async def close_conversation_run(self, conversation_id: str) -> None:
        if not self.available:
            return
        async with self._lock:
            keys = [k for k in self._conversation_runs if k[1] == conversation_id]
            run_ids = [self._conversation_runs.pop(k) for k in keys]
        for run_id in run_ids:
            try:
                await asyncio.to_thread(self._client.set_terminated, run_id, "FINISHED")
            except Exception:  # noqa: BLE001
                pass

    def _rename_session_sync(self, session_id: str, title: str) -> int:
        """Repinta el nombre del run de sesion en todos los experimentos.

        MLflow no exige que los nombres de run sean unicos —los runs se
        identifican por `run_id` y aqui siempre se buscan por `tags.session_id`,
        que si lo es—, asi que dos sesiones pueden llamarse igual sin conflicto.
        """
        renamed = 0
        experiment_ids = [e.experiment_id for e in self._client.search_experiments(max_results=1000)]
        if not experiment_ids:
            return 0
        runs = self._client.search_runs(
            experiment_ids=experiment_ids,
            filter_string=f"tags.session_id = '{session_id}' and tags.level = 'session'",
            max_results=100,
        )
        for run in runs:
            try:
                self._client.set_tag(run.info.run_id, MLFLOW_RUN_NAME, f"session::{title}")
                renamed += 1
            except Exception:  # noqa: BLE001
                continue
        return renamed

    async def rename_session(self, session_id: str, title: str) -> int:
        """Mantiene el nombre del run de MLflow al dia con el de la UI.

        Si la sesion todavia no tiene run (no ha habido interacciones), no hay
        nada que renombrar: el run se creara ya con el nombre nuevo.
        """
        if not self.available or not title.strip():
            return 0
        try:
            return await asyncio.to_thread(self._rename_session_sync, session_id, title.strip())
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow: no se pudo renombrar la sesion %s: %s", session_id, exc)
            return 0

    # -------------------------------------------------------------- borrado -
    def _search_traces(self, experiment_id: str, tag: str, value: str, page_token: Any) -> Any:
        """`search_traces` cambio de firma en MLflow 3.x (`locations`).

        Se intenta primero la nueva y se cae a la antigua para no atarse a una
        version concreta del tracking server.
        """
        criteria = {
            "filter_string": f"tags.{tag} = '{value}'",
            "max_results": 500,
            "page_token": page_token,
            "include_spans": False,
        }
        try:
            return self._client.search_traces(locations=[experiment_id], **criteria)
        except TypeError:
            return self._client.search_traces(experiment_ids=[experiment_id], **criteria)

    def _purge_sync(self, tag: str, value: str) -> dict[str, int]:
        """Borra runs y traces marcados con `tags.<tag> = value` en todo MLflow.

        Se recorren todos los experimentos porque una sesion pudo registrar en
        varios si se cambio el experimento por el camino.
        """
        deleted_runs = 0
        deleted_traces = 0

        # MLflow exporta las trazas en segundo plano. Sin este vaciado, borrar
        # una sesion justo despues de su ultimo mensaje dejaria su ultima traza
        # aterrizando en el servidor *despues* de la purga, es decir huerfana.
        try:
            self._mlflow.flush_trace_async_logging()
        except Exception:  # noqa: BLE001 - no existe en todas las versiones
            pass

        experiment_ids = [e.experiment_id for e in self._client.search_experiments(max_results=1000)]
        if not experiment_ids:
            return {"runs": 0, "traces": 0}

        # Traces primero: al borrar el run padre dejarian de ser localizables.
        for experiment_id in experiment_ids:
            try:
                trace_ids: list[str] = []
                page_token = None
                while True:
                    page = self._search_traces(experiment_id, tag, value, page_token)
                    trace_ids.extend(
                        getattr(t.info, "trace_id", None) or getattr(t.info, "request_id", "")
                        for t in page
                    )
                    page_token = getattr(page, "token", None)
                    if not page_token:
                        break
                trace_ids = [t for t in trace_ids if t]
                for start in range(0, len(trace_ids), MAX_TRACE_IDS_PER_DELETE):
                    deleted_traces += self._client.delete_traces(
                        experiment_id=experiment_id,
                        trace_ids=trace_ids[start : start + MAX_TRACE_IDS_PER_DELETE],
                    )
            except Exception as exc:  # noqa: BLE001
                logger.debug("MLflow: no se pudieron borrar traces de %s: %s", experiment_id, exc)

        # La busqueda por tag devuelve los dos niveles (sesion y conversacion),
        # asi que padres e hijos caen en la misma pasada.
        try:
            runs = self._client.search_runs(
                experiment_ids=experiment_ids,
                filter_string=f"tags.{tag} = '{value}'",
                run_view_type=3,  # ViewType.ALL: incluye los ya marcados como borrados
                max_results=5000,
            )
            for run in runs:
                try:
                    self._client.delete_run(run.info.run_id)
                    deleted_runs += 1
                except Exception:  # noqa: BLE001
                    continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow: fallo buscando runs a borrar: %s", exc)

        return {"runs": deleted_runs, "traces": deleted_traces}

    async def delete_session(self, session_id: str) -> dict[str, int]:
        """Elimina de MLflow todo lo registrado por una sesion."""
        async with self._lock:
            for key in [k for k in self._session_runs if k[1] == session_id]:
                self._session_runs.pop(key, None)
        if not self.available:
            return {"runs": 0, "traces": 0}
        try:
            result = await asyncio.to_thread(self._purge_sync, "session_id", session_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow: no se pudo purgar la sesion %s: %s", session_id, exc)
            return {"runs": 0, "traces": 0}
        logger.info(
            "MLflow: sesion %s purgada (%s runs, %s traces)",
            session_id,
            result["runs"],
            result["traces"],
        )
        return result

    async def delete_conversation(self, conversation_id: str) -> dict[str, int]:
        """Elimina de MLflow lo registrado por una conversacion."""
        async with self._lock:
            for key in [k for k in self._conversation_runs if k[1] == conversation_id]:
                self._conversation_runs.pop(key, None)
            self._steps.pop(conversation_id, None)
        if not self.available:
            return {"runs": 0, "traces": 0}
        try:
            return await asyncio.to_thread(self._purge_sync, "conversation_id", conversation_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow: no se pudo purgar la conversacion %s: %s", conversation_id, exc)
            return {"runs": 0, "traces": 0}

    # --------------------------------------------------------------- traces -
    def _stype(self, name: str) -> Any:
        if self._span_type is None:
            return name
        return getattr(self._span_type, name, name)

    async def start_interaction(
        self,
        session_id: str,
        conversation_id: str,
        interaction_id: str,
        user_message: str,
        attributes: dict[str, Any] | None = None,
        conversation_title: str = "",
        params: dict[str, Any] | None = None,
        experiment: str | None = None,
        session_title: str = "",
        extra_tags: dict[str, str] | None = None,
        run_tags: dict[str, str] | None = None,
    ) -> InteractionTrace:
        trace = InteractionTrace(session_id, conversation_id, interaction_id)
        trace.experiment = (experiment or "").strip() or self.default_experiment
        if not self.available:
            return trace

        trace.run_id, trace.experiment_id = await self.ensure_conversation_run(
            session_id,
            conversation_id,
            conversation_title,
            params,
            experiment,
            session_title,
            tags=run_tags,
        )
        async with self._lock:
            step = self._steps.get(conversation_id, 0)
            self._steps[conversation_id] = step + 1
        trace.step = step

        if self._start_span is None:
            return trace

        ids = {
            # Primero las extra: los tres identificadores nunca se pisan.
            **(extra_tags or {}),
            "session_id": session_id,
            "conversation_id": conversation_id,
            "interaction_id": interaction_id,
        }
        metadata = {
            # `mlflow.trace.session` es el campo nativo de hilo multi-turno:
            # le corresponde nuestra conversacion. La sesion (N conversaciones)
            # se filtra por tag y se agrupa por el run padre.
            META_TRACE_SESSION: conversation_id,
        }
        if trace.run_id:
            metadata[META_SOURCE_RUN] = trace.run_id

        def _open() -> Any:
            return self._start_span(
                name="agent_interaction",
                span_type=self._stype("AGENT"),
                inputs={"user_message": user_message},
                attributes={**ids, **(attributes or {})},
                tags=ids,
                metadata=metadata,
                experiment_id=trace.experiment_id or self.experiment_id,
            )

        try:
            span = await asyncio.to_thread(_open)
            trace.root = SpanHandle(span=span)
            trace.trace_id = getattr(span, "trace_id", "") or ""
        except Exception as exc:  # noqa: BLE001
            logger.debug("MLflow: no se pudo abrir el span raiz: %s", exc)
        return trace

    async def start_span(
        self,
        trace: InteractionTrace,
        name: str,
        span_type: str,
        inputs: dict[str, Any] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> SpanHandle:
        if not self.available or self._start_span is None or not trace.root.active:
            return SpanHandle()

        def _open() -> Any:
            return self._start_span(
                name=name,
                span_type=self._stype(span_type),
                parent_span=trace.root.span,
                inputs=inputs or {},
                attributes={
                    "session_id": trace.session_id,
                    "conversation_id": trace.conversation_id,
                    "interaction_id": trace.interaction_id,
                    **(attributes or {}),
                },
            )

        try:
            return SpanHandle(span=await asyncio.to_thread(_open))
        except Exception as exc:  # noqa: BLE001
            logger.debug("MLflow: no se pudo abrir el span %s: %s", name, exc)
            return SpanHandle()

    async def end_span(
        self,
        handle: SpanHandle,
        outputs: Any = None,
        attributes: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if not handle.active:
            return

        def _close() -> None:
            span = handle.span
            try:
                for key, value in (attributes or {}).items():
                    span.set_attribute(key, value)
                if error:
                    span.set_attribute("error", error)
                    try:
                        from mlflow.entities import SpanStatusCode

                        span.set_status(SpanStatusCode.ERROR)
                    except Exception:  # noqa: BLE001
                        pass
                span.end(outputs=outputs)
            except Exception:  # noqa: BLE001
                try:
                    span.end()
                except Exception:  # noqa: BLE001
                    pass

        try:
            await asyncio.to_thread(_close)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------ artifacts -
    async def log_artifact(self, run_id: str, artifact_file: str, payload: Any) -> str:
        """Sube un JSON completo (sin recortar) al run indicado.

        Es la garantia de auditoria: aunque la UI muestre un extracto, el
        contenido integro queda en el artifact store de MLflow.
        """
        if not self.available or not run_id:
            return ""
        try:
            await asyncio.to_thread(self._client.log_dict, run_id, payload, artifact_file)
            return artifact_file
        except Exception as exc:  # noqa: BLE001
            logger.debug("MLflow: no se pudo guardar el artifact %s: %s", artifact_file, exc)
            return ""

    async def log_text(self, run_id: str, artifact_file: str, text: str) -> str:
        """Igual que `log_artifact` pero para texto plano (resultados grandes)."""
        if not self.available or not run_id:
            return ""
        try:
            await asyncio.to_thread(self._client.log_text, run_id, text, artifact_file)
            return artifact_file
        except Exception as exc:  # noqa: BLE001
            logger.debug("MLflow: no se pudo guardar el texto %s: %s", artifact_file, exc)
            return ""

    # ------------------------------------------------ utilidades de runs ----
    # Las usan las evaluaciones, que escriben en los runs fuera del bucle del
    # agente (resumen de la bateria, nota de cada caso...).
    async def log_metrics(
        self, run_id: str, metrics: dict[str, float], step: int | None = None
    ) -> None:
        if not self.available or not run_id:
            return

        def _log() -> None:
            for key, value in metrics.items():
                try:
                    self._client.log_metric(run_id, key, float(value), step=step or 0)
                except Exception:  # noqa: BLE001
                    continue

        try:
            await asyncio.to_thread(_log)
        except Exception as exc:  # noqa: BLE001
            logger.debug("MLflow: fallo registrando metricas en %s: %s", run_id, exc)

    async def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        if not self.available or not run_id:
            return
        try:
            await asyncio.to_thread(self._log_params, run_id, params)
        except Exception as exc:  # noqa: BLE001
            logger.debug("MLflow: fallo registrando params en %s: %s", run_id, exc)

    async def set_tags(self, run_id: str, tags: dict[str, Any]) -> None:
        if not self.available or not run_id:
            return

        def _tag() -> None:
            for key, value in tags.items():
                try:
                    self._client.set_tag(run_id, key, _truncate(value, 4900))
                except Exception:  # noqa: BLE001
                    continue

        try:
            await asyncio.to_thread(_tag)
        except Exception as exc:  # noqa: BLE001
            logger.debug("MLflow: fallo etiquetando %s: %s", run_id, exc)

    async def log_table(self, run_id: str, rows: list[dict[str, Any]], artifact_file: str) -> str:
        """Tabla visible en la pestana *Evaluation/Artifacts* de MLflow."""
        if not self.available or not run_id or not rows:
            return ""
        columns: list[str] = []
        for row in rows:
            columns.extend(k for k in row if k not in columns)
        table = {c: [row.get(c) for row in rows] for c in columns}
        try:
            await asyncio.to_thread(self._client.log_table, run_id, table, artifact_file)
            return artifact_file
        except Exception as exc:  # noqa: BLE001
            logger.debug("MLflow: no se pudo guardar la tabla %s: %s", artifact_file, exc)
            return ""

    async def terminate_run(self, run_id: str, status: str = "FINISHED") -> None:
        """Cierra un run concreto (y lo saca de la cache si era de conversacion)."""
        if not self.available or not run_id:
            return
        async with self._lock:
            for cache in (self._conversation_runs, self._session_runs):
                for key in [k for k, v in cache.items() if v == run_id]:
                    cache.pop(key, None)
        try:
            await asyncio.to_thread(self._client.set_terminated, run_id, status)
        except Exception:  # noqa: BLE001
            pass

    # -------------------------------------------------------- evaluaciones ---
    def _assessment_source(self, kind: str, source_id: str) -> Any:
        from mlflow.entities import AssessmentSource

        return AssessmentSource(source_type=kind, source_id=source_id or "desconocido")

    async def log_assessments(
        self,
        trace_id: str,
        feedbacks: list[dict[str, Any]],
        expectations: list[dict[str, Any]] | None = None,
    ) -> int:
        """Cuelga juicios y valores esperados de una traza (API nativa de MLflow).

        Cada elemento de `feedbacks` lleva `name`, `value`, `rationale`,
        `source_type` (LLM_JUDGE | CODE | HUMAN), `source_id` y `metadata`.
        Devuelve cuantos se registraron.
        """
        if not self.available or not trace_id or self._mlflow is None:
            return 0
        log_feedback = getattr(self._mlflow, "log_feedback", None)
        log_expectation = getattr(self._mlflow, "log_expectation", None)
        if log_feedback is None:
            return 0

        def _log() -> int:
            # La traza se exporta en segundo plano: tiene que existir en el
            # servidor antes de poder colgarle nada.
            try:
                self._mlflow.flush_trace_async_logging()
            except Exception:  # noqa: BLE001
                pass
            done = 0
            for item in expectations or []:
                if log_expectation is None:
                    break
                try:
                    log_expectation(
                        trace_id=trace_id,
                        name=item["name"],
                        value=item["value"],
                        source=self._assessment_source(
                            item.get("source_type", "HUMAN"), item.get("source_id", "")
                        ),
                        metadata=item.get("metadata"),
                    )
                    done += 1
                except Exception as exc:  # noqa: BLE001
                    logger.debug("MLflow: expectation %s rechazada: %s", item.get("name"), exc)
            for item in feedbacks:
                try:
                    log_feedback(
                        trace_id=trace_id,
                        name=item["name"],
                        value=item["value"],
                        rationale=item.get("rationale") or None,
                        source=self._assessment_source(
                            item.get("source_type", "LLM_JUDGE"), item.get("source_id", "")
                        ),
                        metadata=item.get("metadata"),
                    )
                    done += 1
                except Exception as exc:  # noqa: BLE001
                    logger.debug("MLflow: feedback %s rechazado: %s", item.get("name"), exc)
            return done

        try:
            return await asyncio.to_thread(_log)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow: no se pudieron registrar los assessments de %s: %s", trace_id, exc)
            return 0

    async def log_standalone_trace(
        self,
        name: str,
        span_type: str,
        inputs: Any,
        outputs: Any,
        *,
        experiment: str | None = None,
        tags: dict[str, str] | None = None,
        attributes: dict[str, Any] | None = None,
        source_run_id: str = "",
        error: str | None = None,
    ) -> str:
        """Traza de un solo span para un paso que no es del agente (el evaluador).

        Lleva las mismas etiquetas de sesion/conversacion, asi que se filtra y
        se purga igual que las trazas del agente.
        """
        if not self.available or self._start_span is None:
            return ""
        experiment_id = await self.ensure_experiment(experiment)
        metadata = {META_SOURCE_RUN: source_run_id} if source_run_id else {}

        def _open() -> Any:
            return self._start_span(
                name=name,
                span_type=self._stype(span_type),
                inputs=inputs,
                attributes={**(tags or {}), **(attributes or {})},
                tags=tags or {},
                metadata=metadata,
                experiment_id=experiment_id or self.experiment_id,
            )

        try:
            span = await asyncio.to_thread(_open)
        except Exception as exc:  # noqa: BLE001
            logger.debug("MLflow: no se pudo abrir la traza %s: %s", name, exc)
            return ""
        await self.end_span(SpanHandle(span=span), outputs=outputs, error=error)
        return getattr(span, "trace_id", "") or ""

    # ------------------------------------------------------------- cierre ---
    async def end_interaction(
        self,
        trace: InteractionTrace,
        final_answer: str,
        metrics: dict[str, float],
        artifact: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if not self.available:
            return

        await self.end_span(
            trace.root,
            # Sin recortar: la respuesta final se guarda entera.
            outputs={"final_answer": final_answer},
            attributes={f"metric.{k}": v for k, v in metrics.items()},
            error=error,
        )

        def _log() -> None:
            if trace.run_id:
                for key, value in metrics.items():
                    try:
                        self._client.log_metric(trace.run_id, key, float(value), step=trace.step)
                    except Exception:  # noqa: BLE001
                        continue
                if artifact is not None:
                    try:
                        self._client.log_dict(
                            trace.run_id,
                            artifact,
                            f"interactions/{trace.interaction_id}.json",
                        )
                    except Exception:  # noqa: BLE001
                        pass
                if trace.tool_rows:
                    try:
                        self._client.log_table(
                            trace.run_id,
                            {k: [row.get(k) for row in trace.tool_rows] for k in trace.tool_rows[0]},
                            artifact_file="mcp_tool_calls.json",
                        )
                    except Exception:  # noqa: BLE001
                        pass
                if trace.trace_id:
                    try:
                        self._client.set_tag(trace.run_id, "last_trace_id", trace.trace_id)
                    except Exception:  # noqa: BLE001
                        pass

        try:
            await asyncio.to_thread(_log)
        except Exception as exc:  # noqa: BLE001
            logger.debug("MLflow: fallo registrando metricas: %s", exc)


tracker = MLflowTracker()
