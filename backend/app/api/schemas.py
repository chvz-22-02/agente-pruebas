"""Esquemas de entrada/salida de la API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ConnectMCPRequest(BaseModel):
    """El enlace del MCP llega siempre desde la UI, nunca del entorno."""

    url: str = Field(..., description="URL del servidor MCP, p.ej. http://localhost:3000/mcp")
    name: str = ""
    transport: Literal["auto", "streamable_http", "sse"] = "auto"
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_s: float = 60.0
    capture_raw: bool = True
    reuse: bool = True


class DirectToolCallRequest(BaseModel):
    """Invocacion manual de una herramienta desde la UI, sin pasar por el LLM."""

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ReadResourceRequest(BaseModel):
    uri: str


class CreateSessionRequest(BaseModel):
    title: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Experimento de MLflow donde registrar esta sesion. Si el nombre no
    # existe, se crea. Vacio => el configurado en el backend.
    mlflow_experiment: str = ""


class UpdateSessionRequest(BaseModel):
    title: str | None = None
    mlflow_experiment: str | None = None


class CreateConversationRequest(BaseModel):
    session_id: str | None = None
    title: str = ""
    provider: str = ""
    model: str = ""
    system_prompt: str = ""
    mcp_conn_ids: list[str] = Field(default_factory=list)


class UpdateConversationRequest(BaseModel):
    title: str | None = None
    model: str | None = None
    provider: str | None = None
    system_prompt: str | None = None


class PullModelRequest(BaseModel):
    """Descarga de un modelo en el motor local, lanzada desde la UI."""

    model: str = Field(..., description="Etiqueta del modelo, p.ej. qwen3:4b")
    provider: str | None = None
    base_url: str | None = None


class EnsureExperimentRequest(BaseModel):
    """Selecciona el experimento de MLflow desde la UI, creandolo si no existe."""

    name: str = Field(..., description="Nombre del experimento de MLflow")


class ProbeLLMRequest(BaseModel):
    """Comprueba credenciales y lista los modelos reales del proveedor.

    Va por POST y no por GET a proposito: una clave de API en la query string
    acabaria en los logs de acceso y en el historial del navegador.
    """

    provider: str | None = None
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None


class EvalRoleModel(BaseModel):
    """Modelo de un papel de la evaluacion (simulador o evaluador)."""

    provider: str | None = None
    base_url: str | None = None
    model: str | None = None
    # Solo vive en memoria mientras dura la ejecucion.
    api_key: str | None = None
    temperature: float | None = None
    thinking: bool | None = None
    max_tokens: int | None = None


class EvalAgentModel(EvalRoleModel):
    """El agente bajo prueba: los mismos ajustes que la pestana Modelo."""

    system_prompt: str = ""
    max_iterations: int | None = None


class ValidateEvalRequest(BaseModel):
    personas_yaml: str
    cases_yaml: str


class StartEvalRequest(BaseModel):
    personas_yaml: str
    cases_yaml: str
    name: str = ""
    # Experimento de MLflow; vacio => el del backend.
    mlflow_experiment: str = ""
    agent: EvalAgentModel = Field(default_factory=EvalAgentModel)
    # None => el mismo modelo que el agente.
    simulator: EvalRoleModel | None = None
    judge: EvalRoleModel | None = None
    mcp_conn_ids: list[str] = Field(default_factory=list)
    # Filtros de la UI; vacio => todos.
    case_ids: list[str] = Field(default_factory=list)
    persona_ids: list[str] = Field(default_factory=list)
    repetitions: int = Field(default=1, ge=1, le=20)
    max_turns_override: int | None = Field(default=None, ge=1, le=30)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    conversation_id: str | None = None
    mcp_conn_ids: list[str] = Field(default_factory=list)
    provider: str | None = None
    base_url: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    thinking: bool | None = None
    system_prompt: str = ""
    max_iterations: int | None = None
    # Experimento de MLflow elegido en la UI para este turno. Si viene, manda
    # sobre el de la sesion (y se guarda en ella).
    mlflow_experiment: str = ""
    # Clave del proveedor de nube. Se usa solo para esta peticion: ni se
    # guarda en la base de datos ni se registra en MLflow.
    api_key: str | None = None
