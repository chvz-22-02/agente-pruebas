"""Configuracion central del backend.

Todo es sobreescribible por variables de entorno o por el fichero .env.
Los valores relacionados con el MCP son deliberadamente *runtime*: se envian
desde la UI y nunca se hardcodean aqui.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Las claves de los proveedores de nube (ANTHROPIC_API_KEY, OPENAI_API_KEY,
# GOOGLE_API_KEY...) no son campos de Settings: el catalogo las busca por su
# nombre en el entorno, asi que el .env se carga tambien ahi. Sin esto, una
# clave escrita en backend/.env se ignoraba y solo servia la de la UI.
# `override=False`: una variable de entorno real manda sobre el fichero, que es
# la misma precedencia que aplica pydantic-settings.
load_dotenv(BASE_DIR / ".env", override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Servidor -----------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8090
    # Origenes permitidos para CORS. "*" acepta cualquier maquina de la red.
    # En produccion conviene listar los origenes concretos separados por comas.
    cors_origins: str = "*"

    # --- LLM ----------------------------------------------------------------
    # "ollama" | "openai_compat"  (llama.cpp server, LM Studio, vLLM, ...)
    llm_provider: str = "ollama"
    llm_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen3:8b"
    llm_api_key: str = ""  # solo para openai_compat
    llm_temperature: float = 0.6
    llm_top_p: float = 0.95
    llm_num_ctx: int = 16384  # ventana de contexto solicitada a Ollama
    llm_max_tokens: int = 4096
    llm_request_timeout: float = 600.0
    # Modo razonamiento para modelos hibridos (Qwen3, granite, gpt-oss...)
    llm_thinking: bool = True

    # --- Agente -------------------------------------------------------------
    agent_max_iterations: int = 12
    agent_history_max_messages: int = 60
    agent_system_prompt: str = ""  # vacio => usa el prompt por defecto

    # --- Persistencia -------------------------------------------------------
    db_path: str = str(DATA_DIR / "agente.sqlite3")

    # --- MLflow -------------------------------------------------------------
    mlflow_enabled: bool = True
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    # Experimento por defecto. Es solo el punto de partida: la UI puede elegir
    # otro por sesion (y crearlo si no existe), y esa eleccion se guarda en la
    # columna `sessions.mlflow_experiment`.
    mlflow_experiment: str = "agente-pruebas-mcp"
    # Si el servidor de MLflow no responde, el backend sigue funcionando.
    mlflow_fail_soft: bool = True
    # Cuanto se espera a MLflow antes de rendirse. Por defecto su cliente
    # reintenta 7 veces con backoff (minutos), y eso deja colgada cualquier
    # peticion de la UI que pase por el si el servidor se cae a media sesion.
    mlflow_http_timeout: float = 10.0
    mlflow_http_retries: int = 2

    @property
    def cors_origin_list(self) -> list[str]:
        raw = self.cors_origins.strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]


settings = Settings()
DATA_DIR.mkdir(parents=True, exist_ok=True)
