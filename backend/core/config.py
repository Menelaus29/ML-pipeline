from pydantic_settings import BaseSettings, SettingsConfigDict
from backend.core.utils import UTC7Formatter
import logging

def configure_logging(log_level: str = "INFO") -> None:
    # Configure root logger with UTC+7 timestamps
    handler = logging.StreamHandler()
    handler.setFormatter(UTC7Formatter(fmt="%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.basicConfig(level=log_level, handlers=[handler])

class Settings(BaseSettings):
    # Application settings loaded from environment variables.

    # Ollama LLM
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"

    # Database
    database_url: str = "sqlite+aiosqlite:///./storage/ml_platform.db"

    # Storage
    storage_dir: str = "./storage"

    # Logging
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()