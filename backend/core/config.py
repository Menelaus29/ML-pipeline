from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Falls back to defaults if variables are not set.
    During local dev: values come from .env file.
    Inside Docker: values come from docker-compose env_file.
    """

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


# Single instance imported everywhere in the app
settings = Settings()