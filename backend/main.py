from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import configure_logging, settings
from backend.core.database import init_db
from backend.core.middleware import LoggingMiddleware, _setup_rotating_file_handler

# API routers
from backend.api.datasets import router as datasets_router
from backend.api.pipelines import router as pipelines_router
from backend.api.experiments import router as experiments_router
from backend.api.models import router as models_router
from backend.api.agents import router as agents_router
from backend.api.predictions import router as predictions_router

_STORAGE_SUBDIRS = [
    "datasets",
    "artifacts",
    "notebooks",
    "profiles",
    "logs",
    "cluster_labels",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run startup tasks before the app begins serving requests
    configure_logging(settings.log_level)
    Path(settings.storage_dir).mkdir(parents=True, exist_ok=True)
    await init_db()
    storage = Path(settings.storage_dir)
    for subdir in _STORAGE_SUBDIRS:
        (storage / subdir).mkdir(parents=True, exist_ok=True)

    # Setup rotating file handler for access logs (must run after logs/ dir is created)
    _setup_rotating_file_handler(storage / "logs")

    yield


app = FastAPI(
    title="ML Pipeline Platform",
    description="AI-driven ML pipeline platform with multi-agent orchestration.",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware — order matters: CORS first, then logging
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)

# Routers
app.include_router(datasets_router)
app.include_router(pipelines_router)
app.include_router(experiments_router)
app.include_router(models_router)
app.include_router(agents_router)
app.include_router(predictions_router)