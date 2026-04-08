from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import configure_logging, settings
from backend.core.database import init_db
from backend.api.datasets import router as datasets_router

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
    yield


app = FastAPI(title="ML Pipeline", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets_router)