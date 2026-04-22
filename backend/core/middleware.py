"""
Logging middleware — SOP §5 / Ph 12.

Logs every HTTP request as a JSON line to:
  1. stdout (structured, picked up by Docker / cloud logging)
  2. storage/logs/api.log (rotating file, 5 × 10 MB)

JSON line format:
  {
    "ts": "2025-01-01T00:00:00+07:00",
    "method": "POST",
    "path": "/api/experiments/",
    "status": 201,
    "duration_ms": 42.3,
    "client": "127.0.0.1"
  }
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import time
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from backend.core.utils import to_utc7
from datetime import timezone

logger = logging.getLogger("ml_platform.access")


def _setup_rotating_file_handler(log_dir: Path) -> None:
    """
    Attach a RotatingFileHandler to the access logger if not already present.
    Called once from main.py lifespan after storage dirs are created.
    """
    log_path = log_dir / "api.log"
    if any(isinstance(h, logging.handlers.RotatingFileHandler) for h in logger.handlers):
        return

    fh = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(logging.Formatter("%(message)s"))  # raw JSON lines
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Don't double-print to root logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that logs every request as a structured JSON line.
    Skips /docs, /redoc, /openapi.json to keep logs clean.
    """

    _SKIP_PATHS = frozenset(["/docs", "/redoc", "/openapi.json", "/favicon.ico"])

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in self._SKIP_PATHS:
            return await call_next(request)

        t0 = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)

        from datetime import datetime
        ts = to_utc7(datetime.now(tz=timezone.utc)).isoformat()

        record = {
            "ts": ts,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
            "client": request.client.host if request.client else "unknown",
        }
        logger.info(json.dumps(record))
        return response
