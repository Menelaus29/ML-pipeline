"""
Agents API — SOP §8.

GET  /api/agents/logs                        list paginated agent logs (optional experiment_id filter)
GET  /api/agents/logs/{experiment_id}/stream SSE stream: real-time log lines for an experiment
"""
import asyncio
import json
import logging

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_db
from backend.core.models import AgentLog
from backend.core.schemas import AgentLogRead

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agents", tags=["agents"])

# In-process SSE channel: experiment_id → list of waiting asyncio.Queue objects.
# When an agent appends a log, it puts the serialised AgentLogRead into each queue.
_sse_queues: dict[str, list[asyncio.Queue]] = {}


def broadcast_log(log: AgentLog) -> None:
    """
    Push a new AgentLog to all SSE clients subscribed to its experiment_id.
    Called from agent code immediately after db.flush() so the log id is known.
    Safe to call from any async context.
    """
    if log.experiment_id is None:
        return
    queues = _sse_queues.get(log.experiment_id, [])
    payload = json.dumps({
        "id": log.id,
        "agent_name": log.agent_name,
        "message": log.message,
        "message_type": log.message_type.value if hasattr(log.message_type, "value") else log.message_type,
        "created_at": log.created_at.isoformat() if log.created_at else None,
        "experiment_id": log.experiment_id,
    })
    for q in queues:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass  # Slow consumer — drop message rather than block


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/logs", response_model=list[AgentLogRead])
async def list_agent_logs(
    experiment_id: str | None = None,
    limit: int = Query(default=100, le=500),
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """Return paginated agent logs, newest first, optionally filtered by experiment_id."""
    q = select(AgentLog).order_by(AgentLog.created_at.desc()).limit(limit).offset(offset)
    if experiment_id:
        q = q.where(AgentLog.experiment_id == experiment_id)
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/logs/{experiment_id}/stream")
async def stream_agent_logs(experiment_id: str, request: Request):
    """
    Server-Sent Events (SSE) endpoint.
    Streams AgentLog JSON lines for experiment_id as they are created.
    The client connects and receives events in real time while the agent runs.

    Event format:
        data: {"id": "...", "agent_name": "...", "message": "...", ...}
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    _sse_queues.setdefault(experiment_id, []).append(queue)

    async def event_generator():
        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    # Send heartbeat comment to keep connection alive
                    yield ": heartbeat\n\n"
        finally:
            _sse_queues.get(experiment_id, []).remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering for SSE
        },
    )
