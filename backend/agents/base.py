"""
BaseAgent — SOP §6.

All agents inherit from this class. Provides:
  - _log(message, type)     persist AgentLog to DB + broadcast via SSE
  - _call_ollama(prompt)    async generator streaming tokens from Ollama
  - run(...)                abstract method every subclass must implement

Ollama is always called on the HOST (not inside the container) via OLLAMA_URL.
This preserves GPU hardware acceleration on the host machine.
See SOP §3 note on Ollama-on-host architecture decision.
"""
from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.core.models import AgentLog, MessageType
from backend.core.utils import utcnow

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all ML Platform agents.

    Subclasses implement run() and call _log() / _call_ollama() as needed.
    """

    name: str = "base_agent"

    def __init__(self, db: AsyncSession, experiment_id: Optional[str] = None):
        self.db = db
        self.experiment_id = experiment_id

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    async def _log(
        self,
        message: str,
        message_type: MessageType = MessageType.info,
    ) -> AgentLog:
        """
        Persist an AgentLog record and broadcast to SSE subscribers.
        Returns the created AgentLog for reference.
        """
        log = AgentLog(
            experiment_id=self.experiment_id,
            agent_name=self.name,
            message=message,
            message_type=message_type,
            created_at=utcnow(),
        )
        self.db.add(log)
        await self.db.flush()   # Assign id without committing the outer transaction
        await self.db.refresh(log)

        # Broadcast to SSE subscribers (non-blocking)
        from backend.api.agents import broadcast_log
        broadcast_log(log)

        logger.info("[%s] %s", self.name, message)
        return log

    # ------------------------------------------------------------------
    # Ollama LLM call
    # ------------------------------------------------------------------

    async def _call_ollama(
        self,
        prompt: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """
        Async generator that streams response tokens from Ollama /api/generate.

        Cite: Yao et al. (2022) ReAct — agent reasoning + acting pattern.
        The agent builds a prompt with observations (EDA/results data) and
        the LLM produces an action (recommendation / insight narrative).

        Yields individual text tokens as they arrive.
        Raises RuntimeError if Ollama is unreachable.
        """
        payload = {
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{settings.ollama_url}/api/generate",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            break
        except httpx.ConnectError as e:
            msg = (
                f"Cannot reach Ollama at {settings.ollama_url}. "
                "Ensure Ollama is running on the host machine."
            )
            logger.error(msg)
            raise RuntimeError(msg) from e

    async def _call_ollama_full(
        self,
        prompt: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """
        Convenience wrapper: collect all streamed tokens into a single string.
        """
        tokens = []
        async for token in self._call_ollama(prompt, temperature=temperature, max_tokens=max_tokens):
            tokens.append(token)
        return "".join(tokens)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def run(self, **kwargs) -> dict:
        """
        Execute the agent's primary task.
        Must call _log() at key milestones and return a result dict.
        """
        ...
