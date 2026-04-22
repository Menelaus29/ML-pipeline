"""
TrainingMonitor — SOP §6 (Ph 4).

Polls a running experiment's status and streams progress log lines to the
SSE channel while the user executes the notebook externally (e.g. in Colab
or a local Jupyter kernel).

Because the notebook runs outside the server process, the TrainingMonitor
cannot observe actual kernel progress. Instead it:
  1. Emits a structured "notebook ready" log immediately.
  2. Polls experiment.status every POLL_INTERVAL seconds.
  3. Logs each status transition.
  4. Returns when status reaches 'completed' or the timeout is exceeded.

The frontend uses the SSE stream from /api/agents/logs/{experiment_id}/stream
to show real-time progress in the Training page.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.base import BaseAgent
from backend.core.models import Experiment, ExperimentStatus, MessageType

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 5       # seconds between status checks
_MAX_WAIT_SECONDS = 7200 # 2-hour hard timeout


class TrainingMonitor(BaseAgent):
    name = "training_monitor"

    def __init__(
        self,
        db: AsyncSession,
        experiment_id: str,
    ):
        super().__init__(db=db, experiment_id=experiment_id)

    async def run(self, **kwargs) -> dict:
        """
        Monitor experiment until it reaches 'completed' status.
        Emits log lines at each status transition for the SSE stream.

        Returns dict with final status.
        """
        await self._log(
            f"Training monitor started for experiment {self.experiment_id}. "
            "Download the notebook, run all cells, then upload results.json."
        )

        last_status: Optional[str] = None
        waited = 0

        while waited < _MAX_WAIT_SECONDS:
            # Re-fetch experiment (use a fresh select to avoid stale cache)
            from sqlalchemy import select
            from backend.core.database import get_db

            result = await self.db.execute(
                select(Experiment).where(Experiment.id == self.experiment_id)
            )
            experiment = result.scalar_one_or_none()

            if experiment is None:
                await self._log(
                    f"Experiment {self.experiment_id} not found — monitor exiting.",
                    MessageType.warning,
                )
                return {"status": "error", "reason": "experiment_not_found"}

            current_status = experiment.status.value if hasattr(experiment.status, "value") else str(experiment.status)

            if current_status != last_status:
                await self._log(f"Experiment status → {current_status}")
                last_status = current_status

            if current_status == ExperimentStatus.completed.value:
                await self._log("Training complete — results received and versions created.")
                return {"status": "completed", "experiment_id": self.experiment_id}

            if current_status == "failed":
                await self._log("Training failed.", MessageType.warning)
                return {"status": "failed", "experiment_id": self.experiment_id}

            await asyncio.sleep(_POLL_INTERVAL)
            waited += _POLL_INTERVAL

        await self._log(
            f"Training monitor timed out after {_MAX_WAIT_SECONDS}s.",
            MessageType.warning,
        )
        return {"status": "timeout", "experiment_id": self.experiment_id}
