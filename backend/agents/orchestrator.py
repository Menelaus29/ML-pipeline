"""
Orchestrator — SOP §6 (Ph 7). Cite: Yao et al. (2022) ReAct multi-agent pattern.

Coordinates the full experiment lifecycle:
  1. AnalysisAgent  — EDA + preprocessing recommendation
  2. Notebook gen   — generate .ipynb (already triggered as background task by experiments API)
  3. TrainingMonitor — poll status until completed
  4. InsightAgent   — interpret uploaded results

run_experiment() is called as an asyncio background task from the experiments router.
It updates experiment.status at each milestone and logs progress via AgentLog / SSE.
"""
from __future__ import annotations
import asyncio
import logging
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.base import BaseAgent
from backend.agents.analysis_agent import AnalysisAgent
from backend.agents.training_monitor import TrainingMonitor
from backend.agents.insight_agent import InsightAgent
from backend.core.models import Experiment, ExperimentStatus, MessageType
from backend.core.utils import utcnow

logger = logging.getLogger(__name__)


class Orchestrator(BaseAgent):
    name = "orchestrator"

    def __init__(self, db: AsyncSession, experiment_id: str):
        super().__init__(db=db, experiment_id=experiment_id)

    async def run(self, parsed_results: dict | None = None) -> dict:
        """
        Full orchestration pipeline.

        Parameters
        ----------
        parsed_results : If provided (post-results-upload), skip straight to InsightAgent.
                         If None, run AnalysisAgent → monitor → (await results upload).
        """
        experiment = await self.db.get(Experiment, self.experiment_id)
        if experiment is None:
            logger.error("Orchestrator: experiment %s not found", self.experiment_id)
            return {"status": "error"}

        dataset_id = experiment.dataset_id

        # ── Phase A: Analysis ───────────────────────────────────────────
        await self._log("Orchestrator: starting analysis phase")
        analysis_agent = AnalysisAgent(db=self.db, experiment_id=self.experiment_id)
        try:
            analysis_result = await analysis_agent.run(dataset_id=dataset_id)
        except Exception as e:
            await self._log(f"Analysis phase error: {e}", MessageType.warning)
            analysis_result = {}

        # ── Phase B: Notebook gen runs as background task (already queued) ──
        await self._log(
            "Notebook generation queued. Download from /api/experiments/{id}/notebook "
            "and run all cells, then upload results.json."
        )
        experiment.status = ExperimentStatus.notebook_generated
        await self.db.flush()

        # ── Phase C: Training monitor ───────────────────────────────────
        if parsed_results is None:
            monitor = TrainingMonitor(db=self.db, experiment_id=self.experiment_id)
            monitor_result = await monitor.run()
            if monitor_result.get("status") != "completed":
                await self._log(
                    f"Monitoring ended with status: {monitor_result.get('status')}",
                    MessageType.warning,
                )
                return {"status": monitor_result.get("status")}
            # Re-fetch results from DB after upload
            await self.db.refresh(experiment)

        # ── Phase D: Insight ────────────────────────────────────────────
        if parsed_results is not None:
            await self._log("Orchestrator: starting insight phase")
            insight_agent = InsightAgent(db=self.db, experiment_id=self.experiment_id)
            try:
                insight_result = await insight_agent.run(parsed_results=parsed_results)
            except Exception as e:
                await self._log(f"Insight phase error: {e}", MessageType.warning)
                insight_result = {}
        else:
            insight_result = {}

        await self._log("Orchestrator: all phases complete.")
        experiment.status = ExperimentStatus.completed
        experiment.completed_at = utcnow()
        await self.db.commit()

        return {
            "experiment_id": self.experiment_id,
            "analysis": analysis_result,
            "insight": insight_result,
        }


async def run_experiment_background(experiment_id: str, parsed_results: dict | None = None) -> None:
    """
    Entry point for background task invocation.
    Creates its own DB session so it can run independently of the request lifecycle.
    """
    from backend.core.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        orchestrator = Orchestrator(db=db, experiment_id=experiment_id)
        try:
            await orchestrator.run(parsed_results=parsed_results)
        except Exception as e:
            logger.exception("Orchestrator background task failed for %s: %s", experiment_id, e)
