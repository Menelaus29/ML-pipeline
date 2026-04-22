"""
InsightAgent — SOP §6 (Ph 7).
Interprets training results via Ollama. Branches on problem_type.
Supervised: accuracy/F1/CV overfitting check. Clustering: silhouette/Davies-Bouldin/noise.
Known limitation: small LLMs (3B) may give generic clustering narratives (SOP §6).
"""
from __future__ import annotations
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from backend.agents.base import BaseAgent
from backend.core.models import MessageType

logger = logging.getLogger(__name__)


class InsightAgent(BaseAgent):
    name = "insight_agent"

    def __init__(self, db: AsyncSession, experiment_id: str):
        super().__init__(db=db, experiment_id=experiment_id)

    def _supervised_prompt(self, parsed: dict) -> str:
        models = parsed.get("models", [])
        lines = []
        for m in models:
            metrics = m.get("metrics", {})
            cv = m.get("cv_scores", {})
            lines.append(
                f"  - {m['name']}: "
                + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if v is not None)
                + (f" | CV={cv.get('mean',0):.4f}±{cv.get('std',0):.4f}" if cv else "")
            )
        return (
            f"You are an ML researcher. Interpret these {parsed.get('problem_type')} results:\n"
            + "\n".join(lines)
            + "\n\nWrite 3-5 sentences: best model + primary metric, overfitting signs (high CV std), "
            "one improvement recommendation. Academic tone, cite numbers."
        )

    def _clustering_prompt(self, parsed: dict) -> str:
        models = parsed.get("models", [])
        lines = []
        for m in models:
            mt = m.get("metrics", {})
            lines.append(
                f"  - {m['name']}: silhouette={mt.get('silhouette_score')}, "
                f"davies-bouldin={mt.get('davies_bouldin_score')}, "
                f"clusters={mt.get('n_clusters_found')}, noise={mt.get('noise_points')}"
            )
        return (
            "You are an ML researcher. Interpret these clustering results:\n"
            + "\n".join(lines)
            + "\n\nWrite 3-5 sentences: best algorithm (silhouette>0.5 is good per Rousseeuw 1987), "
            "Davies-Bouldin interpretation, DBSCAN noise assessment, one recommendation. "
            "Cite numbers. (Note: small LLMs may give generic clustering narratives — be specific.)"
        )

    async def run(self, parsed_results: dict) -> dict:
        problem_type = parsed_results.get("problem_type", "classification")
        await self._log(f"Insight agent started ({problem_type})")
        prompt = (
            self._supervised_prompt(parsed_results)
            if problem_type in ("classification", "regression")
            else self._clustering_prompt(parsed_results)
        )
        await self._log("Calling LLM for results interpretation…")
        try:
            narrative = await self._call_ollama_full(prompt, temperature=0.2, max_tokens=512)
        except RuntimeError as e:
            narrative = f"[LLM unavailable: {e}]"
            await self._log(str(e), MessageType.warning)
        await self._log(narrative, MessageType.insight)
        await self._log("Insight agent complete.")
        return {"experiment_id": self.experiment_id, "problem_type": problem_type, "narrative": narrative}
