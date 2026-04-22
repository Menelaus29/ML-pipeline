"""
AnalysisAgent — SOP §6 (Ph 3).

Triggered at the start of an experiment.
1. Loads EDA features from the dataset.
2. Builds a structured prompt describing data quality findings.
3. Calls Ollama to generate a preprocessing recommendation narrative.
4. Persists the narrative as an AgentLog with message_type=insight.
5. Returns a structured dict with EDA findings + narrative.

Supports both supervised and clustering problem types.
For clustering: class_distribution is omitted from the prompt.

Cite: Yao et al. (2022) ReAct — observations (EDA data) → reasoning → action (narrative).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.base import BaseAgent
from backend.core.models import Dataset, MessageType
from backend.services.eda import compute_eda_features
from backend.services.ingestion import _load_dataframe

logger = logging.getLogger(__name__)


class AnalysisAgent(BaseAgent):
    name = "analysis_agent"

    def __init__(
        self,
        db: AsyncSession,
        experiment_id: str,
    ):
        super().__init__(db=db, experiment_id=experiment_id)

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt(self, eda: dict, problem_type: str) -> str:
        null_lines = "\n".join(
            f"  - {col}: {rate*100:.1f}% missing"
            for col, rate in eda.get("null_severity", {}).items()
        )
        outlier_lines = "\n".join(
            f"  - {col}: {info['iqr_outlier_count']} outliers ({info['pct']}%)"
            for col, info in eda.get("outlier_flags", {}).items()
        )
        high_card = ", ".join(eda.get("high_cardinality", [])) or "None"

        class_balance_section = ""
        if problem_type == "classification" and eda.get("class_distribution"):
            dist = eda["class_distribution"]
            total = sum(dist.values())
            lines = [
                f"  - {label}: {count} ({count/total*100:.1f}%)"
                for label, count in dist.items()
            ]
            class_balance_section = (
                "Class distribution:\n" + "\n".join(lines) + "\n\n"
            )
        elif problem_type == "clustering":
            class_balance_section = (
                "Note: this is a clustering problem — no target column or class labels.\n"
                "Focus on feature quality, null rates, and outlier handling.\n\n"
            )

        prompt = (
            f"You are an expert ML data analyst. Analyse the following dataset profile "
            f"for a {problem_type} problem and recommend preprocessing steps.\n\n"
            f"Dataset: {eda['row_count']} rows × {eda['column_count']} columns\n\n"
            f"{'Null / missing values:' + chr(10) + null_lines + chr(10) + chr(10) if null_lines else 'No missing values detected.'+chr(10)+chr(10)}"
            f"{'Outlier flags (IQR):' + chr(10) + outlier_lines + chr(10) + chr(10) if outlier_lines else 'No major outliers detected.'+chr(10)+chr(10)}"
            f"High-cardinality object columns (>=20 unique values): {high_card}\n\n"
            f"{class_balance_section}"
            f"Provide concise, actionable preprocessing recommendations covering:\n"
            f"1. Missing value imputation strategy per column type\n"
            f"2. Outlier treatment method (winsorise, IQR remove, Z-score remove, or none)\n"
            f"3. Feature encoding strategy for categorical and text columns\n"
            f"{'4. Class balancing recommendation (SMOTE, oversample, undersample, class_weight, or none)' if problem_type == 'classification' else '4. Feature selection recommendation (variance threshold or none — note: SelectKBest requires a target label so is not suitable for clustering)'}\n"
            f"\nBe specific and brief. Use bullet points."
        )
        return prompt

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    async def run(self, dataset_id: str) -> dict:
        """
        Run EDA + LLM analysis for the given dataset.

        Returns
        -------
        dict with keys: eda_features, narrative, experiment_id
        """
        await self._log(f"Analysis started for dataset {dataset_id}")

        # Load dataset
        dataset = await self.db.get(Dataset, dataset_id)
        if dataset is None:
            msg = f"Dataset {dataset_id} not found."
            await self._log(msg, MessageType.warning)
            raise ValueError(msg)

        dataset_path = Path(dataset.filepath)
        if not dataset_path.exists():
            msg = f"Dataset file not found: {dataset_path}"
            await self._log(msg, MessageType.warning)
            raise FileNotFoundError(msg)

        problem_type = dataset.problem_type.value if dataset.problem_type else "classification"
        target_column = dataset.target_column

        await self._log(f"Loading dataset ({dataset.row_count} rows, {dataset.column_count} cols)")
        df = _load_dataframe(dataset_path)
        eda = compute_eda_features(df, target_column=target_column, problem_type=problem_type)

        await self._log(
            f"EDA complete — {len(eda['null_severity'])} columns with nulls, "
            f"{len(eda['outlier_flags'])} columns with outliers"
        )

        # Call LLM
        await self._log("Calling LLM for preprocessing recommendation…")
        prompt = self._build_prompt(eda, problem_type)

        try:
            narrative = await self._call_ollama_full(prompt, temperature=0.3, max_tokens=512)
        except RuntimeError as e:
            narrative = f"[LLM unavailable: {e}]"
            await self._log(str(e), MessageType.warning)

        await self._log(narrative, MessageType.insight)
        await self._log("Analysis agent complete.")

        return {
            "experiment_id": self.experiment_id,
            "dataset_id": dataset_id,
            "eda_features": eda,
            "narrative": narrative,
        }
