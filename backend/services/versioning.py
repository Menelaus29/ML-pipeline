"""
Versioning service — SOP §5 & §8 (models.diff endpoint).

Custom versioning system (no MLflow). On each result upload:
  - auto-increments version_number per model_name
  - stores full provenance (dataset_id, preprocessing_config_id, parameters,
    best_params if tuning was used, cv_metrics for supervised models)
  - one ModelVersion per model_name can be is_active at a time

Exposes a diff function: given two version IDs → structured diff of
dataset, preprocessing config, parameters, and metrics.
"""
from __future__ import annotations

import json
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.models import Experiment, ModelVersion
from backend.core.utils import utcnow


# ---------------------------------------------------------------------------
# Create version
# ---------------------------------------------------------------------------

async def get_next_version_number(
    db: AsyncSession,
    model_name: str,
) -> int:
    """Return the next version_number for model_name (max existing + 1, or 1)."""
    result = await db.execute(
        select(ModelVersion.version_number)
        .where(ModelVersion.model_name == model_name)
        .order_by(ModelVersion.version_number.desc())
        .limit(1)
    )
    row = result.scalar_one_or_none()
    return (row or 0) + 1


async def create_version(
    db: AsyncSession,
    experiment_id: str,
    model_name: str,
    artifact_path: str,
    parameters: dict,
    metrics: dict,
    cv_metrics: Optional[dict],          # None for clustering
    cluster_labels_path: Optional[str],  # None for supervised
    notes: Optional[str] = None,
) -> ModelVersion:
    """
    Create a new ModelVersion record with auto-incremented version_number.
    Sets is_active=True and deactivates any previous active version for the same model_name.
    """
    version_number = await get_next_version_number(db, model_name)

    # Deactivate previous active version for this model_name
    prev_result = await db.execute(
        select(ModelVersion)
        .where(ModelVersion.model_name == model_name, ModelVersion.is_active == True)  # noqa: E712
    )
    for prev in prev_result.scalars().all():
        prev.is_active = False

    mv = ModelVersion(
        experiment_id=experiment_id,
        model_name=model_name,
        version_number=version_number,
        artifact_path=artifact_path,
        parameters_json=json.dumps(parameters),
        metrics_json=json.dumps(metrics),
        cv_metrics_json=json.dumps(cv_metrics) if cv_metrics is not None else None,
        cluster_labels_path=cluster_labels_path,
        notes=notes,
        is_active=True,
        created_at=utcnow(),
    )
    db.add(mv)
    await db.flush()
    await db.refresh(mv)
    return mv


# ---------------------------------------------------------------------------
# Diff function
# ---------------------------------------------------------------------------

def _safe_load(json_str: Optional[str]) -> dict:
    if not json_str:
        return {}
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def _diff_dicts(a: dict, b: dict) -> dict:
    """Return {key: {v1: a[key], v2: b[key]}} for all keys where values differ."""
    all_keys = set(a) | set(b)
    return {
        k: {"v1": a.get(k), "v2": b.get(k)}
        for k in all_keys
        if a.get(k) != b.get(k)
    }


async def diff_versions(
    db: AsyncSession,
    version_id_1: str,
    version_id_2: str,
) -> dict:
    """
    Return a structured diff of two ModelVersion records.

    Returns
    -------
    dict with keys:
      - version_ids   : [v1_id, v2_id]
      - model_names   : [v1.model_name, v2.model_name]
      - dataset_diff  : {field: {v1, v2}} or None if same
      - config_diff   : {field: {v1, v2}} or None if same
      - param_diff    : {param: {v1, v2}}
      - metric_diff   : {metric: {v1, v2}}
      - cv_diff       : {metric: {v1, v2}} | None (None for clustering)
    """
    v1 = await db.get(ModelVersion, version_id_1)
    v2 = await db.get(ModelVersion, version_id_2)

    if v1 is None:
        raise ValueError(f"ModelVersion {version_id_1!r} not found.")
    if v2 is None:
        raise ValueError(f"ModelVersion {version_id_2!r} not found.")

    # Load experiments for dataset / config provenance
    exp1 = await db.get(Experiment, v1.experiment_id)
    exp2 = await db.get(Experiment, v2.experiment_id)

    def _exp_summary(exp: Optional[Experiment]) -> dict:
        if exp is None:
            return {}
        return {
            "dataset_id": exp.dataset_id,
            "preprocessing_config_id": exp.preprocessing_config_id,
        }

    e1 = _exp_summary(exp1)
    e2 = _exp_summary(exp2)

    dataset_diff = _diff_dicts(
        {"dataset_id": e1.get("dataset_id")},
        {"dataset_id": e2.get("dataset_id")},
    ) or None

    config_diff = _diff_dicts(
        {"preprocessing_config_id": e1.get("preprocessing_config_id")},
        {"preprocessing_config_id": e2.get("preprocessing_config_id")},
    ) or None

    param_diff = _diff_dicts(
        _safe_load(v1.parameters_json),
        _safe_load(v2.parameters_json),
    )

    metric_diff = _diff_dicts(
        _safe_load(v1.metrics_json),
        _safe_load(v2.metrics_json),
    )

    # CV diff (None for clustering versions)
    cv1 = _safe_load(v1.cv_metrics_json)
    cv2 = _safe_load(v2.cv_metrics_json)
    if cv1 or cv2:
        cv_diff = _diff_dicts(cv1, cv2) or None
    else:
        cv_diff = None  # both clustering — no CV data

    return {
        "version_ids": [version_id_1, version_id_2],
        "model_names": [v1.model_name, v2.model_name],
        "dataset_diff": dataset_diff,
        "config_diff": config_diff,
        "param_diff": param_diff,
        "metric_diff": metric_diff,
        "cv_diff": cv_diff,
    }


# ---------------------------------------------------------------------------
# Activate / deactivate
# ---------------------------------------------------------------------------

async def activate_version(db: AsyncSession, version_id: str) -> ModelVersion:
    """Set is_active=True on version_id and deactivate others of the same model_name."""
    mv = await db.get(ModelVersion, version_id)
    if mv is None:
        raise ValueError(f"ModelVersion {version_id!r} not found.")

    prev_result = await db.execute(
        select(ModelVersion)
        .where(ModelVersion.model_name == mv.model_name, ModelVersion.is_active == True)  # noqa: E712
    )
    for prev in prev_result.scalars().all():
        prev.is_active = False

    mv.is_active = True
    await db.flush()
    await db.refresh(mv)
    return mv
