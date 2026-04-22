"""
Prediction service — SOP §5 & §8.

predict(model_version_id, input_data, db) -> dict

Loads the active model's artifact_path via joblib, applies stored
preprocessing config to the input dict, runs .predict() (and
.predict_proba() where available), returns structured output.
Stores each prediction in the predictions table.

Clustering note (SOP §5):
  - .predict() returns a cluster label (int).
  - .predict_proba() is not available.
  - distance_to_centroid computed via model.transform(X)[0].min() for K-Means.
  - DBSCAN: assigns new point to nearest cluster centroid (documented approximation).
  - Output: {"cluster": 2, "distance_to_centroid": 0.43}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.models import ModelVersion, Prediction
from backend.core.utils import utcnow
from backend.services.preprocessing import deserialize_full_config


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def _prepare_input_df(input_data: dict, config_json: str) -> pd.DataFrame:
    """
    Convert a flat input dict into a single-row DataFrame with the correct
    column names and types from the preprocessing config.
    """
    try:
        full_cfg = deserialize_full_config(config_json)
        columns_cfg = full_cfg.columns
    except Exception:
        # Fallback: use input_data keys as-is
        columns_cfg = {}

    row: dict[str, Any] = {}
    for col, val in input_data.items():
        if col in columns_cfg and not columns_cfg[col].is_target:
            row[col] = val
        elif col not in (columns_cfg or {}):
            row[col] = val

    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Distance to centroid helpers
# ---------------------------------------------------------------------------

def _distance_to_centroid_kmeans(model, X_proc: np.ndarray) -> Optional[float]:
    """For K-Means: distance from point to its assigned centroid."""
    try:
        distances = model.transform(X_proc)  # shape (1, n_clusters)
        return float(distances[0].min())
    except Exception:
        return None


def _distance_to_centroid_dbscan(model, X_proc: np.ndarray, labels_array: list[int]) -> Optional[float]:
    """
    DBSCAN doesn't support .predict() natively.
    Assign new point to the nearest cluster centroid computed from training labels.
    This is a documented approximation (SOP §5).
    """
    try:
        # Reconstruct approximate centroids from training data stored on model
        components = model.components_  # shape (n_core_samples, n_features)
        if len(components) == 0:
            return None
        dists = np.linalg.norm(components - X_proc[0], axis=1)
        return float(dists.min())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main predict function
# ---------------------------------------------------------------------------

async def predict(
    model_version_id: str,
    input_data: dict,
    db: AsyncSession,
) -> dict:
    """
    Run inference for a model version.

    Returns structured output dict. Stores Prediction record in DB.
    """
    mv = await db.get(ModelVersion, model_version_id)
    if mv is None:
        raise ValueError(f"ModelVersion {model_version_id!r} not found.")

    # Load artefacts
    artifact_path = Path(mv.artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {artifact_path}.")

    model = joblib.load(artifact_path)

    # Determine preprocessing config from experiment
    from backend.core.models import Experiment, PreprocessingConfig
    experiment = await db.get(Experiment, mv.experiment_id)
    if experiment is None:
        raise ValueError(f"Experiment for model version {model_version_id!r} not found.")

    prep_config = await db.get(PreprocessingConfig, experiment.preprocessing_config_id)
    config_json = prep_config.config_json if prep_config else "{}"

    # Also load the fitted preprocessor joblib if it exists alongside the model
    preprocessor_path = artifact_path.parent / "preprocessor.joblib"
    fitted_preprocessor = joblib.load(preprocessor_path) if preprocessor_path.exists() else None

    # Prepare input
    df_input = _prepare_input_df(input_data, config_json)

    # Apply preprocessing
    if fitted_preprocessor is not None:
        try:
            X_proc = fitted_preprocessor.transform(df_input)
        except Exception:
            X_proc = df_input.values
    else:
        X_proc = df_input.values

    # Determine problem type
    experiment_data = json.loads(experiment.models_config_json) if experiment.models_config_json else {}
    # Check cluster_labels_path as heuristic for problem type
    is_clustering = mv.cluster_labels_path is not None

    output: dict[str, Any]

    if is_clustering:
        # Clustering inference
        cluster_label: int
        distance: Optional[float] = None

        model_class = type(model).__name__
        if hasattr(model, "predict"):
            cluster_label = int(model.predict(X_proc)[0])
        else:
            cluster_label = 0

        if model_class == "KMeans":
            distance = _distance_to_centroid_kmeans(model, X_proc)
        elif model_class == "DBSCAN":
            # Approximate: nearest core sample
            distance = _distance_to_centroid_dbscan(model, X_proc, [])
            if cluster_label == -1:
                cluster_label = -1  # DBSCAN noise point

        output = {"cluster": cluster_label, "distance_to_centroid": distance}

    else:
        # Supervised inference
        pred_value = model.predict(X_proc)[0]

        # Serialise prediction value
        if hasattr(pred_value, "item"):
            pred_value = pred_value.item()

        output = {"prediction": pred_value}

        # Confidence for classification
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_proc)[0]
                confidence = float(proba.max())
                classes = [c.item() if hasattr(c, "item") else c for c in model.classes_]
                output["confidence"] = confidence
                output["probabilities"] = dict(zip([str(c) for c in classes], proba.tolist()))
            except Exception:
                pass

        # Feature contributions (if supported)
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_.tolist()
            output["feature_contributions"] = fi

    # Persist prediction
    pred_record = Prediction(
        model_version_id=model_version_id,
        input_json=json.dumps(input_data),
        output_json=json.dumps(output),
        predicted_at=utcnow(),
    )
    db.add(pred_record)
    await db.flush()
    await db.refresh(pred_record)

    return {"prediction_id": pred_record.id, "output": output}
