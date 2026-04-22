"""
Results parser service — SOP §5.
Validates uploaded results.json against the supervised or clustering schema.
Returns structured parsed data ready for versioning.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Schema validators
# ---------------------------------------------------------------------------

_SUPERVISED_MODEL_KEYS = {"name", "parameters", "metrics", "cv_scores"}
_CLUSTERING_MODEL_KEYS = {"name", "parameters", "metrics"}


def _require_keys(obj: dict, keys: set, context: str) -> None:
    missing = keys - obj.keys()
    if missing:
        raise ValueError(f"Missing required keys in {context}: {missing}")


def parse_supervised_results(raw: dict) -> dict:
    """
    Validate and normalise a supervised results.json payload.

    Expected top-level keys: experiment_id, problem_type, models,
    training_duration_seconds, timestamp.
    """
    _require_keys(raw, {"experiment_id", "problem_type", "models", "training_duration_seconds"}, "results.json")

    if raw["problem_type"] not in ("classification", "regression"):
        raise ValueError(
            f"Expected problem_type 'classification' or 'regression', "
            f"got {raw['problem_type']!r}."
        )

    models = raw.get("models", [])
    if not isinstance(models, list) or len(models) == 0:
        raise ValueError("results.json 'models' must be a non-empty list.")

    parsed_models = []
    for m in models:
        _require_keys(m, _SUPERVISED_MODEL_KEYS, f"model entry '{m.get('name', '?')}'")
        cv = m.get("cv_scores", {})
        parsed_models.append({
            "name": str(m["name"]),
            "parameters": m.get("parameters", {}),
            "best_params": m.get("best_params", {}),
            "metrics": m.get("metrics", {}),
            "cv_scores": {
                "mean": float(cv.get("mean", 0.0)),
                "std":  float(cv.get("std", 0.0)),
                "folds": [float(v) for v in cv.get("folds", [])],
            },
            "confusion_matrix": m.get("confusion_matrix", []),
            "roc_curve": m.get("roc_curve", {}),
            "feature_importances": m.get("feature_importances", {}),
        })

    return {
        "experiment_id": str(raw["experiment_id"]),
        "problem_type": raw["problem_type"],
        "models": parsed_models,
        "training_duration_seconds": float(raw.get("training_duration_seconds", 0)),
        "timestamp": str(raw.get("timestamp", "")),
    }


def parse_clustering_results(raw: dict) -> dict:
    """
    Validate and normalise a clustering results.json payload (SOP §5 ★ new schema).
    """
    _require_keys(raw, {"experiment_id", "problem_type", "models", "training_duration_seconds"}, "results.json")

    if raw["problem_type"] != "clustering":
        raise ValueError(f"Expected problem_type 'clustering', got {raw['problem_type']!r}.")

    models = raw.get("models", [])
    if not isinstance(models, list) or len(models) == 0:
        raise ValueError("results.json 'models' must be a non-empty list.")

    parsed_models = []
    for m in models:
        _require_keys(m, _CLUSTERING_MODEL_KEYS, f"model entry '{m.get('name', '?')}'")
        metrics = m.get("metrics", {})
        parsed_models.append({
            "name": str(m["name"]),
            "parameters": m.get("parameters", {}),
            "metrics": {
                "silhouette_score":    metrics.get("silhouette_score"),
                "davies_bouldin_score": metrics.get("davies_bouldin_score"),
                "n_clusters_found":    int(metrics.get("n_clusters_found", 0)),
                "noise_points":        int(metrics.get("noise_points", 0)),
                "inertia":             metrics.get("inertia"),
            },
            "elbow_data": m.get("elbow_data"),           # K-Means only; None otherwise
            "cluster_label_counts": m.get("cluster_label_counts", {}),
            "pca_projection": m.get("pca_projection", {}),
        })

    return {
        "experiment_id": str(raw["experiment_id"]),
        "problem_type": "clustering",
        "models": parsed_models,
        "training_duration_seconds": float(raw.get("training_duration_seconds", 0)),
        "timestamp": str(raw.get("timestamp", "")),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_results(results_json_bytes: bytes) -> dict:
    """
    Parse and validate a results.json upload.
    Detects problem_type and routes to the correct validator.

    Returns the normalised results dict.
    Raises ValueError for schema violations.
    """
    try:
        raw = json.loads(results_json_bytes)
    except json.JSONDecodeError as e:
        raise ValueError(f"results.json is not valid JSON: {e}") from e

    problem_type = raw.get("problem_type", "")
    if problem_type in ("classification", "regression"):
        return parse_supervised_results(raw)
    elif problem_type == "clustering":
        return parse_clustering_results(raw)
    else:
        raise ValueError(
            f"Unknown problem_type in results.json: {problem_type!r}. "
            "Expected 'classification', 'regression', or 'clustering'."
        )


def extract_best_model(parsed: dict) -> dict:
    """
    Return the model entry with the best primary metric.
    For supervised: highest first numeric metric value.
    For clustering: highest silhouette_score (or 0 if unavailable).
    """
    models = parsed.get("models", [])
    if not models:
        return {}

    if parsed["problem_type"] == "clustering":
        return max(
            models,
            key=lambda m: m["metrics"].get("silhouette_score") or 0.0,
        )

    # Supervised: pick the model with the highest value of the first metric
    def _primary(m: dict) -> float:
        metrics = m.get("metrics", {})
        if not metrics:
            return 0.0
        return float(list(metrics.values())[0])

    return max(models, key=_primary)
