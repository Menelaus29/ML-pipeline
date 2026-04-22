"""
Tuning service — SOP §5 Hyperparameter Tuning.

Translates a tuning config JSON into the correct sklearn call
(GridSearchCV or RandomizedSearchCV).

The tuning config structure (per-model):
  {
    "<model_name>": {
      "strategy": "grid" | "random",
      "param_grid": { ... },
      "cv_folds": 5,
      "scoring": "accuracy"
    }
  }

Cite: Bergstra & Bengio (2012) on random search.
"""
from __future__ import annotations

import json
from typing import Any

# Default n_iter for RandomizedSearchCV (SOP §5)
_DEFAULT_N_ITER = 20


def validate_tuning_config(tuning_config_json: str | None, problem_type: str) -> dict:
    """
    Validate and normalise tuning config JSON.

    - Returns empty dict if tuning_config_json is None or problem_type is 'clustering'
      (SOP §5: tuning not supported for clustering).
    - Raises ValueError for malformed configs.
    """
    if not tuning_config_json or problem_type == "clustering":
        return {}

    try:
        config = json.loads(tuning_config_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid tuning config JSON: {e}") from e

    if not isinstance(config, dict):
        raise ValueError("Tuning config must be a JSON object keyed by model name.")

    for model_name, model_cfg in config.items():
        if not isinstance(model_cfg, dict):
            raise ValueError(f"Tuning config for '{model_name}' must be an object.")
        strategy = model_cfg.get("strategy", "grid")
        if strategy not in ("grid", "random"):
            raise ValueError(
                f"Tuning strategy for '{model_name}' must be 'grid' or 'random', "
                f"got '{strategy}'."
            )
        if "param_grid" not in model_cfg:
            raise ValueError(f"Tuning config for '{model_name}' is missing 'param_grid'.")

    return config


def build_tuning_code_snippet(model_name: str, model_var: str, tuning_cfg: dict) -> str:
    """
    Generate the Python code string (to embed in a notebook cell) that wraps
    `model_var` in GridSearchCV or RandomizedSearchCV.

    Parameters
    ----------
    model_name : Human-readable model name (for comments).
    model_var  : Python variable name holding the estimator (e.g. 'model').
    tuning_cfg : Single-model tuning config dict with keys:
                   strategy, param_grid, cv_folds, scoring.

    Returns
    -------
    Python source string ready to embed in a notebook code cell.
    """
    strategy = tuning_cfg.get("strategy", "grid")
    param_grid = tuning_cfg.get("param_grid", {})
    cv_folds = tuning_cfg.get("cv_folds", 5)
    scoring = tuning_cfg.get("scoring", "accuracy")
    n_iter = tuning_cfg.get("n_iter", _DEFAULT_N_ITER)

    param_grid_repr = repr(param_grid)

    if strategy == "grid":
        return (
            f"# Hyperparameter tuning — GridSearchCV ({model_name})\n"
            f"from sklearn.model_selection import GridSearchCV\n"
            f"_tuner = GridSearchCV(\n"
            f"    {model_var},\n"
            f"    param_grid={param_grid_repr},\n"
            f"    cv={cv_folds},\n"
            f"    scoring={scoring!r},\n"
            f"    n_jobs=-1,\n"
            f"    refit=True,\n"
            f")\n"
            f"_tuner.fit(X_train_proc, y_train)\n"
            f"{model_var} = _tuner.best_estimator_\n"
            f"best_params = _tuner.best_params_\n"
            f"best_score  = _tuner.best_score_\n"
            f"print(f'Best params ({model_name}): {{best_params}}')\n"
            f"print(f'Best CV score  : {{best_score:.4f}}')\n"
        )
    else:  # random
        return (
            f"# Hyperparameter tuning — RandomizedSearchCV ({model_name})\n"
            f"# Cite: Bergstra & Bengio (2012) — random search is more efficient than grid search\n"
            f"from sklearn.model_selection import RandomizedSearchCV\n"
            f"_tuner = RandomizedSearchCV(\n"
            f"    {model_var},\n"
            f"    param_distributions={param_grid_repr},\n"
            f"    n_iter={n_iter},\n"
            f"    cv={cv_folds},\n"
            f"    scoring={scoring!r},\n"
            f"    n_jobs=-1,\n"
            f"    refit=True,\n"
            f"    random_state=42,\n"
            f")\n"
            f"_tuner.fit(X_train_proc, y_train)\n"
            f"{model_var} = _tuner.best_estimator_\n"
            f"best_params = _tuner.best_params_\n"
            f"best_score  = _tuner.best_score_\n"
            f"print(f'Best params ({model_name}): {{best_params}}')\n"
            f"print(f'Best CV score  : {{best_score:.4f}}')\n"
        )


def get_default_param_grid(model_name: str) -> dict:
    """
    Return the default tuning param_grid for a model name, per SOP §7.
    Used as a fallback when the user doesn't supply a custom grid.
    """
    defaults = {
        "LogisticRegression": {"C": [0.01, 0.1, 1, 10]},
        "RandomForestClassifier": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10],
        },
        "RandomForestRegressor": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10],
        },
        "SVC": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
        "SVR": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
        "GradientBoostingClassifier": {
            "learning_rate": [0.01, 0.1, 0.3],
            "n_estimators": [50, 100],
        },
        "GradientBoostingRegressor": {
            "learning_rate": [0.01, 0.1, 0.3],
            "n_estimators": [50, 100],
        },
        "KNeighborsClassifier": {"n_neighbors": [3, 5, 7, 11]},
        "KNeighborsRegressor": {"n_neighbors": [3, 5, 7, 11]},
        "Ridge": {"alpha": [0.1, 1, 10, 100]},
        "Lasso": {"alpha": [0.01, 0.1, 1, 10]},
    }
    return defaults.get(model_name, {})
