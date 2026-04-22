"""
Retrain service — SOP §5 (services/retrain.py).

Generates a retrain notebook for an existing active ModelVersion applied
to a new dataset.

Retrain is supported for supervised model versions only.
Clustering model versions return HTTP 400.

generate_retrain_notebook(source_version_id, new_dataset_id, db) -> Path:
  1. Load the ModelVersion by source_version_id
  2. Retrieve its original preprocessing config via version.experiment.preprocessing_config_id
  3. Load the new dataset file
  4. Generate a .ipynb that:
     a) Loads the new dataset
     b) Reconstructs the identical preprocessing pipeline from the original config
     c) Applies the same model class with the same parameters_json (no tuning)
     d) Produces results.json in the standard supervised schema
  5. Write to storage/notebooks/retrain_{source_version_id}_{new_dataset_id}.ipynb

On results upload, the orchestrator calls create_version() normally —
a new ModelVersion is created with version_number incremented, linked to
the same model_name. The two versions are diffable via the existing diff endpoint.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.core.models import Dataset, Experiment, ModelVersion, PreprocessingConfig
from backend.services.notebook_gen import (
    _supervised_install_cell,
    _supervised_load_cell,
    _supervised_preprocessing_cell,
)


# ---------------------------------------------------------------------------
# Model class → sklearn import mapping
# ---------------------------------------------------------------------------

_MODEL_IMPORTS = {
    "LogisticRegression":         "from sklearn.linear_model import LogisticRegression",
    "Ridge":                      "from sklearn.linear_model import Ridge",
    "Lasso":                      "from sklearn.linear_model import Lasso",
    "RandomForestClassifier":     "from sklearn.ensemble import RandomForestClassifier",
    "RandomForestRegressor":      "from sklearn.ensemble import RandomForestRegressor",
    "GradientBoostingClassifier": "from sklearn.ensemble import GradientBoostingClassifier",
    "GradientBoostingRegressor":  "from sklearn.ensemble import GradientBoostingRegressor",
    "SVC":                        "from sklearn.svm import SVC",
    "SVR":                        "from sklearn.svm import SVR",
    "KNeighborsClassifier":       "from sklearn.neighbors import KNeighborsClassifier",
    "KNeighborsRegressor":        "from sklearn.neighbors import KNeighborsRegressor",
}


def _retrain_train_cell(
    model_name: str,
    parameters: dict,
    problem_type: str,
    source_version_id: str,
) -> nbformat.NotebookNode:
    import_line = _MODEL_IMPORTS.get(model_name, f"# Unknown model: {model_name}")
    lines = [
        "import time, json, joblib",
        import_line,
        "from sklearn.model_selection import train_test_split, cross_val_score",
        "",
        f"# Retrain: same model class + same parameters as source version {source_version_id}",
        f"model = {model_name}(**{parameters!r})",
        "",
    ]
    if problem_type == "classification":
        lines += [
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)",
            "X_train_proc = preprocessor.fit_transform(X_train, y_train)",
            "X_test_proc  = preprocessor.transform(X_test)",
            "model.fit(X_train_proc, y_train)",
            "from sklearn.metrics import accuracy_score, f1_score",
            "_cv = cross_val_score(model, X_train_proc, y_train, cv=5, scoring='accuracy')",
            "_preds = model.predict(X_test_proc)",
            "_metrics = {'accuracy': accuracy_score(y_test, _preds), 'f1_macro': f1_score(y_test, _preds, average='macro', zero_division=0)}",
            "_cm = []",
            "try:",
            "    from sklearn.metrics import confusion_matrix",
            "    _cm = confusion_matrix(y_test, _preds).tolist()",
            "except: pass",
        ]
    else:  # regression
        lines += [
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)",
            "X_train_proc = preprocessor.fit_transform(X_train, y_train)",
            "X_test_proc  = preprocessor.transform(X_test)",
            "model.fit(X_train_proc, y_train)",
            "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score",
            "_cv = cross_val_score(model, X_train_proc, y_train, cv=5, scoring='r2')",
            "_preds = model.predict(X_test_proc)",
            "_metrics = {'mse': mean_squared_error(y_test,_preds), 'mae': mean_absolute_error(y_test,_preds), 'r2': r2_score(y_test,_preds)}",
            "_cm = []",
        ]

    lines += [
        "_t_end = time.time()",
        "import datetime",
        "_out = {",
        f"    'experiment_id': 'retrain_{source_version_id}',",
        f"    'problem_type': '{problem_type}',",
        "    'models': [{'name': '" + model_name + "', 'parameters': " + repr(parameters) + ",",
        "               'best_params': {}, 'metrics': _metrics,",
        "               'cv_scores': {'mean': float(_cv.mean()), 'std': float(_cv.std()), 'folds': _cv.tolist()},",
        "               'confusion_matrix': _cm, 'feature_importances': {}}],",
        "    'training_duration_seconds': _t_end - _t0,",
        "    'timestamp': datetime.datetime.utcnow().isoformat(),",
        "}",
        "with open('results.json', 'w') as f: json.dump(_out, f, indent=2)",
        "joblib.dump(model, 'best_model.joblib')",
        "joblib.dump(preprocessor, 'preprocessor.joblib')",
        "print('Retrain complete. results.json written.')",
    ]
    return new_code_cell("\n".join(lines))


async def generate_retrain_notebook(
    source_version_id: str,
    new_dataset_id: str,
    db: AsyncSession,
) -> Path:
    """
    Generate a retrain notebook and return its path.

    Raises:
        ValueError  — if source version is a clustering model (not supported)
        FileNotFoundError — if dataset file is missing
    """
    # 1. Load source ModelVersion
    mv = await db.get(ModelVersion, source_version_id)
    if mv is None:
        raise ValueError(f"ModelVersion {source_version_id!r} not found.")

    if mv.cluster_labels_path is not None:
        raise ValueError("Retraining is not supported for clustering models.")

    # 2. Load experiment → preprocessing config
    experiment = await db.get(Experiment, mv.experiment_id)
    if experiment is None:
        raise ValueError(f"Experiment for version {source_version_id!r} not found.")

    prep_config_record = await db.get(PreprocessingConfig, experiment.preprocessing_config_id)
    if prep_config_record is None:
        raise ValueError("Preprocessing config not found for this model version.")

    from backend.services.preprocessing import deserialize_full_config
    full_config = deserialize_full_config(prep_config_record.config_json)

    # Determine target column
    target_column: Optional[str] = None
    for col, cfg in full_config.columns.items():
        if cfg.is_target:
            target_column = col
            break

    # 3. Load new dataset
    new_dataset = await db.get(Dataset, new_dataset_id)
    if new_dataset is None:
        raise ValueError(f"Dataset {new_dataset_id!r} not found.")

    dataset_path = Path(new_dataset.filepath)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    # 4. Determine problem type from original experiment
    problem_type = new_dataset.problem_type.value if new_dataset.problem_type else "classification"

    # 5. Build notebook
    parameters = json.loads(mv.parameters_json) if mv.parameters_json else {}
    model_name = mv.model_name

    _t0_line = "import time; _t0 = time.time()"
    config_dict = full_config.model_dump()

    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            f"# Retrain Notebook\n"
            f"*Source version:* `{source_version_id}`  \n"
            f"*New dataset:* `{new_dataset_id}` ({new_dataset.name})  \n"
            f"*Model:* `{model_name}` (same parameters, no tuning)"
        ),
        _supervised_install_cell(),
        new_code_cell(_t0_line),
        _supervised_load_cell(dataset_path, dataset_path.suffix.lower()),
        new_code_cell(
            f"TARGET = {target_column!r}\n"
            f"X = df.drop(columns=[TARGET])\n"
            f"y = df[TARGET]"
        ),
        # Rebuild preprocessor using original config
        new_code_cell(_build_preprocessor_cell(config_dict, target_column)),
        _retrain_train_cell(model_name, parameters, problem_type, source_version_id),
    ]

    # 6. Write notebook
    storage_dir = Path(settings.storage_dir)
    output_path = storage_dir / "notebooks" / f"retrain_{source_version_id}_{new_dataset_id}.ipynb"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    return output_path


def _build_preprocessor_cell(config_dict: dict, target_column: Optional[str]) -> str:
    """Build the preprocessor reconstruction cell for the retrain notebook."""
    lines = [
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder",
        "from sklearn.impute import SimpleImputer, KNNImputer",
        "from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression",
        "import joblib",
        "",
        "# Reconstruct identical preprocessing pipeline from original config",
        "_transformers = []",
    ]
    columns_cfg = config_dict.get("columns", {})
    for col, cfg in columns_cfg.items():
        if isinstance(cfg, dict):
            col_type = cfg.get("type", "numerical")
            strategy = cfg.get("strategy", "none")
            is_target = cfg.get("is_target", False)
        else:
            continue
        if is_target or strategy == "none":
            continue
        if col_type == "numerical":
            sc = {"standardize": "StandardScaler()", "normalize": "MinMaxScaler()", "robust": "RobustScaler()"}.get(strategy)
            if sc:
                lines.append(f"_transformers.append(('num_{col}', {sc}, {[col]!r}))")
        elif col_type == "categorical":
            enc = {"onehot": "OneHotEncoder(handle_unknown='ignore',sparse_output=False)", "label": "OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)", "ordinal": "OrdinalEncoder()"}.get(strategy)
            if enc:
                lines.append(f"_transformers.append(('cat_{col}', {enc}, {[col]!r}))")

    fs = config_dict.get("feature_selection", {})
    fs_method = fs.get("method", "none") if isinstance(fs, dict) else "none"

    lines += [
        "_ct = ColumnTransformer(_transformers, remainder='drop')",
        "_steps = [('preprocessor', _ct)]",
    ]
    if fs_method == "select_k_best":
        lines.append(f"_steps.append(('selector', SelectKBest(f_classif, k={fs.get('k', 10)})))")
    elif fs_method == "variance_threshold":
        lines.append(f"_steps.append(('selector', VarianceThreshold(threshold={fs.get('variance_threshold', 0.0)})))")

    lines += [
        "preprocessor = Pipeline(_steps)",
        "print('Preprocessor (retrain) ready.')",
    ]
    return "\n".join(lines)
