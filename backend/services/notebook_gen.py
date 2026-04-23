"""
Notebook generator service — SOP §5.
Generates fully self-contained .ipynb files using nbformat.
Branches on problem_type: supervised (classification|regression) vs clustering.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from backend.services.tuning import build_tuning_code_snippet, validate_tuning_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_file(file_path: Path) -> str:
    return base64.b64encode(file_path.read_bytes()).decode()


def _nb_code(*lines: str) -> nbformat.NotebookNode:
    return new_code_cell("\n".join(lines))


def _nb_md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(text)


# ---------------------------------------------------------------------------
# Supervised notebook
# ---------------------------------------------------------------------------

def _supervised_install_cell() -> nbformat.NotebookNode:
    return _nb_code(
        "# Install dependencies",
        "import subprocess, sys",
        "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',",
        "    'scikit-learn', 'pandas', 'numpy', 'joblib', 'imbalanced-learn'])",
    )


def _supervised_load_cell(file_path: Path, suffix: str) -> nbformat.NotebookNode:
    encoded = _encode_file(file_path)
    loader = "pd.read_csv(io.BytesIO(raw))" if suffix == ".csv" else "pd.read_json(io.BytesIO(raw))"
    return _nb_code(
        "import base64, io, pandas as pd, numpy as np",
        f"_RAW_B64 = '{encoded}'",
        "raw = base64.b64decode(_RAW_B64)",
        f"df = {loader}",
        "print(df.shape)",
    )


def _supervised_preprocessing_cell(full_config: dict, target_column: str, problem_type: str) -> nbformat.NotebookNode:
    lines = [
        "from sklearn.model_selection import train_test_split",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder",
        "from sklearn.impute import SimpleImputer, KNNImputer",
        "from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression",
        "import joblib",
        "",
        f"TARGET = {target_column!r}",
        "X = df.drop(columns=[TARGET])",
        "y = df[TARGET]",
        "",
        "# 80/20 stratified split for classification, regular for regression",
    ]
    if problem_type == "classification":
        lines.append("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)")
    else:
        lines.append("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)")

    # Outlier treatment
    ot = full_config.get("outlier_treatment", {})
    ot_method = ot.get("method", "none") if isinstance(ot, dict) else getattr(ot, "method", "none")
    if hasattr(ot_method, "value"):
        ot_method = ot_method.value
    ot_threshold = ot.get("threshold", 1.5) if isinstance(ot, dict) else getattr(ot, "threshold", 1.5)

    if ot_method != "none":
        lines += [
            "",
            "# Outlier treatment (applied to training data only to avoid leakage)",
            f"_OT_METHOD = {ot_method!r}",
            f"_OT_THRESHOLD = {ot_threshold}",
            "_num_cols = X_train.select_dtypes(include='number').columns.tolist()",
        ]
        if ot_method == "winsorise":
            lines += [
                "_winsor_bounds = {}",
                "for _col in _num_cols:",
                "    _q1, _q3 = X_train[_col].quantile(0.25), X_train[_col].quantile(0.75)",
                "    _iqr = _q3 - _q1",
                "    _winsor_bounds[_col] = (_q1 - _OT_THRESHOLD * _iqr, _q3 + _OT_THRESHOLD * _iqr)",
                "for _col, (_lo, _hi) in _winsor_bounds.items():",
                "    X_train[_col] = X_train[_col].clip(_lo, _hi)",
                "    X_test[_col]  = X_test[_col].clip(_lo, _hi)",
            ]
        elif ot_method in ("iqr_remove", "zscore_remove"):
            lines += [
                "_keep = pd.Series([True]*len(X_train), index=X_train.index)",
                "for _col in _num_cols:",
                "    _q1, _q3 = X_train[_col].quantile(0.25), X_train[_col].quantile(0.75)",
                "    _iqr = _q3 - _q1" if ot_method == "iqr_remove" else "    _std = X_train[_col].std()",
                "    _lo = _q1 - _OT_THRESHOLD*_iqr" if ot_method == "iqr_remove" else "    _lo = X_train[_col].mean() - _OT_THRESHOLD*_std",
                "    _hi = _q3 + _OT_THRESHOLD*_iqr" if ot_method == "iqr_remove" else "    _hi = X_train[_col].mean() + _OT_THRESHOLD*_std",
                "    _keep &= (X_train[_col] >= _lo) & (X_train[_col] <= _hi)",
                "X_train = X_train[_keep].reset_index(drop=True)",
                "y_train = y_train[_keep].reset_index(drop=True)",
            ]

    # Build ColumnTransformer
    columns_cfg = full_config.get("columns", {})
    transformers = []
    for col, cfg in columns_cfg.items():
        if isinstance(cfg, dict):
            col_type = cfg.get("type", "numerical")
            strategy = cfg.get("strategy", "none")
            imputation = cfg.get("imputation", "none")
            fill_value = cfg.get("imputation_fill_value")
            is_target = cfg.get("is_target", False)
        else:
            col_type = getattr(cfg, "type", "numerical")
            strategy = getattr(cfg, "strategy", "none")
            imputation = getattr(cfg, "imputation", "none")
            fill_value = getattr(cfg, "imputation_fill_value", None)
            is_target = getattr(cfg, "is_target", False)
            if hasattr(col_type, "value"): col_type = col_type.value
            if hasattr(strategy, "value"): strategy = strategy.value
            if hasattr(imputation, "value"): imputation = imputation.value
        if is_target or strategy == "none":
            continue
        transformers.append((col, col_type, strategy, imputation, fill_value))

    lines += ["", "# Build preprocessing pipeline", "from sklearn.compose import make_column_selector", "_transformers = []"]
    for col, col_type, strategy, imputation, fill_value in transformers:
        imputer_code = "None"
        if imputation == "mean":
            imputer_code = "SimpleImputer(strategy='mean')"
        elif imputation == "median":
            imputer_code = "SimpleImputer(strategy='median')"
        elif imputation == "most_frequent":
            imputer_code = "SimpleImputer(strategy='most_frequent')"
        elif imputation == "constant":
            imputer_code = f"SimpleImputer(strategy='constant', fill_value={fill_value!r})"
        elif imputation == "knn":
            imputer_code = "KNNImputer()"

        if col_type == "numerical":
            scaler = {"standardize": "StandardScaler()", "normalize": "MinMaxScaler()", "robust": "RobustScaler()"}.get(strategy, "None")
            if imputer_code != "None":
                lines.append(f"_transformers.append(('num_{col}', Pipeline([('imp', {imputer_code}), ('sc', {scaler})]), {[col]!r}))")
            else:
                lines.append(f"_transformers.append(('num_{col}', {scaler}, {[col]!r}))")
        elif col_type == "categorical":
            enc = {"onehot": "OneHotEncoder(handle_unknown='ignore', sparse_output=False)", "label": "OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)", "ordinal": "OrdinalEncoder()"}.get(strategy, "None")
            if imputer_code != "None":
                lines.append(f"_transformers.append(('cat_{col}', Pipeline([('imp', {imputer_code}), ('enc', {enc})]), {[col]!r}))")
            else:
                lines.append(f"_transformers.append(('cat_{col}', {enc}, {[col]!r}))")

    # Feature selection
    fs = full_config.get("feature_selection", {})
    fs_method = fs.get("method", "none") if isinstance(fs, dict) else getattr(fs, "method", "none")
    if hasattr(fs_method, "value"):
        fs_method = fs_method.value

    lines += [
        "_ct = ColumnTransformer(_transformers, remainder='drop')",
        "_pipeline_steps = [('preprocessor', _ct)]",
    ]
    if fs_method == "select_k_best":
        fs_k = fs.get("k", 10) if isinstance(fs, dict) else getattr(fs, "k", 10)
        fs_func = fs.get("score_func", "f_classif") if isinstance(fs, dict) else getattr(fs, "score_func", "f_classif")
        if hasattr(fs_func, "value"): fs_func = fs_func.value
        lines.append(f"_pipeline_steps.append(('selector', SelectKBest({fs_func}, k={fs_k})))")
    elif fs_method == "variance_threshold":
        fs_thresh = fs.get("variance_threshold", 0.0) if isinstance(fs, dict) else getattr(fs, "variance_threshold", 0.0)
        lines.append(f"_pipeline_steps.append(('selector', VarianceThreshold(threshold={fs_thresh})))")

    lines += [
        "preprocessor = Pipeline(_pipeline_steps)",
        "X_train_proc = preprocessor.fit_transform(X_train, y_train)",
        "X_test_proc  = preprocessor.transform(X_test)",
        "joblib.dump(preprocessor, 'preprocessor.joblib')",
        "print('Preprocessor fitted, shape:', X_train_proc.shape)",
    ]
    return _nb_code(*lines)


def _supervised_train_cell(models_config: list[dict], tuning_config: dict, problem_type: str, experiment_id: str = "EXPERIMENT_ID") -> nbformat.NotebookNode:
    lines = [
        "import time, json, joblib",
        "from sklearn.linear_model import LogisticRegression, Ridge, Lasso",
        "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor",
        "from sklearn.svm import SVC, SVR",
        "from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor",
        "from sklearn.model_selection import cross_val_score",
        "",
        "_results = []",
        "_t0 = time.time()",
    ]
    for model_cfg in models_config:
        mname = model_cfg.get("name", "")
        params = model_cfg.get("parameters", {})
        lines += [
            f"\n# --- {mname} ---",
            f"model = {mname}(**{params!r})",
        ]
        if mname in tuning_config:
            snippet = build_tuning_code_snippet(mname, "model", tuning_config[mname])
            lines.append(snippet)
        else:
            lines.append("best_params = {}")

        if problem_type == "classification":
            lines += [
                "model.fit(X_train_proc, y_train)",
                "from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix",
                "_cv = cross_val_score(model, X_train_proc, y_train, cv=5, scoring='accuracy')",
                "_preds = model.predict(X_test_proc)",
                "_metrics = {'accuracy': accuracy_score(y_test, _preds), 'f1_macro': f1_score(y_test, _preds, average='macro', zero_division=0)}",
                "try:",
                "    _proba = model.predict_proba(X_test_proc)",
                "    _classes = list(model.classes_)",
                "    if len(_classes)==2:",
                "        _metrics['roc_auc'] = roc_auc_score(y_test, _proba[:,1])",
                "except: pass",
                "_cm = confusion_matrix(y_test, _preds).tolist()",
                "_fi = dict(zip(range(X_train_proc.shape[1]), model.feature_importances_.tolist())) if hasattr(model,'feature_importances_') else {}",
            ]
        else:  # regression
            lines += [
                "model.fit(X_train_proc, y_train)",
                "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score",
                "_cv = cross_val_score(model, X_train_proc, y_train, cv=5, scoring='r2')",
                "_preds = model.predict(X_test_proc)",
                "_metrics = {'mse': mean_squared_error(y_test,_preds), 'mae': mean_absolute_error(y_test,_preds), 'r2': r2_score(y_test,_preds)}",
                "_cm = []",
                "_fi = {}",
            ]
        lines += [
            f"_results.append({{'name':{mname!r},'parameters':{params!r},'best_params':best_params,",
            "    'metrics':_metrics,'cv_scores':{'mean':float(_cv.mean()),'std':float(_cv.std()),'folds':_cv.tolist()},",
            "    'confusion_matrix':_cm,'feature_importances':_fi})",
            # BUG FIX: use a plain string literal filename, not an f-string referencing the class object
            f"joblib.dump(model, '{mname}_model.joblib')",
            f"print('Done: {mname}', _metrics)",
        ]
    lines += [
        "\n_duration = time.time() - _t0",
        "import datetime",
        # BUG FIX: embed the real experiment_id, not the literal string 'EXPERIMENT_ID'
        f"_out = {{'experiment_id': {experiment_id!r}, 'problem_type': {problem_type!r},",
        "        'models': _results, 'training_duration_seconds': _duration,",
        "        'timestamp': datetime.datetime.utcnow().isoformat()}",
        "with open('results.json','w') as f: json.dump(_out, f, indent=2)",
        "# Save best model",
        "_best = max(_results, key=lambda r: list(r['metrics'].values())[0])",
        "import shutil",
        "shutil.copy(f\"{_best['name']}_model.joblib\", 'best_model.joblib')",
        "print('Results saved to results.json')",
    ]
    return _nb_code(*lines)


def generate_supervised_notebook(
    experiment_id: str,
    dataset_path: Path,
    full_config: dict,
    models_config: list[dict],
    tuning_config_json: str | None,
    problem_type: str,
    output_path: Path,
) -> Path:
    tuning_config = validate_tuning_config(tuning_config_json, problem_type)
    target_column = None
    columns_cfg = full_config.get("columns", {})
    for col, cfg in columns_cfg.items():
        is_tgt = cfg.get("is_target", False) if isinstance(cfg, dict) else getattr(cfg, "is_target", False)
        if is_tgt:
            target_column = col
            break

    nb = new_notebook()
    nb.cells = [
        _nb_md(f"# ML Pipeline — {problem_type.capitalize()} Notebook\n*Experiment: {experiment_id}*"),
        _supervised_install_cell(),
        _supervised_load_cell(dataset_path, dataset_path.suffix.lower()),
        _supervised_preprocessing_cell(full_config, target_column or "target", problem_type),
        # Pass experiment_id so the notebook writes the real ID into results.json
        _supervised_train_cell(models_config, tuning_config, problem_type, experiment_id=experiment_id),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    return output_path


# ---------------------------------------------------------------------------
# Clustering notebook
# ---------------------------------------------------------------------------

def _clustering_preprocess_cell(full_config: dict) -> nbformat.NotebookNode:
    columns_cfg = full_config.get("columns", {})
    transformers = []
    for col, cfg in columns_cfg.items():
        if isinstance(cfg, dict):
            col_type = cfg.get("type", "numerical")
            strategy = cfg.get("strategy", "none")
            is_target = cfg.get("is_target", False)
        else:
            col_type = getattr(cfg.type, "value", cfg.type) if hasattr(cfg.type, "value") else str(cfg.type)
            strategy = getattr(cfg.strategy, "value", cfg.strategy) if hasattr(cfg.strategy, "value") else str(cfg.strategy)
            is_target = getattr(cfg, "is_target", False)
        if is_target or strategy == "none":
            continue
        transformers.append((col, col_type, strategy))

    lines = [
        "# Clustering: no train/test split — all data used for fitting (SOP §5)",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder",
        "from sklearn.impute import SimpleImputer, KNNImputer",
        "import joblib",
        "",
        "X = df.copy()",
        "_transformers = []",
    ]
    for col, col_type, strategy in transformers:
        if col_type == "numerical":
            sc = {"standardize": "StandardScaler()", "normalize": "MinMaxScaler()", "robust": "RobustScaler()"}.get(strategy)
            if sc:
                lines.append(f"_transformers.append(('num_{col}', {sc}, {[col]!r}))")
        elif col_type == "categorical":
            enc = {"onehot": "OneHotEncoder(handle_unknown='ignore',sparse_output=False)", "label": "OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)", "ordinal": "OrdinalEncoder()"}.get(strategy)
            if enc:
                lines.append(f"_transformers.append(('cat_{col}', {enc}, {[col]!r}))")
    lines += [
        "_ct = ColumnTransformer(_transformers, remainder='drop')",
        "preprocessor = Pipeline([('preprocessor', _ct)])",
        "X_proc = preprocessor.fit_transform(X)",
        "joblib.dump(preprocessor, 'preprocessor.joblib')",
        "print('Preprocessing done, shape:', X_proc.shape)",
    ]
    return _nb_code(*lines)


def _clustering_pca_cell() -> nbformat.NotebookNode:
    return _nb_code(
        "# PCA projection — always computed for 2D scatter visualisation (SOP §5)",
        "from sklearn.decomposition import PCA",
        "_pca2 = PCA(n_components=2, random_state=42)",
        "_pca_proj = _pca2.fit_transform(X_proc)",
        "pca_x = _pca_proj[:, 0].tolist()",
        "pca_y = _pca_proj[:, 1].tolist()",
        "print('PCA projection computed')",
    )


def _clustering_train_cell(models_config: list[dict], experiment_id: str = "EXPERIMENT_ID") -> nbformat.NotebookNode:
    lines = [
        "import time, json, joblib, numpy as np",
        "from sklearn.cluster import KMeans, DBSCAN",
        "from sklearn.decomposition import PCA",
        "from sklearn.metrics import silhouette_score, davies_bouldin_score",
        "",
        "_results = []",
        "_t0 = time.time()",
    ]
    for model_cfg in models_config:
        mname = model_cfg.get("name", "")
        params = model_cfg.get("parameters", {})
        lines += [
            f"\n# --- {mname} ---",
            f"_params = {params!r}",
        ]
        if mname == "KMeans":
            n_clusters = params.get("n_clusters", 3)
            lines += [
                f"model = KMeans(n_clusters={n_clusters}, init='k-means++', random_state=42, n_init='auto')",
                "labels = model.fit_predict(X_proc)",
                "# Elbow curve",
                "_elbow_k, _elbow_inertia = [], []",
                "for _k in range(2, 11):",
                "    _km = KMeans(n_clusters=_k, random_state=42, n_init='auto').fit(X_proc)",
                "    _elbow_k.append(_k); _elbow_inertia.append(float(_km.inertia_))",
                "elbow_data = {'k': _elbow_k, 'inertia': _elbow_inertia}",
                "_inertia = float(model.inertia_)",
            ]
        elif mname == "DBSCAN":
            eps = params.get("eps", 0.5)
            min_samples = params.get("min_samples", 5)
            lines += [
                f"model = DBSCAN(eps={eps}, min_samples={min_samples}, metric='euclidean')",
                "labels = model.fit_predict(X_proc)",
                "elbow_data = None",
                "_inertia = None",
            ]
        elif mname == "PCA":
            n_components = params.get("n_components", 2)
            lines += [
                f"model = PCA(n_components={n_components}, random_state=42)",
                "model.fit(X_proc)",
                "labels = np.zeros(len(X_proc), dtype=int)",
                "elbow_data = None",
                "_inertia = None",
            ]
        else:
            lines += [
                f"model = None",
                "labels = np.zeros(len(X_proc), dtype=int)",
                "elbow_data = None",
                "_inertia = None",
            ]

        lines += [
            "# Clustering metrics",
            "_n_clusters_found = len(set(labels) - {-1})",
            "_noise_points = int(np.sum(labels == -1))",
            "_mask = labels != -1",
            "_sil, _db = None, None",
            "if _n_clusters_found >= 2:",
            "    _sil = float(silhouette_score(X_proc[_mask], labels[_mask]))",
            "    _db  = float(davies_bouldin_score(X_proc[_mask], labels[_mask]))",
            "_metrics = {'silhouette_score':_sil,'davies_bouldin_score':_db,",
            "            'n_clusters_found':_n_clusters_found,'noise_points':_noise_points,'inertia':_inertia}",
            # cluster label counts
            "_label_counts = {str(l): int(np.sum(labels==l)) for l in np.unique(labels)}",
            # pca projection coloured by label
            "_pca_section = {'x': pca_x, 'y': pca_y, 'labels': labels.tolist()}",
            f"_results.append({{'name':{mname!r},'parameters':_params,'metrics':_metrics,",
            "    'elbow_data':elbow_data,'cluster_label_counts':_label_counts,'pca_projection':_pca_section})",
            # BUG FIX: plain string literal, not f-string referencing the class object
            f"with open('cluster_labels_{mname}.json','w') as f: json.dump(labels.tolist(),f)",
            f"joblib.dump(model, '{mname}_model.joblib')",
            f"print('Done: {mname}', _metrics)",
        ]

    lines += [
        "\n_duration = time.time() - _t0",
        "import datetime",
        # BUG FIX: embed real experiment_id as a string literal, not a variable reference
        f"_out = {{'experiment_id': {experiment_id!r}, 'problem_type': 'clustering',",
        "        'models':_results,'training_duration_seconds':_duration,",
        "        'timestamp':datetime.datetime.utcnow().isoformat()}",
        "with open('results.json','w') as f: json.dump(_out,f,indent=2)",
        "print('Clustering results saved.')",
    ]
    return _nb_code(*lines)


def generate_clustering_notebook(
    experiment_id: str,
    dataset_path: Path,
    full_config: dict,
    models_config: list[dict],
    output_path: Path,
) -> Path:
    nb = new_notebook()
    nb.cells = [
        _nb_md(f"# ML Pipeline — Clustering Notebook\n*Experiment: {experiment_id}*"),
        _nb_code(
            "import subprocess, sys",
            "subprocess.check_call([sys.executable,'-m','pip','install','-q','scikit-learn','pandas','numpy','joblib'])",
        ),
        _supervised_load_cell(dataset_path, dataset_path.suffix.lower()),
        _clustering_preprocess_cell(full_config),
        _clustering_pca_cell(),
        _clustering_train_cell(models_config, experiment_id=experiment_id),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    return output_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_notebook(
    experiment_id: str,
    dataset_path: Path,
    full_config: dict,
    models_config: list[dict],
    problem_type: str,
    tuning_config_json: str | None,
    storage_dir: Path,
) -> Path:
    """
    Generate a notebook for the given experiment and return its path.
    Branches on problem_type: supervised vs clustering.
    """
    output_path = storage_dir / "notebooks" / f"{experiment_id}.ipynb"
    if problem_type in ("classification", "regression"):
        return generate_supervised_notebook(
            experiment_id, dataset_path, full_config,
            models_config, tuning_config_json, problem_type, output_path,
        )
    else:
        return generate_clustering_notebook(
            experiment_id, dataset_path, full_config,
            models_config, output_path,
        )
