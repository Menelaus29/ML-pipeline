"""
Preprocessing service — SOP §5 full implementation.

Implements the four-section config:
  1. columns          — per-column type / strategy / imputation
  2. outlier_treatment — winsorise | iqr_remove | zscore_remove | none
  3. feature_selection — select_k_best | variance_threshold | none
  4. class_balancing  — smote | oversample | undersample | class_weight | none
                        (ignored for regression/clustering; handled in notebook)
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OrdinalEncoder,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from backend.core.schemas import (
    ClassBalancingMethod,
    ColumnConfig,
    ColumnType,
    FeatureSelectionMethod,
    FullPreprocessingConfig,
    ImputationStrategy,
    OutlierMethod,
)


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

_NUMERICAL_SCALERS = {
    "standardize": lambda: StandardScaler(),
    "normalize":   lambda: MinMaxScaler(),
    "robust":      lambda: RobustScaler(),
}

_CATEGORICAL_ENCODERS = {
    "onehot":  lambda: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "label":   lambda: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    "ordinal": lambda: OrdinalEncoder(),
}

_TEXT_VECTORIZERS = {
    "tfidf": lambda: TfidfVectorizer(max_features=500),
    "count": lambda: CountVectorizer(),
}

_SCORE_FUNCS = {
    "f_classif":          f_classif,
    "f_regression":       f_regression,
    "mutual_info_classif": mutual_info_classif,
}


# ---------------------------------------------------------------------------
# WinsoriseTransformer — custom sklearn transformer (SOP §5)
# Caps values at Q1 − threshold×IQR and Q3 + threshold×IQR.
# Persisted inside the saved preprocessor joblib.
# ---------------------------------------------------------------------------

class WinsoriseTransformer(BaseEstimator, TransformerMixin):
    """
    Winsorise numerical columns in-place.
    Bounds are computed on fit() so the same bounds are applied at transform()
    — preventing data leakage when applied to a test split.
    """

    def __init__(self, threshold: float = 1.5):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None) -> "WinsoriseTransformer":
        self._bounds: dict[str, tuple[float, float]] = {}
        for col in X.select_dtypes(include="number").columns:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self._bounds[col] = (
                q1 - self.threshold * iqr,
                q3 + self.threshold * iqr,
            )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        for col, (lower, upper) in self._bounds.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lower, upper=upper)
        return X


# ---------------------------------------------------------------------------
# Outlier treatment (applied to raw DataFrame BEFORE ColumnTransformer)
# SOP §5: only numerical columns are affected.
# ---------------------------------------------------------------------------

def apply_outlier_treatment(
    df: pd.DataFrame,
    config: dict,  # outlier_treatment section dict (or OutlierTreatmentConfig)
) -> pd.DataFrame:
    """
    Apply outlier treatment to a DataFrame in-place equivalent.

    Parameters
    ----------
    df     : DataFrame to treat (all splits share the same function).
    config : dict with keys ``method`` and ``threshold``, or an
             OutlierTreatmentConfig pydantic model.

    Returns
    -------
    Treated DataFrame (rows may be removed for iqr_remove / zscore_remove).

    Notes
    -----
    For iqr_remove / zscore_remove the bounds should be computed on the
    TRAINING split only.  The caller is responsible for passing the correct
    df.  For winsorise, use WinsoriseTransformer (fit on train, transform
    both splits) — this function is provided as a convenience for notebook
    code that needs a standalone helper.
    """
    # Accept both dict and pydantic model
    if hasattr(config, "method"):
        method = config.method.value if hasattr(config.method, "value") else config.method
        threshold = config.threshold
    else:
        method = config.get("method", "none")
        threshold = config.get("threshold", 1.5)

    if method == "none":
        return df

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return df

    if method == "winsorise":
        for col in num_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            df = df.copy()
            df[col] = df[col].clip(lower=lower, upper=upper)
        return df

    if method == "iqr_remove":
        mask = pd.Series([True] * len(df), index=df.index)
        for col in num_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            mask &= (df[col] >= lower) & (df[col] <= upper)
        return df[mask].reset_index(drop=True)

    if method == "zscore_remove":
        mask = pd.Series([True] * len(df), index=df.index)
        for col in num_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std == 0:
                continue
            z = (df[col] - mean) / std
            mask &= z.abs() <= threshold
        return df[mask].reset_index(drop=True)

    raise ValueError(f"Unknown outlier method: {method!r}")


# ---------------------------------------------------------------------------
# Imputer builder helpers
# ---------------------------------------------------------------------------

def _build_simple_imputer(strategy: str, fill_value=None) -> SimpleImputer:
    if strategy == "constant":
        return SimpleImputer(strategy="constant", fill_value=fill_value)
    return SimpleImputer(strategy=strategy)


def _build_column_pipeline(imputer, transformer) -> Pipeline:
    """Chain imputer → transformer inside a sklearn Pipeline step."""
    steps = []
    if imputer is not None:
        steps.append(("imputer", imputer))
    steps.append(("transformer", transformer))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Main pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(
    config: dict[str, ColumnConfig],
    feature_selection_config=None,
) -> Pipeline:
    """
    Build a full sklearn Pipeline from a validated column config dict.

    Structure (SOP §5):
      Pipeline([
        ("preprocessor", ColumnTransformer(...)),   # per-column scaling / encoding
        ("selector",     SelectKBest | VarianceThreshold),  # optional final step
      ])

    KNN imputation is handled as a separate ColumnTransformer step on all
    numerical columns collectively, before per-column transformers run.

    Parameters
    ----------
    config                 : dict[col_name, ColumnConfig]
    feature_selection_config : FeatureSelectionConfig | None
    """
    transformers = []

    # Collect KNN-imputed numerical columns separately (applied to all at once)
    knn_numerical_cols: list[str] = []

    # Group numerical columns by (imputation, strategy)
    numerical_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    numerical_passthrough: list[str] = []
    numerical_passthrough_knn: list[str] = []  # knn imputed, strategy=none

    categorical_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    categorical_passthrough: list[str] = []

    for col_name, col_cfg in config.items():
        if col_cfg.is_target:
            continue  # target column never enters the feature transformer

        imputation = col_cfg.imputation.value if hasattr(col_cfg.imputation, "value") else col_cfg.imputation
        strategy = col_cfg.strategy

        if col_cfg.type == ColumnType.numerical:
            if imputation == "knn":
                knn_numerical_cols.append(col_name)
                if strategy == "none":
                    numerical_passthrough_knn.append(col_name)
                else:
                    numerical_by_key[(imputation, strategy)].append(col_name)
            elif strategy == "none" and imputation == "none":
                numerical_passthrough.append(col_name)
            else:
                numerical_by_key[(imputation, strategy)].append(col_name)

        elif col_cfg.type == ColumnType.categorical:
            if strategy == "none" and imputation == "none":
                categorical_passthrough.append(col_name)
            else:
                categorical_by_key[(imputation, strategy)].append(col_name)

        elif col_cfg.type == ColumnType.text:
            if strategy == "none":
                # Text columns with strategy=none are dropped via remainder='drop'
                continue
            vectorizer = _TEXT_VECTORIZERS[strategy]()
            # Each text column gets its own vectorizer
            transformers.append((f"text_{strategy}_{col_name}", vectorizer, col_name))

    # KNN imputer as a joint step on all KNN-numerical columns
    if knn_numerical_cols:
        transformers.append((
            "num_knn_imputer",
            _build_column_pipeline(KNNImputer(), "passthrough"),
            knn_numerical_cols,
        ))

    # Per-(imputation, strategy) group for numerical columns
    for (imputation, strategy), cols in numerical_by_key.items():
        col_cfg_sample = next(
            cfg for col, cfg in config.items()
            if col in cols
        )
        imputer = None
        if imputation not in ("none", "knn"):
            imputer = _build_simple_imputer(
                imputation,
                fill_value=col_cfg_sample.imputation_fill_value,
            )
        elif imputation == "knn":
            # Already handled above; just apply scaler
            pass

        if strategy != "none":
            transformer = _NUMERICAL_SCALERS[strategy]()
            if imputer is not None:
                step = _build_column_pipeline(imputer, transformer)
            else:
                step = transformer
            transformers.append((f"num_{imputation}_{strategy}", step, cols))

    # Passthrough for numerical columns with no imputation and no scaling
    if numerical_passthrough:
        transformers.append(("num_passthrough", "passthrough", numerical_passthrough))

    # Passthrough for KNN-imputed numericals with strategy=none (imputation only)
    if numerical_passthrough_knn:
        transformers.append(("num_knn_passthrough", KNNImputer(), numerical_passthrough_knn))

    # Per-(imputation, strategy) group for categorical columns
    for (imputation, strategy), cols in categorical_by_key.items():
        col_cfg_sample = next(
            cfg for col, cfg in config.items()
            if col in cols
        )
        imputer = None
        if imputation not in ("none", "knn"):
            imputer = _build_simple_imputer(
                imputation,
                fill_value=col_cfg_sample.imputation_fill_value,
            )

        if strategy != "none":
            encoder = _CATEGORICAL_ENCODERS[strategy]()
            if imputer is not None:
                step = _build_column_pipeline(imputer, encoder)
            else:
                step = encoder
            transformers.append((f"cat_{imputation}_{strategy}", step, cols))
        elif imputer is not None:
            # Impute only, no encoding (strategy=none with imputation)
            transformers.append((f"cat_{imputation}_passthrough", imputer, cols))

    # Passthrough for categorical columns with no imputation and no encoding
    if categorical_passthrough:
        transformers.append(("cat_passthrough", "passthrough", categorical_passthrough))

    column_transformer = ColumnTransformer(transformers=transformers, remainder="drop")

    # Build top-level pipeline: preprocessor → optional selector
    pipeline_steps: list[tuple] = [("preprocessor", column_transformer)]

    if feature_selection_config is not None:
        fs_method = (
            feature_selection_config.method.value
            if hasattr(feature_selection_config.method, "value")
            else feature_selection_config.method
        )
        if fs_method == "select_k_best":
            score_func_name = (
                feature_selection_config.score_func.value
                if hasattr(feature_selection_config.score_func, "value")
                else feature_selection_config.score_func
            )
            score_func = _SCORE_FUNCS[score_func_name]
            selector = SelectKBest(score_func=score_func, k=feature_selection_config.k)
            pipeline_steps.append(("selector", selector))
        elif fs_method == "variance_threshold":
            selector = VarianceThreshold(
                threshold=feature_selection_config.variance_threshold
            )
            pipeline_steps.append(("selector", selector))
        # else: "none" → no selector step

    return Pipeline(pipeline_steps)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def serialize_full_config(config: FullPreprocessingConfig) -> str:
    """Serialise the full four-section config to a JSON string for DB storage."""
    return config.model_dump_json()


def deserialize_full_config(json_str: str) -> FullPreprocessingConfig:
    """Deserialise a JSON string from DB back to a FullPreprocessingConfig."""
    return FullPreprocessingConfig.model_validate_json(json_str)


# Legacy helpers kept for backward compatibility with existing tests
def serialize_pipeline_config(config: dict[str, ColumnConfig]) -> str:
    """Serialise a columns-only config dict to JSON (legacy / test helper)."""
    return json.dumps({col: cfg.model_dump() for col, cfg in config.items()})


def deserialize_pipeline_config(json_str: str) -> dict[str, ColumnConfig]:
    """Deserialise a columns-only JSON string (legacy / test helper)."""
    raw = json.loads(json_str)
    return {col: ColumnConfig(**cfg) for col, cfg in raw.items()}