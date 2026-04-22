"""
EDA service — SOP §6 (Analysis Agent input preparation).

Provides compute_eda_features() which the Analysis Agent calls to produce
a structured dict of data-quality findings.  The agent then passes this
dict to the LLM for narrative generation.

Returns a dict that is safe to serialise to JSON (no numpy types).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _safe_float(val) -> Optional[float]:
    """Convert numpy scalar to Python float, or None if not finite."""
    try:
        v = float(val)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def compute_eda_features(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    problem_type: Optional[str] = None,
) -> dict:
    """
    Compute structured EDA findings for the Analysis Agent.

    Parameters
    ----------
    df            : Loaded dataset DataFrame.
    target_column : Name of target column (None for clustering).
    problem_type  : 'classification' | 'regression' | 'clustering' | None.

    Returns
    -------
    dict with keys:
      - row_count, column_count
      - null_severity   : {col: null_rate}   (only cols with null_rate > 0)
      - high_cardinality: [col, ...]          (object cols with nunique >= 20)
      - class_distribution: {label: count}   (classification only)
      - correlation_matrix: {col: {col: r}}  (numerical cols only)
      - outlier_flags   : {col: {iqr_outlier_count, pct}}
      - dtypes          : {col: dtype_str}
    """
    findings: dict = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "problem_type": problem_type,
        "target_column": target_column,
    }

    # --- dtypes ---
    findings["dtypes"] = {col: str(df[col].dtype) for col in df.columns}

    # --- null severity (cols with any nulls) ---
    null_rates = df.isnull().mean()
    findings["null_severity"] = {
        col: round(float(rate), 4)
        for col, rate in null_rates.items()
        if rate > 0
    }

    # --- high cardinality (object columns with nunique >= 20) ---
    findings["high_cardinality"] = [
        col
        for col in df.select_dtypes(include="object").columns
        if df[col].nunique() >= 20
    ]

    # --- class distribution (classification only) ---
    if problem_type == "classification" and target_column and target_column in df.columns:
        vc = df[target_column].value_counts()
        findings["class_distribution"] = {str(k): int(v) for k, v in vc.items()}
    else:
        findings["class_distribution"] = None  # skipped for clustering / regression

    # --- correlation matrix (numerical columns only) ---
    num_df = df.select_dtypes(include="number")
    if len(num_df.columns) >= 2:
        corr = num_df.corr()
        findings["correlation_matrix"] = {
            col: {
                other: _safe_float(val)
                for other, val in row.items()
            }
            for col, row in corr.to_dict().items()
        }
    else:
        findings["correlation_matrix"] = {}

    # --- IQR-based outlier detection (numerical columns) ---
    outlier_flags: dict = {}
    for col in num_df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((series < lower) | (series > upper)).sum())
        if outlier_count > 0:
            outlier_flags[col] = {
                "iqr_outlier_count": outlier_count,
                "pct": round(outlier_count / len(series) * 100, 2),
            }
    findings["outlier_flags"] = outlier_flags

    return findings


def load_and_compute_eda(
    file_path: Path,
    target_column: Optional[str] = None,
    problem_type: Optional[str] = None,
) -> dict:
    """
    Convenience wrapper: load a CSV/JSON file and compute EDA features.
    Used by the Analysis Agent when it has a file path.
    """
    from backend.services.ingestion import _load_dataframe

    df = _load_dataframe(file_path)
    return compute_eda_features(df, target_column=target_column, problem_type=problem_type)
