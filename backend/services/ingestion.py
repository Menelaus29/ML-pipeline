import uuid
import json
from pathlib import Path

import pandas as pd


def _infer_column_type(series: pd.Series) -> str:
    """
    Infer the semantic column type following SOP §5 rules (in order):
    1. numerical  — int or float dtype (excluding bool)
    2. categorical — object dtype AND nunique < 20  (checked FIRST for object)
    3. text        — object dtype AND mean whitespace-token count > 5  (checked SECOND)
    4. fallback    — object dtype matching neither rule above → text
    5. non-numeric, non-object (bool, datetime, …) → categorical

    Note: pandas treats bool as numeric, so we check it explicitly first.
    """
    # Rule 5 (checked before numeric): bool, datetime, period, timedelta → categorical
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "categorical"

    if pd.api.types.is_numeric_dtype(series):
        return "numerical"

    if pd.api.types.is_object_dtype(series):
        # Rule 2: low-cardinality object → categorical
        if series.nunique() < 20:
            return "categorical"
        # Rule 3: long free-text → text
        mean_tokens = series.dropna().map(lambda x: len(str(x).split())).mean()
        if mean_tokens > 5:
            return "text"
        # Rule 4: fallback for object dtype matching neither rule
        return "text"

    # Rule 5: bool, datetime, timedelta, period, … → categorical
    return "categorical"


def _load_dataframe(file_path: Path) -> pd.DataFrame:
    """Load a CSV or JSON file into a DataFrame."""
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to parse CSV file '{file_path.name}': {e}")
    if suffix == ".json":
        try:
            return pd.read_json(file_path)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON file '{file_path.name}': {e}")
    raise ValueError(
        f"Unsupported file type '{suffix}'. Only .csv and .json are accepted."
    )


def parse_upload(file_path: Path, filename: str) -> dict:
    """
    Parse an uploaded dataset file and return profile information.

    Returns:
        dict with keys: row_count, column_count, inferred_schema,
                        null_rates, descriptive_stats
    """
    df = _load_dataframe(file_path)

    row_count = len(df)
    column_count = len(df.columns)
    inferred_schema = {col: _infer_column_type(df[col]) for col in df.columns}
    null_rates = df.isnull().mean().round(4).to_dict()

    raw_stats = df.describe(include="all")
    descriptive_stats = raw_stats.where(pd.notnull(raw_stats), other=None).to_dict()

    return {
        "row_count": row_count,
        "column_count": column_count,
        "inferred_schema": inferred_schema,
        "null_rates": null_rates,
        "descriptive_stats": descriptive_stats,
    }


def save_upload(upload: bytes, filename: str, storage_dir: Path) -> Path:
    """Save raw upload bytes to storage/datasets/ with a UUID prefix; return the path."""
    storage_dir.mkdir(parents=True, exist_ok=True)
    unique_name = f"{uuid.uuid4()}_{filename}"
    dest_path = storage_dir / unique_name
    dest_path.write_bytes(upload)
    return dest_path