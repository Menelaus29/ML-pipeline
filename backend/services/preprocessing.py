import json
from collections import defaultdict

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OrdinalEncoder,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from backend.core.schemas import ColumnConfig, ColumnType


# Maps strategy name to its instantiated sklearn scaler
_NUMERICAL_SCALERS = {
    "standardize": lambda: StandardScaler(),
    "normalize":   lambda: MinMaxScaler(),
    "robust":      lambda: RobustScaler(),
}

# Maps strategy name to its instantiated sklearn encoder
_CATEGORICAL_ENCODERS = {
    "onehot":  lambda: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "label":   lambda: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    "ordinal": lambda: OrdinalEncoder(),
}

# Maps strategy name to its instantiated sklearn text vectorizer
_TEXT_VECTORIZERS = {
    "tfidf": lambda: TfidfVectorizer(max_features=500),
    "count": lambda: CountVectorizer(),
}


def build_pipeline(config: dict[str, ColumnConfig]) -> ColumnTransformer:
    # Build a fitted-ready ColumnTransformer from a validated column config dict
    transformers = []

    # Group numerical columns by strategy so one transformer handles all columns
    # sharing the same scaler — avoids redundant transformer entries
    numerical_by_strategy: dict[str, list[str]] = defaultdict(list)
    categorical_by_strategy: dict[str, list[str]] = defaultdict(list)
    numerical_passthrough: list[str] = []
    categorical_passthrough: list[str] = []

    for col_name, col_cfg in config.items():
        if col_cfg.is_target:
            continue  # target column never enters the feature transformer

        if col_cfg.type == ColumnType.numerical:
            if col_cfg.strategy == "none":
                numerical_passthrough.append(col_name)
            else:
                numerical_by_strategy[col_cfg.strategy].append(col_name)

        elif col_cfg.type == ColumnType.categorical:
            if col_cfg.strategy == "none":
                categorical_passthrough.append(col_name)
            else:
                categorical_by_strategy[col_cfg.strategy].append(col_name)

        elif col_cfg.type == ColumnType.text:
            if col_cfg.strategy == "none":
                # Text columns with strategy=none are dropped via remainder='drop'
                continue
            vectorizer = _TEXT_VECTORIZERS[col_cfg.strategy]()
            # Each text column gets its own vectorizer — they produce variable-width output
            transformers.append((f"text_{col_cfg.strategy}_{col_name}", vectorizer, col_name))

    # One transformer entry per numerical strategy group
    for strategy, cols in numerical_by_strategy.items():
        scaler = _NUMERICAL_SCALERS[strategy]()
        transformers.append((f"num_{strategy}", scaler, cols))

    # Passthrough for numerical columns with strategy=none
    if numerical_passthrough:
        transformers.append(("num_passthrough", "passthrough", numerical_passthrough))

    # One transformer entry per categorical strategy group
    for strategy, cols in categorical_by_strategy.items():
        encoder = _CATEGORICAL_ENCODERS[strategy]()
        transformers.append((f"cat_{strategy}", encoder, cols))

    # Passthrough for categorical columns with strategy=none
    if categorical_passthrough:
        transformers.append(("cat_passthrough", "passthrough", categorical_passthrough))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def serialize_pipeline_config(config: dict[str, ColumnConfig]) -> str:
    return json.dumps({col: cfg.model_dump() for col, cfg in config.items()})


def deserialize_pipeline_config(json_str: str) -> dict[str, ColumnConfig]:
    raw = json.loads(json_str)
    return {col: ColumnConfig(**cfg) for col, cfg in raw.items()}