import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.core.schemas import ColumnConfig
from backend.services.preprocessing import (
    build_pipeline,
    deserialize_pipeline_config,
    serialize_pipeline_config,
)


def _make_config(**kwargs: dict) -> dict[str, ColumnConfig]:
    # Build a ColumnConfig dict from keyword args of {col: (type, strategy, is_target)}
    return {
        col: ColumnConfig(type=t, strategy=s, is_target=it)
        for col, (t, s, it) in kwargs.items()
    }


def test_numerical_standardize_produces_standard_scaler():
    config = _make_config(sepal_length=("numerical", "standardize", False))
    pipeline = build_pipeline(config)
    transformer_map = {name: t for name, t, _ in pipeline.transformers}
    assert "num_standardize" in transformer_map
    assert isinstance(transformer_map["num_standardize"], StandardScaler)


def test_categorical_onehot_produces_onehot_encoder():
    config = _make_config(species=("categorical", "onehot", False))
    pipeline = build_pipeline(config)
    transformer_map = {name: t for name, t, _ in pipeline.transformers}
    assert "cat_onehot" in transformer_map
    assert isinstance(transformer_map["cat_onehot"], OneHotEncoder)


def test_mixed_config_produces_two_transformers():
    # One numerical + one categorical → exactly two explicit transformers
    # remainder='drop' is not counted in pipeline.transformers
    config = _make_config(
        sepal_length=("numerical", "standardize", False),
        species=("categorical", "onehot", False),
    )
    pipeline = build_pipeline(config)
    assert isinstance(pipeline, ColumnTransformer)
    assert len(pipeline.transformers) == 2


def test_target_column_excluded_from_transformers():
    # is_target=True must cause the column to be skipped entirely
    config = _make_config(
        sepal_length=("numerical", "standardize", False),
        species=("categorical", "onehot", True),
    )
    pipeline = build_pipeline(config)
    all_cols = [cols for _, _, cols in pipeline.transformers]
    assert "species" not in str(all_cols)


def test_serialize_deserialize_round_trip():
    config = {
        "sepal_length": ColumnConfig(type="numerical", strategy="standardize", is_target=False),
        "petal_width":  ColumnConfig(type="numerical", strategy="robust",      is_target=False),
        "species":      ColumnConfig(type="categorical", strategy="onehot",    is_target=True),
    }
    recovered = deserialize_pipeline_config(serialize_pipeline_config(config))

    assert set(recovered.keys()) == set(config.keys())
    for col, original in config.items():
        assert recovered[col].type      == original.type
        assert recovered[col].strategy  == original.strategy
        assert recovered[col].is_target == original.is_target