import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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


def _get_column_transformer(pipeline) -> ColumnTransformer:
    """
    build_pipeline() now returns a sklearn Pipeline wrapping a ColumnTransformer
    (to support the optional feature-selection step appended at the end).
    Extract the ColumnTransformer from the first Pipeline step.
    """
    if isinstance(pipeline, Pipeline):
        return pipeline.named_steps["preprocessor"]
    return pipeline  # backward compat


def test_numerical_standardize_produces_standard_scaler():
    config = _make_config(sepal_length=("numerical", "standardize", False))
    pipeline = build_pipeline(config)
    ct = _get_column_transformer(pipeline)
    transformer_map = {name: t for name, t, _ in ct.transformers}
    # Key format changed to include imputation prefix: "num_{imputation}_{strategy}"
    assert any("standardize" in name for name in transformer_map), (
        f"Expected a 'standardize' transformer, got: {list(transformer_map.keys())}"
    )
    matching = [t for name, t in transformer_map.items() if "standardize" in name]
    assert isinstance(matching[0], StandardScaler)


def test_categorical_onehot_produces_onehot_encoder():
    config = _make_config(species=("categorical", "onehot", False))
    pipeline = build_pipeline(config)
    ct = _get_column_transformer(pipeline)
    transformer_map = {name: t for name, t, _ in ct.transformers}
    assert any("onehot" in name for name in transformer_map), (
        f"Expected an 'onehot' transformer, got: {list(transformer_map.keys())}"
    )
    matching = [t for name, t in transformer_map.items() if "onehot" in name]
    assert isinstance(matching[0], OneHotEncoder)


def test_mixed_config_produces_two_transformers():
    # One numerical + one categorical → exactly two explicit transformers in the ColumnTransformer
    config = _make_config(
        sepal_length=("numerical", "standardize", False),
        species=("categorical", "onehot", False),
    )
    pipeline = build_pipeline(config)
    # build_pipeline now returns a Pipeline, not a bare ColumnTransformer
    assert isinstance(pipeline, Pipeline), "build_pipeline should return a sklearn Pipeline"
    ct = _get_column_transformer(pipeline)
    assert isinstance(ct, ColumnTransformer)
    assert len(ct.transformers) == 2


def test_target_column_excluded_from_transformers():
    # is_target=True must cause the column to be skipped entirely
    config = _make_config(
        sepal_length=("numerical", "standardize", False),
        species=("categorical", "onehot", True),
    )
    pipeline = build_pipeline(config)
    ct = _get_column_transformer(pipeline)
    all_cols = [cols for _, _, cols in ct.transformers]
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