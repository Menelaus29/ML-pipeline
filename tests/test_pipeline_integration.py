import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from backend.core.schemas import ColumnConfig
from backend.services.preprocessing import build_pipeline

FIXTURES = Path(__file__).parent / "fixtures"

# All four numerical cols standardized; species is target → excluded from transformer
_IRIS_CONFIG: dict[str, ColumnConfig] = {
    "sepal_length": ColumnConfig(type="numerical", strategy="standardize", is_target=False),
    "sepal_width":  ColumnConfig(type="numerical", strategy="standardize", is_target=False),
    "petal_length": ColumnConfig(type="numerical", strategy="standardize", is_target=False),
    "petal_width":  ColumnConfig(type="numerical", strategy="standardize", is_target=False),
    "species":      ColumnConfig(type="categorical", strategy="onehot",    is_target=True),
}


@pytest.fixture(scope="module")
def iris_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "iris.csv")


@pytest.fixture(scope="module")
def transformed_iris(iris_df: pd.DataFrame) -> np.ndarray:
    # Fit and transform iris using the standard config; reused across tests
    pipeline = build_pipeline(_IRIS_CONFIG)
    return pipeline.fit_transform(iris_df)


def test_transform_returns_numpy_array(transformed_iris: np.ndarray):
    assert isinstance(transformed_iris, np.ndarray)


def test_transform_output_shape(transformed_iris: np.ndarray):
    # 10 rows; 4 numerical cols scaled (species excluded via is_target=True)
    assert transformed_iris.shape == (10, 4)


def test_standard_scaler_produces_zero_mean(transformed_iris: np.ndarray):
    # StandardScaler guarantees zero mean on the data it was fitted on
    col_means = transformed_iris.mean(axis=0)
    for mean in col_means:
        assert mean == pytest.approx(0.0, abs=1e-10)