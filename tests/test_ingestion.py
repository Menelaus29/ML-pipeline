import pytest
from pathlib import Path

from backend.services.ingestion import parse_upload

FIXTURES = Path(__file__).parent / "fixtures"


def test_row_count_correct():
    result = parse_upload(FIXTURES / "iris.csv", "iris.csv")
    assert result["row_count"] == 10


def test_numerical_column_inferred():
    result = parse_upload(FIXTURES / "iris.csv", "iris.csv")
    assert result["inferred_schema"]["sepal_length"] == "numerical"


def test_categorical_column_inferred():
    result = parse_upload(FIXTURES / "iris.csv", "iris.csv")
    assert result["inferred_schema"]["species"] == "categorical"


def test_null_rate_computed():
    result = parse_upload(FIXTURES / "titanic_small.csv", "titanic_small.csv")
    assert result["null_rates"]["age"] == pytest.approx(0.2)


def test_unsupported_file_type_raises(tmp_path):
    bad_file = tmp_path / "data.txt"
    bad_file.write_text("some content")
    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_upload(bad_file, "data.txt")


def test_malformed_csv_raises(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_bytes(b"\x00\x01\x02\x03\xff\xfe")
    with pytest.raises(ValueError, match="Failed to parse CSV file"):
        parse_upload(bad_csv, "bad.csv")