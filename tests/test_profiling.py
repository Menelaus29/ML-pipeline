import logging
from pathlib import Path

from backend.services.profiling import generate_profile

FIXTURES = Path(__file__).parent / "fixtures"


def test_generate_profile_creates_file(tmp_path):
    # Intentionally slow — ydata-profiling runs a full report
    output = generate_profile("test123", FIXTURES / "iris.csv", "Iris Test", tmp_path)
    assert output is not None
    assert output == tmp_path / "test123.html"
    assert output.exists()


def test_generated_file_is_html(tmp_path):
    output = generate_profile("test456", FIXTURES / "iris.csv", "Iris HTML Test", tmp_path)
    assert output is not None
    content = output.read_text(encoding="utf-8")
    assert "<html" in content


def test_bad_path_returns_none_and_logs_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="backend.services.profiling"):
        result = generate_profile(
            "bad_id",
            Path("/nonexistent/path/data.csv"),
            "Bad Dataset",
            tmp_path,
        )
    assert result is None
    assert any("Profiling failed" in record.message for record in caplog.records)