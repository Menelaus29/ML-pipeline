import logging
from pathlib import Path

from ydata_profiling import ProfileReport

logger = logging.getLogger(__name__)

def generate_profile(
    dataset_id: str,
    file_path: Path,
    dataset_name: str,
    storage_dir: Path,
) -> Path | None:
    # Generate a ydata-profiling HTML report for a dataset + save it to disk
    try:
        from backend.services.ingestion import _load_dataframe

        df = _load_dataframe(file_path)
        storage_dir.mkdir(parents=True, exist_ok=True)
        output_path = storage_dir / f"{dataset_id}.html"

        profile = ProfileReport(df, title=dataset_name, explorative=True, minimal=False)
        profile.to_file(output_path)

        return output_path
        
    except Exception as e:
        logger.warning(f"Profiling failed for dataset '{dataset_id}': {e}")
        return None