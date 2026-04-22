"""
Experiments API — SOP §8.

POST   /api/experiments/                    create + generate notebook
GET    /api/experiments/                    list all (with optional dataset_id filter)
GET    /api/experiments/{id}                get single
PATCH  /api/experiments/{id}/status        update status (training / completed)
POST   /api/experiments/{id}/results       upload results.json → parse + create versions
GET    /api/experiments/{id}/notebook      download generated notebook (.ipynb)
"""
import json
import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.core.database import get_db
from backend.core.models import Dataset, Experiment, ExperimentStatus, ModelVersion, PreprocessingConfig
from backend.core.schemas import ExperimentCreate, ExperimentRead
from backend.services.notebook_gen import generate_notebook
from backend.services.preprocessing import deserialize_full_config
from backend.services.results_parser import extract_best_model, parse_results
from backend.services.versioning import create_version
from backend.core.utils import utcnow

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/experiments", tags=["experiments"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _generate_notebook_task(experiment_id: str) -> None:
    """Background task: generate notebook and update experiment.notebook_path."""
    from backend.core.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        experiment = await db.get(Experiment, experiment_id)
        if experiment is None:
            logger.error("Notebook gen: experiment %s not found", experiment_id)
            return

        dataset = await db.get(Dataset, experiment.dataset_id)
        prep_config = await db.get(PreprocessingConfig, experiment.preprocessing_config_id)
        if dataset is None or prep_config is None:
            logger.error("Notebook gen: missing dataset or config for %s", experiment_id)
            return

        full_config = deserialize_full_config(prep_config.config_json)
        models_config = json.loads(experiment.models_config_json)
        problem_type = dataset.problem_type.value if dataset.problem_type else "classification"

        try:
            nb_path = generate_notebook(
                experiment_id=experiment_id,
                dataset_path=Path(dataset.filepath),
                full_config=full_config.model_dump(),
                models_config=models_config,
                problem_type=problem_type,
                tuning_config_json=experiment.tuning_config_json,
                storage_dir=Path(settings.storage_dir),
            )
            experiment.notebook_path = str(nb_path)
            experiment.status = ExperimentStatus.notebook_generated
            await db.commit()
            logger.info("Notebook generated for experiment %s → %s", experiment_id, nb_path)
        except Exception as exc:
            logger.exception("Notebook generation failed for experiment %s: %s", experiment_id, exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/", response_model=ExperimentRead, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    payload: ExperimentCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Create an experiment and trigger notebook generation as a background task."""
    dataset = await db.get(Dataset, payload.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset {payload.dataset_id} not found.")

    prep_config = await db.get(PreprocessingConfig, payload.preprocessing_config_id)
    if prep_config is None:
        raise HTTPException(
            status_code=404,
            detail=f"PreprocessingConfig {payload.preprocessing_config_id} not found.",
        )

    # Validate models_config_json is parseable JSON
    try:
        models_config = json.loads(payload.models_config_json)
        if not isinstance(models_config, list):
            raise ValueError("models_config_json must be a JSON array.")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid models_config_json: {e}")

    experiment = Experiment(
        dataset_id=payload.dataset_id,
        preprocessing_config_id=payload.preprocessing_config_id,
        models_config_json=payload.models_config_json,
        tuning_config_json=payload.tuning_config_json,
        status=ExperimentStatus.pending,
        created_at=utcnow(),
    )
    db.add(experiment)
    await db.flush()
    await db.refresh(experiment)

    # Trigger notebook generation
    background_tasks.add_task(_generate_notebook_task, experiment.id)
    logger.info("Created experiment %s, notebook gen queued", experiment.id)
    return experiment


@router.get("/", response_model=list[ExperimentRead])
async def list_experiments(
    dataset_id: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """List all experiments, optionally filtered by dataset_id."""
    q = select(Experiment).order_by(Experiment.created_at.desc())
    if dataset_id:
        q = q.where(Experiment.dataset_id == dataset_id)
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/{experiment_id}", response_model=ExperimentRead)
async def get_experiment(experiment_id: str, db: AsyncSession = Depends(get_db)):
    experiment = await db.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found.")
    return experiment


@router.patch("/{experiment_id}/status", response_model=ExperimentRead)
async def update_experiment_status(
    experiment_id: str,
    new_status: ExperimentStatus,
    db: AsyncSession = Depends(get_db),
):
    """Manually update experiment status (e.g. training → completed)."""
    experiment = await db.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found.")
    experiment.status = new_status
    if new_status == ExperimentStatus.completed:
        experiment.completed_at = utcnow()
    await db.commit()
    await db.refresh(experiment)
    return experiment


@router.post("/{experiment_id}/results", status_code=status.HTTP_201_CREATED)
async def upload_results(
    experiment_id: str,
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
):
    """
    Upload results.json from a completed notebook run.
    Parses the file, creates ModelVersion records, marks experiment completed.
    """
    experiment = await db.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found.")

    raw_bytes = await file.read()
    try:
        parsed = parse_results(raw_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    dataset = await db.get(Dataset, experiment.dataset_id)
    problem_type = dataset.problem_type.value if dataset and dataset.problem_type else "classification"

    created_versions = []
    storage_dir = Path(settings.storage_dir)
    artifacts_dir = storage_dir / "artifacts" / experiment_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for model_result in parsed["models"]:
        model_name = model_result["name"]
        metrics = model_result["metrics"]
        parameters = {**model_result.get("parameters", {}), **model_result.get("best_params", {})}
        cv_metrics = model_result.get("cv_scores") if problem_type != "clustering" else None
        cluster_labels_path = None

        # Save cluster labels if present
        if problem_type == "clustering" and model_result.get("pca_projection"):
            labels_path = artifacts_dir / f"cluster_labels_{model_name}.json"
            labels_path.write_text(
                json.dumps(model_result["pca_projection"].get("labels", [])),
                encoding="utf-8",
            )
            cluster_labels_path = str(labels_path)

        # Artifact path (placeholder — real artifact would be uploaded separately)
        artifact_path = str(artifacts_dir / f"{model_name}_model.joblib")

        mv = await create_version(
            db=db,
            experiment_id=experiment_id,
            model_name=model_name,
            artifact_path=artifact_path,
            parameters=parameters,
            metrics=metrics,
            cv_metrics=cv_metrics,
            cluster_labels_path=cluster_labels_path,
        )
        created_versions.append({"id": mv.id, "model_name": mv.model_name, "version_number": mv.version_number})

    # Mark experiment completed
    experiment.status = ExperimentStatus.completed
    experiment.completed_at = utcnow()
    await db.commit()

    best = extract_best_model(parsed)
    logger.info(
        "Results uploaded for experiment %s: %d model versions created, best=%s",
        experiment_id, len(created_versions), best.get("name"),
    )
    return {
        "experiment_id": experiment_id,
        "versions_created": created_versions,
        "best_model": best.get("name"),
    }


@router.get("/{experiment_id}/notebook")
async def download_notebook(experiment_id: str, db: AsyncSession = Depends(get_db)):
    """Download the generated .ipynb for this experiment."""
    experiment = await db.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found.")
    if not experiment.notebook_path or not Path(experiment.notebook_path).exists():
        raise HTTPException(
            status_code=404,
            detail="Notebook not yet generated for this experiment.",
        )
    return FileResponse(
        experiment.notebook_path,
        media_type="application/octet-stream",
        filename=f"experiment_{experiment_id}.ipynb",
    )
