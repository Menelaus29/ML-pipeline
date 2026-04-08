import json
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.core.database import get_db
from backend.core.models import Dataset, DatasetStatus
from backend.core.schemas import DatasetRead, DatasetUpdate
from backend.core.utils import utcnow
from backend.services.ingestion import parse_upload, save_upload
from backend.services.profiling import generate_profile
from backend.core.config import settings

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

def _run_profiling(dataset_id: str, file_path: Path, dataset_name: str) -> None:
    # Run profiling and update dataset status; called as a background task
    import asyncio
    from backend.core.database import AsyncSessionLocal

    profile_storage = Path(settings.storage_dir) / "profiles"
    output_path = generate_profile(dataset_id, file_path, dataset_name, profile_storage)

    async def _update():
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
            dataset = result.scalar_one_or_none()
            if dataset:
                dataset.profile_path = str(output_path) if output_path else None
                dataset.status = DatasetStatus.ready
                await db.commit()

    asyncio.run(_update())


@router.post("/upload", response_model=DatasetRead, status_code=201)
async def upload_dataset(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> DatasetRead:
    # Accept a file upload, persist dataset record, trigger profiling in background
    contents = await file.read()
    storage_dir = Path(settings.storage_dir) / "datasets"

    # save_upload now correctly takes filename as a separate argument
    file_path = save_upload(contents, file.filename, storage_dir)

    parsed = parse_upload(file_path, file.filename)

    dataset = Dataset(
        name=file.filename,
        filepath=str(file_path),
        upload_timestamp=utcnow(),
        row_count=parsed["row_count"],
        column_count=parsed["column_count"],
        inferred_schema_json=json.dumps(parsed["inferred_schema"]),
        status=DatasetStatus.profiling,  # immediately set to profiling
        problem_type=None,
        target_column=None,
        profile_path=None,
    )

    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)

    background_tasks.add_task(
        _run_profiling,
        str(dataset.id),
        file_path,
        file.filename,
    )

    return dataset


@router.get("/", response_model=list[DatasetRead])
async def list_datasets(db: AsyncSession = Depends(get_db)) -> list[DatasetRead]:
    # Return all datasets ordered by upload_timestamp descending
    result = await db.execute(
        select(Dataset).order_by(Dataset.upload_timestamp.desc())
    )
    return result.scalars().all()


@router.get("/{dataset_id}", response_model=DatasetRead)
async def get_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)) -> DatasetRead:
    # Return a single dataset by ID
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return dataset

@router.patch("/{dataset_id}", response_model=DatasetRead)
async def update_dataset(
    dataset_id: str,
    update: DatasetUpdate,
    db: AsyncSession = Depends(get_db),
) -> DatasetRead:
    # Update problem_type and/or target_column on a dataset
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    if update.problem_type is not None:
        dataset.problem_type = update.problem_type
    if update.target_column is not None:
        dataset.target_column = update.target_column

    await db.commit()
    await db.refresh(dataset)
    return dataset

@router.get("/{dataset_id}/profile")
async def get_profile(dataset_id: str, db: AsyncSession = Depends(get_db)):
    # Return the HTML profile report, or 202 if not yet ready
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    if not dataset.profile_path or dataset.status != DatasetStatus.ready:
        return JSONResponse(
            status_code=202,
            content={"detail": "Profile report is still being generated. Try again shortly."},
        )

    profile_path = Path(dataset.profile_path)
    if not profile_path.exists():
        raise HTTPException(status_code=404, detail="Profile file not found on disk.")

    return FileResponse(
        path=str(profile_path),
        media_type="text/html",
        filename=f"profile_{dataset_id}.html",
    )