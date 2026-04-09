import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_db
from backend.core.models import Dataset, PreprocessingConfig
from backend.core.schemas import PreprocessingConfigCreate, PreprocessingConfigRead
from backend.services.preprocessing import serialize_pipeline_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/preprocessing", tags=["preprocessing"])


@router.post("/configs", response_model=PreprocessingConfigRead, status_code=status.HTTP_201_CREATED)
async def create_preprocessing_config(
    payload: PreprocessingConfigCreate,
    db: AsyncSession = Depends(get_db),
):
    # Validate + serialize + store a preprocessing config for a dataset
    dataset = await db.get(Dataset, payload.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset {payload.dataset_id} not found.")

    record = PreprocessingConfig(
        dataset_id=payload.dataset_id,
        label=payload.label,
        config_json=serialize_pipeline_config(payload.config),
    )
    db.add(record)
    await db.flush()
    await db.refresh(record)

    logger.info("Created preprocessing config %s for dataset %s", record.id, payload.dataset_id)
    return record


# Must be declared before GET /configs/{dataset_id} to prevent FastAPI
# matching the literal string "detail" as a dataset_id path parameter.
@router.get("/configs/detail/{config_id}", response_model=PreprocessingConfigRead)
async def get_preprocessing_config(
    config_id: str,
    db: AsyncSession = Depends(get_db),
):
    # Return a single preprocessing config by its ID
    record = await db.get(PreprocessingConfig, config_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Preprocessing config {config_id} not found.")
    return record


@router.get("/configs/{dataset_id}", response_model=list[PreprocessingConfigRead])
async def list_preprocessing_configs(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
):
    # Return all preprocessing configs for a dataset, ordered by created_at ascending
    result = await db.execute(
        select(PreprocessingConfig)
        .where(PreprocessingConfig.dataset_id == dataset_id)
        .order_by(PreprocessingConfig.created_at)
    )
    return result.scalars().all()


@router.delete("/configs/{config_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_preprocessing_config(
    config_id: str,
    db: AsyncSession = Depends(get_db),
):
    # Delete a preprocessing config by its ID 
    record = await db.get(PreprocessingConfig, config_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Preprocessing config {config_id} not found.")

    await db.delete(record)
    logger.info("Deleted preprocessing config %s", config_id)