"""
Model Versions API — SOP §8.

GET    /api/models/                         list all versions (optional model_name filter)
GET    /api/models/{version_id}             get single version
POST   /api/models/{version_id}/activate    set is_active=True, deactivate siblings
GET    /api/models/diff                     diff two versions (?v1=&v2=)
POST   /api/models/{version_id}/retrain     generate retrain notebook for new dataset
DELETE /api/models/{version_id}            delete a version (cannot delete active version)
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_db
from backend.core.models import ModelVersion
from backend.core.schemas import ModelVersionRead
from backend.services.versioning import activate_version, diff_versions

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/models", tags=["model-versions"])


@router.get("/diff")
async def diff_model_versions(
    v1: str = Query(..., description="First version ID"),
    v2: str = Query(..., description="Second version ID"),
    db: AsyncSession = Depends(get_db),
):
    """Return a structured diff of two ModelVersion records (params, metrics, dataset, config)."""
    try:
        return await diff_versions(db, v1, v2)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/", response_model=list[ModelVersionRead])
async def list_model_versions(
    model_name: str | None = None,
    experiment_id: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """List all model versions, optionally filtered by model_name or experiment_id."""
    q = select(ModelVersion).order_by(ModelVersion.created_at.desc())
    if model_name:
        q = q.where(ModelVersion.model_name == model_name)
    if experiment_id:
        q = q.where(ModelVersion.experiment_id == experiment_id)
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/{version_id}", response_model=ModelVersionRead)
async def get_model_version(version_id: str, db: AsyncSession = Depends(get_db)):
    mv = await db.get(ModelVersion, version_id)
    if mv is None:
        raise HTTPException(status_code=404, detail=f"ModelVersion {version_id} not found.")
    return mv


@router.post("/{version_id}/activate", response_model=ModelVersionRead)
async def activate_model_version(version_id: str, db: AsyncSession = Depends(get_db)):
    """Set is_active=True on this version and deactivate all others with the same model_name."""
    try:
        mv = await activate_version(db, version_id)
        await db.commit()
        await db.refresh(mv)
        logger.info("Activated model version %s (%s v%d)", version_id, mv.model_name, mv.version_number)
        return mv
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{version_id}/retrain")
async def retrain_model_version(
    version_id: str,
    new_dataset_id: str = Query(..., description="ID of the new dataset to retrain on"),
    db: AsyncSession = Depends(get_db),
):
    """Generate a retrain notebook for an existing supervised model version on a new dataset."""
    from backend.services.retrain import generate_retrain_notebook
    try:
        nb_path = await generate_retrain_notebook(
            source_version_id=version_id,
            new_dataset_id=new_dataset_id,
            db=db,
        )
        return {
            "source_version_id": version_id,
            "new_dataset_id": new_dataset_id,
            "notebook_path": str(nb_path),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{version_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model_version(version_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a model version. Active versions cannot be deleted."""
    mv = await db.get(ModelVersion, version_id)
    if mv is None:
        raise HTTPException(status_code=404, detail=f"ModelVersion {version_id} not found.")
    if mv.is_active:
        raise HTTPException(
            status_code=409,
            detail="Cannot delete the active model version. Activate another version first.",
        )
    await db.delete(mv)
    await db.commit()
    logger.info("Deleted model version %s", version_id)
