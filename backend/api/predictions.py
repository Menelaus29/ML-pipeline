"""
Predictions API — SOP §8 (Ph 10).

POST /api/predictions/                  run inference on a model version
GET  /api/predictions/                  list predictions (optional model_version_id filter)
GET  /api/predictions/{prediction_id}   get single prediction
"""
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_db
from backend.core.models import ModelVersion, Prediction
from backend.core.schemas import PredictionRead

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/predictions", tags=["predictions"])


class PredictRequest(BaseModel):
    model_version_id: str
    # Flat dict of feature_name → value (any JSON-serialisable type)
    input_data: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/", status_code=status.HTTP_201_CREATED)
async def predict(payload: PredictRequest, db: AsyncSession = Depends(get_db)):
    """
    Run inference for the given model version.
    Loads the stored artifact + preprocessor, applies them to input_data,
    and returns the structured prediction output.
    Persists each prediction to the DB for audit.
    """
    mv = await db.get(ModelVersion, payload.model_version_id)
    if mv is None:
        raise HTTPException(
            status_code=404,
            detail=f"ModelVersion {payload.model_version_id} not found.",
        )

    from backend.services.prediction import predict as run_predict
    try:
        result = await run_predict(
            model_version_id=payload.model_version_id,
            input_data=payload.input_data,
            db=db,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Prediction error for version %s: %s", payload.model_version_id, e)
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return result


@router.get("/", response_model=list[PredictionRead])
async def list_predictions(
    model_version_id: str | None = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List predictions, newest first, optionally filtered by model_version_id."""
    q = select(Prediction).order_by(Prediction.predicted_at.desc()).limit(limit).offset(offset)
    if model_version_id:
        q = q.where(Prediction.model_version_id == model_version_id)
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/{prediction_id}", response_model=PredictionRead)
async def get_prediction(prediction_id: str, db: AsyncSession = Depends(get_db)):
    pred = await db.get(Prediction, prediction_id)
    if pred is None:
        raise HTTPException(status_code=404, detail=f"Prediction {prediction_id} not found.")
    return pred
