from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict

from backend.core.models import DatasetStatus, ExperimentStatus, MessageType, ProblemType
from datetime import datetime
from pydantic import field_serializer
from backend.core.utils import to_utc7

# Dataset
class DatasetBase(BaseModel):
    name: str
    # Nullable until user sets it via PATCH after upload
    problem_type: Optional[ProblemType] = None
    status: DatasetStatus = DatasetStatus.pending

class DatasetCreate(DatasetBase):
    pass

class DatasetRead(DatasetBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    row_count: int
    column_count: int
    upload_timestamp: datetime
    # Null until background profiling task completes
    profile_path: Optional[str] = None

    @field_serializer("upload_timestamp")
    def serialize_upload_timestamp(self, dt: datetime) -> str:
        return to_utc7(dt).isoformat()


# PreprocessingConfig
class PreprocessingConfigBase(BaseModel):
    label: str
    config_json: str

class PreprocessingConfigCreate(PreprocessingConfigBase):
    dataset_id: str

class PreprocessingConfigRead(PreprocessingConfigBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    created_at: datetime

    @field_serializer("created_at")
    def serialize_created_at(self, dt: datetime) -> str:
        return to_utc7(dt).isoformat()


# Experiment
class ExperimentBase(BaseModel):
    models_config_json: str
    tuning_config_json: Optional[str] = None

class ExperimentCreate(ExperimentBase):
    dataset_id: str
    preprocessing_config_id: str

class ExperimentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: ExperimentStatus
    created_at: datetime
    # Null until results are uploaded
    completed_at: Optional[datetime] = None

    @field_serializer("created_at")
    def serialize_created_at(self, dt: datetime) -> str:
        return to_utc7(dt).isoformat()

    @field_serializer("completed_at")
    def serialize_completed_at(self, dt: datetime | None) -> str | None:
        return to_utc7(dt).isoformat() if dt is not None else None


# ModelVersion
class ModelVersionBase(BaseModel):
    model_name: str
    parameters_json: str
    metrics_json: str

class ModelVersionCreate(ModelVersionBase):
    experiment_id: str
    version_number: int
    artifact_path: str
    # Null for clustering versions — no cross-validation
    cv_metrics_json: Optional[str] = None
    # Null for supervised versions
    cluster_labels_path: Optional[str] = None
    notes: Optional[str] = None

class ModelVersionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    model_name: str
    version_number: int
    metrics_json: str
    cv_metrics_json: Optional[str] = None
    cluster_labels_path: Optional[str] = None
    is_active: bool


# AgentLog
class AgentLogBase(BaseModel):
    agent_name: str
    message: str
    message_type: MessageType = MessageType.info

class AgentLogCreate(AgentLogBase):
    # Nullable FK — some logs are not tied to a specific experiment
    experiment_id: Optional[str] = None

class AgentLogRead(AgentLogBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    experiment_id: Optional[str] = None
    created_at: datetime

    @field_serializer("created_at")
    def serialize_created_at(self, dt: datetime) -> str:
        return to_utc7(dt).isoformat()


# Prediction
class PredictionBase(BaseModel):
    input_json: str
    output_json: str

class PredictionCreate(PredictionBase):
    model_version_id: str

class PredictionRead(PredictionBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    predicted_at: datetime

    @field_serializer("predicted_at")
    def serialize_predicted_at(self, dt: datetime) -> str:
        return to_utc7(dt).isoformat()