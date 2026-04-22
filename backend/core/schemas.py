from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, computed_field, field_serializer, model_validator
import json
from enum import Enum

from backend.core.models import DatasetStatus, ExperimentStatus, MessageType, ProblemType
from backend.core.utils import to_utc7


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

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
    profile_path: Optional[str] = None
    target_column: Optional[str] = None
    # Null until upload parsing completes
    inferred_schema_json: Optional[str] = None

    @computed_field
    @property
    def inferred_schema(self) -> dict | None:
        # Deserialise inferred_schema_json into a dict for API responses
        if self.inferred_schema_json is None:
            return None
        return json.loads(self.inferred_schema_json)

    @field_serializer("upload_timestamp")
    def serialize_upload_timestamp(self, dt: datetime) -> str:
        return to_utc7(dt).isoformat()


class DatasetUpdate(BaseModel):
    # Fields allowed for dataset update
    problem_type: ProblemType | None = None
    target_column: str | None = None


# ---------------------------------------------------------------------------
# Preprocessing — column-level enums
# ---------------------------------------------------------------------------

class ColumnType(str, Enum):
    numerical = "numerical"
    categorical = "categorical"
    text = "text"


class NumericalStrategy(str, Enum):
    standardize = "standardize"
    normalize = "normalize"
    robust = "robust"
    none = "none"


class CategoricalStrategy(str, Enum):
    onehot = "onehot"
    label = "label"
    ordinal = "ordinal"
    none = "none"


class TextStrategy(str, Enum):
    tfidf = "tfidf"
    count = "count"
    none = "none"


class ImputationStrategy(str, Enum):
    mean = "mean"
    median = "median"
    most_frequent = "most_frequent"
    constant = "constant"
    knn = "knn"
    none = "none"


# Maps each column type to its valid strategies
_VALID_STRATEGIES: dict[ColumnType, set[str]] = {
    ColumnType.numerical: {s.value for s in NumericalStrategy},
    ColumnType.categorical: {s.value for s in CategoricalStrategy},
    ColumnType.text: {s.value for s in TextStrategy},
}


class ColumnConfig(BaseModel):
    """Validates a single column's preprocessing config entry."""
    type: ColumnType
    strategy: str
    # Per-column imputation — applied before the scaler/encoder in a Pipeline step
    imputation: ImputationStrategy = ImputationStrategy.none
    # Only used when imputation == "constant"
    imputation_fill_value: Optional[Any] = None
    is_target: bool = False

    @model_validator(mode="after")
    def strategy_must_match_type(self) -> "ColumnConfig":
        allowed = _VALID_STRATEGIES[self.type]
        if self.strategy not in allowed:
            raise ValueError(
                f"Strategy '{self.strategy}' is not valid for type '{self.type}'. "
                f"Allowed: {sorted(allowed)}"
            )
        return self


# ---------------------------------------------------------------------------
# Preprocessing — global pipeline controls (SOP §5 four-section config)
# ---------------------------------------------------------------------------

class OutlierMethod(str, Enum):
    winsorise = "winsorise"
    iqr_remove = "iqr_remove"
    zscore_remove = "zscore_remove"
    none = "none"


class FeatureSelectionMethod(str, Enum):
    select_k_best = "select_k_best"
    variance_threshold = "variance_threshold"
    none = "none"


class ScoreFunc(str, Enum):
    f_classif = "f_classif"
    f_regression = "f_regression"
    mutual_info_classif = "mutual_info_classif"


class ClassBalancingMethod(str, Enum):
    smote = "smote"
    oversample = "oversample"
    undersample = "undersample"
    class_weight = "class_weight"
    none = "none"


class OutlierTreatmentConfig(BaseModel):
    method: OutlierMethod = OutlierMethod.none
    # IQR / Z-score threshold; used only when method != none
    threshold: float = 1.5


class FeatureSelectionConfig(BaseModel):
    method: FeatureSelectionMethod = FeatureSelectionMethod.none
    # Number of top features; used only for select_k_best
    k: int = 10
    # Score function; used only for select_k_best
    score_func: ScoreFunc = ScoreFunc.f_classif
    # Variance threshold; used only for variance_threshold
    variance_threshold: float = 0.0


class ClassBalancingConfig(BaseModel):
    # Ignored for regression and clustering problem types
    method: ClassBalancingMethod = ClassBalancingMethod.none


class FullPreprocessingConfig(BaseModel):
    """
    Full four-section config as specified in SOP §5.
    Sent by the frontend; persisted as JSON in preprocessing_configs.config_json.
    """
    columns: dict[str, ColumnConfig]
    outlier_treatment: OutlierTreatmentConfig = OutlierTreatmentConfig()
    feature_selection: FeatureSelectionConfig = FeatureSelectionConfig()
    class_balancing: ClassBalancingConfig = ClassBalancingConfig()


# ---------------------------------------------------------------------------
# PreprocessingConfig DB schemas
# ---------------------------------------------------------------------------

class PreprocessingConfigBase(BaseModel):
    label: str
    config_json: str


class PreprocessingConfigCreate(BaseModel):
    dataset_id: str
    label: str
    # Full four-section config — serialised to JSON before storing
    config: FullPreprocessingConfig


class PreprocessingConfigRead(PreprocessingConfigBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    created_at: datetime

    @field_serializer("created_at")
    def serialize_created_at(self, dt: datetime) -> str:
        return to_utc7(dt).isoformat()


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class ExperimentBase(BaseModel):
    models_config_json: str
    tuning_config_json: Optional[str] = None


class ExperimentCreate(ExperimentBase):
    dataset_id: str
    preprocessing_config_id: str


class ExperimentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    preprocessing_config_id: str
    models_config_json: str
    tuning_config_json: Optional[str] = None
    notebook_path: Optional[str] = None
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


# ---------------------------------------------------------------------------
# ModelVersion
# ---------------------------------------------------------------------------

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
    experiment_id: str
    model_name: str
    version_number: int
    metrics_json: str
    cv_metrics_json: Optional[str] = None
    artifact_path: str
    parameters_json: str
    cluster_labels_path: Optional[str] = None
    created_at: datetime
    notes: Optional[str] = None
    is_active: bool

    @field_serializer("created_at")
    def serialize_created_at(self, dt: datetime) -> str:
        return to_utc7(dt).isoformat()


# ---------------------------------------------------------------------------
# AgentLog
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class PredictionBase(BaseModel):
    input_json: str
    output_json: str


class PredictionCreate(PredictionBase):
    model_version_id: str


class PredictionRead(PredictionBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    model_version_id: str
    predicted_at: datetime

    @field_serializer("predicted_at")
    def serialize_predicted_at(self, dt: datetime) -> str:
        return to_utc7(dt).isoformat()