import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Enum, ForeignKey,
    Integer, String, Text,
)
from sqlalchemy.orm import relationship
from backend.core.database import Base
from backend.core.utils import utcnow

class ProblemType(str, enum.Enum):
    classification = "classification"
    regression = "regression"
    clustering = "clustering"


class DatasetStatus(str, enum.Enum):
    pending = "pending"
    profiling = "profiling"
    ready = "ready"


class ExperimentStatus(str, enum.Enum):
    pending = "pending"
    notebook_generated = "notebook_generated"
    training = "training"
    completed = "completed"


class MessageType(str, enum.Enum):
    info = "info"
    insight = "insight"
    warning = "warning"


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    upload_timestamp = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    row_count = Column(Integer, nullable=False)
    column_count = Column(Integer, nullable=False)
    problem_type = Column(Enum(ProblemType), nullable=True)
    status = Column(Enum(DatasetStatus), nullable=False, default=DatasetStatus.pending)
    profile_path = Column(String, nullable=True)

    preprocessing_configs = relationship("PreprocessingConfig", back_populates="dataset")
    experiments = relationship("Experiment", back_populates="dataset")


class PreprocessingConfig(Base):
    __tablename__ = "preprocessing_configs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False)
    config_json = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    label = Column(String, nullable=False)

    dataset = relationship("Dataset", back_populates="preprocessing_configs")
    experiments = relationship("Experiment", back_populates="preprocessing_config")


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False)
    preprocessing_config_id = Column(String(36), ForeignKey("preprocessing_configs.id"), nullable=False)
    models_config_json = Column(Text, nullable=False)
    tuning_config_json = Column(Text, nullable=True)
    notebook_path = Column(String, nullable=True)
    status = Column(Enum(ExperimentStatus), nullable=False, default=ExperimentStatus.pending)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    dataset = relationship("Dataset", back_populates="experiments")
    preprocessing_config = relationship("PreprocessingConfig", back_populates="experiments")
    model_versions = relationship("ModelVersion", back_populates="experiment")
    agent_logs = relationship("AgentLog", back_populates="experiment")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id = Column(String(36), ForeignKey("experiments.id"), nullable=False)
    model_name = Column(String, nullable=False)
    version_number = Column(Integer, nullable=False)
    metrics_json = Column(Text, nullable=False)
    cv_metrics_json = Column(Text, nullable=True)
    artifact_path = Column(String, nullable=False)
    parameters_json = Column(Text, nullable=False)
    cluster_labels_path = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    notes = Column(String, nullable=True)
    is_active = Column(Boolean, nullable=False, default=False)

    experiment = relationship("Experiment", back_populates="model_versions")
    predictions = relationship("Prediction", back_populates="model_version")


class AgentLog(Base):
    __tablename__ = "agent_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    # nullable FK — some logs are not tied to a specific experiment
    experiment_id = Column(String(36), ForeignKey("experiments.id"), nullable=True)
    agent_name = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    message_type = Column(Enum(MessageType), nullable=False, default=MessageType.info)

    experiment = relationship("Experiment", back_populates="agent_logs")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_version_id = Column(String(36), ForeignKey("model_versions.id"), nullable=False)
    input_json = Column(Text, nullable=False)
    output_json = Column(Text, nullable=False)
    predicted_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    model_version = relationship("ModelVersion", back_populates="predictions")