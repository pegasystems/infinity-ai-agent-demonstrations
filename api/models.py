from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# --- Shared sub-models ---

class ConnectionConfig(BaseModel):
    base_url: str
    agent_name: str
    a2a_app_path: str = ""
    token_url_override: Optional[str] = None


class AgentIdentityConfig(BaseModel):
    role: str = ""
    domain: str = ""
    organization: str = ""
    off_topic_guidance: str = ""


class WorkflowStage(BaseModel):
    name: str
    description: str = ""


class WorkflowConfig(BaseModel):
    id: str
    description: str = ""
    stages: list[WorkflowStage] = []


class ToolPatternsConfig(BaseModel):
    patterns: list[str] = []
    labels: dict[str, str] = {}


class StepAgentPatternsConfig(BaseModel):
    patterns: list[str] = []


class SilentUploadPatternsConfig(BaseModel):
    patterns: list[str] = []


# --- Project config ---

class ProjectConfigCreate(BaseModel):
    project_name: str
    version: str = "1.0"
    agent_type: str = "conversational"
    connection: ConnectionConfig
    agent_identity: AgentIdentityConfig
    workflows: list[WorkflowConfig] = []
    hallucination_context: list[str] = []
    tool_patterns: Optional[ToolPatternsConfig] = None
    step_agent_patterns: Optional[StepAgentPatternsConfig] = None
    silent_upload_patterns: Optional[SilentUploadPatternsConfig] = None


class ProjectConfigResponse(BaseModel):
    filename: str
    project_name: str
    version: str
    agent_type: str = "conversational"
    connection: ConnectionConfig
    agent_identity: AgentIdentityConfig
    workflows: list[WorkflowConfig]
    hallucination_context: list[str]


class ProjectConfigSummary(BaseModel):
    filename: str
    project_name: str
    version: str
    agent_type: str = "conversational"
    agent_name: str
    domain: str
    workflow_count: int


class ProjectConfigListResponse(BaseModel):
    configs: list[ProjectConfigSummary]
    count: int


# --- Golden datasets ---

class GoldenDatasetCreate(BaseModel):
    agent_output: dict[str, Any]
    session_name: Optional[str] = None
    project_config_filename: Optional[str] = None
    workflow_id: Optional[str] = None
    case_id: Optional[str] = None


class GoldenDatasetResponse(BaseModel):
    golden_filename: str
    profile_filename: str
    session_name: str
    turn_count: int
    tools_detected: list[str]
    project_name: Optional[str] = None


class GoldenDatasetSummary(BaseModel):
    filename: str
    name: str
    recorded_at: str
    turn_count: int
    tools_used: list[str]
    tools_count: int
    project_name: Optional[str] = None
    workflow_id: Optional[str] = None


class GoldenDatasetListResponse(BaseModel):
    datasets: list[GoldenDatasetSummary]
    count: int


class GoldenDatasetRename(BaseModel):
    name: str


class GoldenDatasetReplace(BaseModel):
    turns: list[dict[str, Any]]
    summary: Optional[dict[str, Any]] = None
    name: Optional[str] = None
    recorded_at: Optional[str] = None


# --- Evaluations ---

class EvaluationMetricInput(BaseModel):
    name: str
    threshold: float


class EvaluationRequest(BaseModel):
    golden_dataset: str
    metrics: list[EvaluationMetricInput]
    project_config_filename: Optional[str] = None
    llm_profile: Optional[str] = None
    case_id: Optional[str] = None


class EvaluationMetricResult(BaseModel):
    name: str
    threshold: float
    passed: bool


class EvaluationStartResponse(BaseModel):
    run_id: str
    status: str
    golden_dataset: str
    metrics: list[EvaluationMetricInput]
    started_at: str
    case_id: Optional[str] = None


class EvaluationStatusResponse(BaseModel):
    run_id: str
    status: str
    golden_dataset: str
    metrics: list[EvaluationMetricInput]
    started_at: str
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    results: Optional[list[EvaluationMetricResult]] = None
    error: Optional[str] = None
    log: Optional[str] = None
    case_id: Optional[str] = None


class EvaluationListResponse(BaseModel):
    evaluations: list[EvaluationStatusResponse]
    count: int


# --- Metrics ---

class MetricInfo(BaseModel):
    id: str
    name: str
    description: str
    default_threshold: float


class MetricsListResponse(BaseModel):
    metrics: list[MetricInfo]


# --- LLM Profiles ---

class LlmProfileInfo(BaseModel):
    name: str
    provider: str
    created_at: Optional[str] = None


class LlmProfileListResponse(BaseModel):
    profiles: list[LlmProfileInfo]
    count: int


# --- OAuth token ---

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    scope: str


class ClientInfo(BaseModel):
    client_id: str
    scopes: list[str] = Field(default_factory=list)
