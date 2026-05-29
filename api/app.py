from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .auth import oauth_token
from .evaluation_runner import list_llm_profile_details
from .models import LlmProfileListResponse, MetricInfo, MetricsListResponse
from .routers import datasets, evaluations, projects

AVAILABLE_METRICS = [
    {"id": "knowledge_retention", "name": "Knowledge Retention", "description": "Checks whether context from early turns is retained throughout the session", "default_threshold": 0.5},
    {"id": "hallucination", "name": "Hallucination", "description": "Detects hallucinated content not grounded in context", "default_threshold": 0.5},
    {"id": "conversation_completeness", "name": "Conversation Completeness", "description": "Verifies the agent completed all expected workflow stages", "default_threshold": 0.5},
    {"id": "role_adherence", "name": "Role Adherence", "description": "Checks that the agent stays in its designated role throughout", "default_threshold": 0.7},
    {"id": "pega_tool_correctness", "name": "Pega Tool Correctness", "description": "Verifies each turn invokes the expected Pega tools (from golden baseline)", "default_threshold": 1.0},
    {"id": "business_case_lifecycle", "name": "Business Case Lifecycle", "description": "Verifies a business case is created and persists across all turns", "default_threshold": 1.0},
    {"id": "contextual_precision", "name": "Contextual Precision", "description": "Evaluates precision of context retrieval", "default_threshold": 0.7},
    {"id": "contextual_recall", "name": "Contextual Recall", "description": "Evaluates recall of context retrieval", "default_threshold": 0.7},
    {"id": "toxicity", "name": "Toxicity", "description": "Detects toxic or harmful content in responses", "default_threshold": 0.5},
    {"id": "bias", "name": "Bias", "description": "Detects biased content in responses", "default_threshold": 0.5},
    {"id": "business_case_adherence", "name": "Business Case Adherence", "description": "Verifies the correct business case type is created per turn", "default_threshold": 1.0},
]


def create_app() -> FastAPI:
    app = FastAPI(
        title="DeepEval Pega API",
        description="REST API for managing project configurations, golden datasets, and evaluation runs.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.post("/oauth/token", tags=["auth"])(oauth_token)

    app.include_router(projects.router)
    app.include_router(datasets.router)
    app.include_router(evaluations.router)

    @app.get("/metrics", tags=["metrics"], response_model=MetricsListResponse)
    async def list_metrics():
        return MetricsListResponse(
            metrics=[MetricInfo(**m) for m in AVAILABLE_METRICS]
        )

    @app.get("/llm-profiles", tags=["llm-profiles"], response_model=LlmProfileListResponse)
    async def list_profiles():
        profiles = list_llm_profile_details()
        return LlmProfileListResponse(profiles=profiles, count=len(profiles))

    return app
