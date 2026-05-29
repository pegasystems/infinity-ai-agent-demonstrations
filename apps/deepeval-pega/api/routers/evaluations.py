from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..auth import require_scope
from ..evaluation_runner import AVAILABLE_METRICS, get_run, list_llm_profiles, list_runs, start_evaluation
from ..models import (
    EvaluationListResponse,
    EvaluationRequest,
    EvaluationStartResponse,
    EvaluationStatusResponse,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_GOLDEN_DIR = _PROJECT_ROOT / "golden_sessions"
_TEMPLATES_DIR = _PROJECT_ROOT / "project_templates"

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@router.post(
    "",
    response_model=EvaluationStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_eval(
    body: EvaluationRequest,
    _=Depends(require_scope("evaluations:write")),
):
    golden_path = _GOLDEN_DIR / body.golden_dataset
    if not golden_path.exists():
        raise HTTPException(status_code=404, detail=f"Golden dataset not found: {body.golden_dataset}")

    invalid = [m.name for m in body.metrics if m.name not in AVAILABLE_METRICS]
    if invalid:
        raise HTTPException(status_code=422, detail=f"Unknown metric names: {invalid}")

    project_config = body.project_config_filename
    if project_config:
        if not (_TEMPLATES_DIR / project_config).exists():
            raise HTTPException(status_code=404, detail=f"Project config not found: {project_config}")

    llm_profile = body.llm_profile
    if llm_profile:
        available = list_llm_profiles()
        if llm_profile not in available:
            raise HTTPException(status_code=404, detail=f"LLM profile not found: {llm_profile}")

    if project_config:
        config_path = _TEMPLATES_DIR / project_config
        if config_path.exists():
            config_data = json.loads(config_path.read_text())
            if config_data.get("agent_type") == "step_agent" and not body.case_id:
                raise HTTPException(
                    status_code=422,
                    detail="Step agent requires a case_id. Provide an existing Pega case ID.",
                )

    run_id = start_evaluation(
        golden_dataset=body.golden_dataset,
        metrics=body.metrics,
        project_config=project_config,
        llm_profile=llm_profile,
        case_id=body.case_id,
    )

    run = get_run(run_id)
    return EvaluationStartResponse(
        run_id=run_id,
        status=run.status if run else "running",
        golden_dataset=body.golden_dataset,
        metrics=body.metrics,
        started_at=run.started_at if run else "",
        case_id=body.case_id,
    )


@router.get("/{run_id}", response_model=EvaluationStatusResponse)
async def get_evaluation(
    run_id: str,
    _=Depends(require_scope("evaluations:read")),
):
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Evaluation run not found: {run_id}")
    return run


@router.get("", response_model=EvaluationListResponse)
async def list_evaluations(
    limit: int = Query(default=20, ge=1, le=100),
    _=Depends(require_scope("evaluations:read")),
):
    runs = list_runs(limit=limit)
    return EvaluationListResponse(evaluations=runs, count=len(runs))
