from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import require_scope
from ..models import (
    ProjectConfigCreate,
    ProjectConfigListResponse,
    ProjectConfigResponse,
    ProjectConfigSummary,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_TEMPLATES_DIR = _PROJECT_ROOT / "project_templates"

router = APIRouter(prefix="/projects", tags=["projects"])


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "", name.strip().lower().replace(" ", "_"))[:30]


@router.post(
    "",
    response_model=ProjectConfigResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_project(
    body: ProjectConfigCreate,
    _=Depends(require_scope("projects:write")),
):
    safe = _safe_name(body.project_name)
    if not safe:
        raise HTTPException(status_code=422, detail="project_name produces an empty filename")

    filepath = _TEMPLATES_DIR / f"project_config.{safe}.json"
    if filepath.exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Config already exists: {filepath.name}",
        )

    config_data = {
        "project_name": body.project_name.strip(),
        "version": body.version,
        "agent_type": body.agent_type,
        "connection": body.connection.model_dump(),
        "agent_identity": body.agent_identity.model_dump(),
        "workflows": [w.model_dump() for w in body.workflows],
        "hallucination_context": body.hallucination_context,
        "tool_patterns": (body.tool_patterns or {"patterns": [], "labels": {}})
            if isinstance(body.tool_patterns, dict)
            else (body.tool_patterns.model_dump() if body.tool_patterns else {"patterns": [], "labels": {}}),
        "step_agent_patterns": (body.step_agent_patterns.model_dump() if body.step_agent_patterns else {"patterns": []}),
        "silent_upload_patterns": (body.silent_upload_patterns.model_dump() if body.silent_upload_patterns else {"patterns": []}),
    }

    _TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps(config_data, indent=2) + "\n")

    return ProjectConfigResponse(
        filename=filepath.name,
        project_name=config_data["project_name"],
        version=config_data["version"],
        agent_type=body.agent_type,
        connection=body.connection,
        agent_identity=body.agent_identity,
        workflows=body.workflows,
        hallucination_context=body.hallucination_context,
    )


@router.get("", response_model=ProjectConfigListResponse)
async def list_projects(_=Depends(require_scope("projects:read"))):
    configs: list[ProjectConfigSummary] = []
    if _TEMPLATES_DIR.exists():
        for f in sorted(_TEMPLATES_DIR.glob("project_config.*.json")):
            if "template" in f.name or "schema" in f.name:
                continue
            try:
                data = json.loads(f.read_text())
                conn = data.get("connection", {})
                identity = data.get("agent_identity", {})
                workflows = data.get("workflows", [])
                configs.append(ProjectConfigSummary(
                    filename=f.name,
                    project_name=data.get("project_name", f.stem),
                    version=data.get("version", ""),
                    agent_type=data.get("agent_type", "conversational"),
                    agent_name=conn.get("agent_name", ""),
                    domain=identity.get("domain", ""),
                    workflow_count=len(workflows),
                ))
            except Exception:
                pass
    return ProjectConfigListResponse(configs=configs, count=len(configs))
