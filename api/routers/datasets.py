from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import require_scope
from ..models import (
    GoldenDatasetCreate,
    GoldenDatasetListResponse,
    GoldenDatasetRename,
    GoldenDatasetReplace,
    GoldenDatasetResponse,
    GoldenDatasetSummary,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_GOLDEN_DIR = _PROJECT_ROOT / "golden_sessions"
_TEMPLATES_DIR = _PROJECT_ROOT / "project_templates"

# Ensure project root is on sys.path so capture_golden_session can be imported
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _load_profile_for_golden(golden_data: dict) -> Optional[dict]:
    profile_ref = golden_data.get("profile")
    if not profile_ref:
        return None
    profile_path = _GOLDEN_DIR / profile_ref
    if not profile_path.exists():
        return None
    try:
        return json.loads(profile_path.read_text())
    except Exception:
        return None


def _build_summary(filename: str, data: dict) -> GoldenDatasetSummary:
    profile = _load_profile_for_golden(data)
    summary = data.get("summary", {})
    return GoldenDatasetSummary(
        filename=filename,
        name=data.get("name", Path(filename).stem),
        recorded_at=data.get("recorded_at", ""),
        turn_count=len(data.get("turns", [])),
        tools_used=summary.get("all_tools_used", []),
        tools_count=len(summary.get("all_tools_used", [])),
        project_name=profile.get("project_name") if profile else None,
        workflow_id=(profile.get("workflow", {}) or {}).get("id") if profile else None,
    )


def _list_datasets(project_name: Optional[str] = None) -> list[GoldenDatasetSummary]:
    datasets: list[GoldenDatasetSummary] = []
    if not _GOLDEN_DIR.exists():
        return datasets
    for f in sorted(_GOLDEN_DIR.glob("golden_*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
            s = _build_summary(f.name, data)
            if project_name and s.project_name != project_name:
                continue
            datasets.append(s)
        except Exception:
            pass
    return datasets


@router.post(
    "",
    response_model=GoldenDatasetResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_dataset(
    body: GoldenDatasetCreate,
    _=Depends(require_scope("datasets:write")),
):
    if "conversation_history" not in body.agent_output:
        raise HTTPException(
            status_code=422,
            detail="agent_output must contain a 'conversation_history' key",
        )

    from capture_golden_session import capture_from_structured_output, load_project_config

    pcfg = None
    if body.project_config_filename:
        config_path = _TEMPLATES_DIR / body.project_config_filename
        if not config_path.exists():
            raise HTTPException(status_code=404, detail=f"Project config not found: {body.project_config_filename}")
        pcfg = load_project_config(str(config_path))

    try:
        golden_path, profile_path = capture_from_structured_output(
            agent_output=body.agent_output,
            session_name=body.session_name,
            output_dir=str(_GOLDEN_DIR),
            project_config=pcfg,
            workflow_id=body.workflow_id,
            case_id=body.case_id,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Capture failed: {e}")

    golden_data = json.loads(golden_path.read_text())
    profile_data = json.loads(profile_path.read_text()) if profile_path.exists() else {}
    summary = golden_data.get("summary", {})

    return GoldenDatasetResponse(
        golden_filename=golden_path.name,
        profile_filename=profile_path.name,
        session_name=golden_data.get("name", golden_path.stem),
        turn_count=len(golden_data.get("turns", [])),
        tools_detected=summary.get("all_tools_used", []),
        project_name=profile_data.get("project_name"),
    )


@router.get("", response_model=GoldenDatasetListResponse)
async def list_all_datasets(_=Depends(require_scope("datasets:read"))):
    datasets = _list_datasets()
    return GoldenDatasetListResponse(datasets=datasets, count=len(datasets))


@router.get("/by-project/{project_name}", response_model=GoldenDatasetListResponse)
async def list_datasets_by_project(
    project_name: str,
    _=Depends(require_scope("datasets:read")),
):
    datasets = _list_datasets(project_name=project_name)
    return GoldenDatasetListResponse(datasets=datasets, count=len(datasets))


@router.get("/{filename}")
async def get_dataset(
    filename: str,
    _=Depends(require_scope("datasets:read")),
):
    path = _GOLDEN_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {filename}")
    return json.loads(path.read_text())


@router.patch("/{filename}", response_model=GoldenDatasetSummary)
async def rename_dataset(
    filename: str,
    body: GoldenDatasetRename,
    _=Depends(require_scope("datasets:write")),
):
    path = _GOLDEN_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {filename}")
    if not body.name.strip():
        raise HTTPException(status_code=422, detail="Name cannot be empty")
    data = json.loads(path.read_text())
    data["name"] = body.name.strip()
    path.write_text(json.dumps(data, indent=2) + "\n")
    return _build_summary(filename, data)


@router.put("/{filename}", response_model=GoldenDatasetSummary)
async def replace_dataset(
    filename: str,
    body: GoldenDatasetReplace,
    _=Depends(require_scope("datasets:write")),
):
    path = _GOLDEN_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {filename}")
    if not body.turns:
        raise HTTPException(status_code=422, detail="turns array cannot be empty")
    original = json.loads(path.read_text())
    new_data = {
        "name": body.name or original.get("name", Path(filename).stem),
        "recorded_at": body.recorded_at or original.get("recorded_at", ""),
        "profile": original.get("profile", ""),
        "turns": body.turns,
    }
    all_tools = []
    for turn in body.turns:
        all_tools.extend(turn.get("expected_tools", []))
        all_tools.extend((turn.get("insight") or {}).get("tools_detected", []))
    all_tools = sorted(set(all_tools))
    new_data["summary"] = body.summary or {
        "turn_count": len(body.turns),
        "all_tools_used": all_tools,
    }
    path.write_text(json.dumps(new_data, indent=2) + "\n")
    return _build_summary(filename, new_data)


@router.delete("/{filename}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    filename: str,
    _=Depends(require_scope("datasets:write")),
):
    path = _GOLDEN_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {filename}")
    path.unlink()
    profile_name = filename.replace("golden_", "profile_", 1)
    profile_path = _GOLDEN_DIR / profile_name
    if profile_path.exists():
        profile_path.unlink()
