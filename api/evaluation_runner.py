from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import EvaluationMetricInput, EvaluationMetricResult, EvaluationStatusResponse, LlmProfileInfo

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_GOLDEN_SESSIONS_DIR = _PROJECT_ROOT / "golden_sessions"
_PROJECT_TEMPLATES_DIR = _PROJECT_ROOT / "project_templates"
_LLM_PROFILES_DIR = _PROJECT_ROOT / "llm_profiles"
_LLM_CREDENTIALS_FILE = _LLM_PROFILES_DIR / ".credentials.json"

AVAILABLE_METRICS = {
    "knowledge_retention": "Knowledge Retention",
    "hallucination": "Hallucination",
    "conversation_completeness": "Conversation Completeness",
    "role_adherence": "Role Adherence",
    "pega_tool_correctness": "Pega Tool Correctness",
    "business_case_lifecycle": "Business Case Lifecycle",
    "business_case_adherence": "Business Case Adherence",
    "contextual_precision": "Contextual Precision",
    "contextual_recall": "Contextual Recall",
    "toxicity": "Toxicity",
    "bias": "Bias",
}

_TEST_TO_METRIC = {
    "test_knowledge_retention": "knowledge_retention",
    "test_no_hallucination_per_turn": "hallucination",
    "test_conversation_completeness": "conversation_completeness",
    "test_role_adherence": "role_adherence",
    "test_tool_invocations_match_golden": "pega_tool_correctness",
    "test_case_lifecycle": "business_case_lifecycle",
    "test_business_case_adherence": "business_case_adherence",
    "test_contextual_precision": "contextual_precision",
    "test_contextual_recall": "contextual_recall",
    "test_toxicity": "toxicity",
    "test_bias": "bias",
}

_runs: dict[str, dict] = {}


def _profile_safe_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _find_profile_path(display_name: str) -> Optional[Path]:
    if not _LLM_PROFILES_DIR.exists():
        return None
    for f in _LLM_PROFILES_DIR.glob("llm_profile.*.json"):
        try:
            data = json.loads(f.read_text())
            if data.get("name") == display_name:
                return f
        except Exception:
            continue
    return None


def _load_profile_env_vars(display_name: str) -> dict[str, str]:
    profile_path = _find_profile_path(display_name)
    if not profile_path:
        raise ValueError(f"LLM profile not found: {display_name}")

    data = json.loads(profile_path.read_text())
    safe_key = profile_path.stem.replace("llm_profile.", "")
    provider = data.get("provider", "gemini")

    env_vars: dict[str, str] = {"LLM_PROVIDER": provider}

    if provider == "gemini":
        if data.get("gemini_model_id"):
            env_vars["GEMINI_MODEL_ID"] = data["gemini_model_id"]
    elif provider == "bedrock":
        env_vars["AWS_AUTH_METHOD"] = data.get("aws_auth_method", "access_keys")
        if data.get("aws_profile"):
            env_vars["AWS_PROFILE"] = data["aws_profile"]
        if data.get("aws_region"):
            env_vars["AWS_REGION"] = data["aws_region"]
        if data.get("aws_bedrock_model_id"):
            env_vars["AWS_BEDROCK_MODEL_ID"] = data["aws_bedrock_model_id"]
    elif provider == "openai":
        if data.get("openai_model_id"):
            env_vars["OPENAI_MODEL_ID"] = data["openai_model_id"]
    elif provider == "copilot":
        if data.get("copilot_model_id"):
            env_vars["GITHUB_COPILOT_MODEL_ID"] = data["copilot_model_id"]

    if _LLM_CREDENTIALS_FILE.exists():
        try:
            vault = json.loads(_LLM_CREDENTIALS_FILE.read_text())
            creds = vault.get(safe_key, {})
            for k, v in creds.items():
                if v:
                    env_vars[k] = v
        except Exception:
            pass

    return env_vars


def list_llm_profiles() -> list[str]:
    return [p.name for p in list_llm_profile_details()]


def list_llm_profile_details() -> list[LlmProfileInfo]:
    profiles: list[LlmProfileInfo] = []
    seen: set[str] = set()
    if _LLM_PROFILES_DIR.exists():
        for f in sorted(_LLM_PROFILES_DIR.glob("llm_profile.*.json")):
            try:
                data = json.loads(f.read_text())
                name = data.get("name", f.stem)
                if name not in seen:
                    seen.add(name)
                    profiles.append(LlmProfileInfo(
                        name=name,
                        provider=data.get("provider", "unknown"),
                        created_at=data.get("created_at"),
                    ))
            except Exception:
                pass
    return profiles


def _parse_pytest_results(output: str, metrics: list[str]) -> list[EvaluationMetricResult]:
    passed_tests: set[str] = set()
    failed_tests: set[str] = set()
    last_test_seen = ""

    for line in output.splitlines():
        stripped = line.lstrip()
        for test_name in _TEST_TO_METRIC:
            if test_name in line:
                last_test_seen = test_name
                if "PASSED" in line:
                    passed_tests.add(test_name)
                    last_test_seen = ""
                elif "FAILED" in line:
                    failed_tests.add(test_name)
                    last_test_seen = ""
                break
        else:
            if last_test_seen:
                if stripped.startswith("PASSED"):
                    passed_tests.add(last_test_seen)
                    last_test_seen = ""
                elif stripped.startswith("FAILED"):
                    failed_tests.add(last_test_seen)
                    last_test_seen = ""

    results: list[EvaluationMetricResult] = []
    metric_to_test = {v: k for k, v in _TEST_TO_METRIC.items()}
    for m in metrics:
        metric_id = m.name if isinstance(m, EvaluationMetricInput) else m
        threshold = m.threshold if isinstance(m, EvaluationMetricInput) else 0.0
        test_name = metric_to_test.get(metric_id)
        if not test_name:
            continue
        if test_name in passed_tests:
            results.append(EvaluationMetricResult(
                name=metric_id, threshold=threshold, passed=True,
            ))
        elif test_name in failed_tests:
            results.append(EvaluationMetricResult(
                name=metric_id, threshold=threshold, passed=False,
            ))
    return results


async def _run_evaluation_task(run_id: str) -> None:
    run = _runs[run_id]
    run["status"] = "running"

    metrics_input: list[EvaluationMetricInput] = run["metrics"]
    metric_ids = [m.name for m in metrics_input]

    golden_path = str(_GOLDEN_SESSIONS_DIR / run["golden_dataset"])
    cmd = [
        "python", "-m", "pytest",
        "test_golden_session.py",
        f"--golden={golden_path}",
        f"--metrics={','.join(metric_ids)}",
        "-v", "-s", "--tb=short",
    ]
    if run.get("project_config"):
        cmd.append(f"--project-config={_PROJECT_TEMPLATES_DIR / run['project_config']}")
    if run.get("case_id"):
        cmd.append(f"--case-id={run['case_id']}")

    env = {**os.environ}
    if run.get("llm_profile"):
        try:
            profile_env = _load_profile_env_vars(run["llm_profile"])
            env.update(profile_env)
        except ValueError:
            run["status"] = "failed"
            run["error"] = f"LLM profile not found: {run['llm_profile']}"
            now = datetime.now(timezone.utc).isoformat()
            run["completed_at"] = now
            started = datetime.fromisoformat(run["started_at"])
            run["duration_seconds"] = (datetime.fromisoformat(now) - started).total_seconds()
            return
    for m in metrics_input:
        env[f"EVAL_THRESHOLD_{m.name.upper()}"] = str(m.threshold)

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(_PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
        output = stdout.decode() + stderr.decode()
        run["log"] = output

        if process.returncode in (0, 1):
            run["status"] = "completed"
            run["results"] = _parse_pytest_results(output, metrics_input)
        else:
            run["status"] = "failed"
            run["error"] = f"pytest exited with code {process.returncode}"
    except asyncio.TimeoutError:
        run["status"] = "timed_out"
        run["error"] = "Evaluation timed out after 5 minutes"
    except Exception as e:
        run["status"] = "failed"
        run["error"] = str(e)
    finally:
        run["completed_at"] = datetime.now(timezone.utc).isoformat()
        started = datetime.fromisoformat(run["started_at"])
        completed = datetime.fromisoformat(run["completed_at"])
        run["duration_seconds"] = (completed - started).total_seconds()


def start_evaluation(
    golden_dataset: str,
    metrics: list[EvaluationMetricInput],
    project_config: Optional[str] = None,
    llm_profile: Optional[str] = None,
    case_id: Optional[str] = None,
) -> str:
    run_id = str(uuid.uuid4())
    _runs[run_id] = {
        "run_id": run_id,
        "status": "running",
        "golden_dataset": golden_dataset,
        "metrics": metrics,
        "project_config": project_config,
        "llm_profile": llm_profile,
        "case_id": case_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "duration_seconds": None,
        "results": None,
        "error": None,
        "log": None,
    }
    asyncio.create_task(_run_evaluation_task(run_id))
    return run_id


def get_run(run_id: str) -> Optional[EvaluationStatusResponse]:
    run = _runs.get(run_id)
    if not run:
        return None
    return EvaluationStatusResponse(
        run_id=run["run_id"],
        status=run["status"],
        golden_dataset=run["golden_dataset"],
        metrics=run["metrics"],
        started_at=run["started_at"],
        completed_at=run.get("completed_at"),
        duration_seconds=run.get("duration_seconds"),
        results=run.get("results"),
        error=run.get("error"),
        log=run.get("log"),
        case_id=run.get("case_id"),
    )


def list_runs(limit: int = 20) -> list[EvaluationStatusResponse]:
    sorted_runs = sorted(_runs.values(), key=lambda r: r["started_at"], reverse=True)
    return [
        EvaluationStatusResponse(
            run_id=r["run_id"],
            status=r["status"],
            golden_dataset=r["golden_dataset"],
            metrics=r["metrics"],
            started_at=r["started_at"],
            completed_at=r.get("completed_at"),
            duration_seconds=r.get("duration_seconds"),
            results=r.get("results"),
            error=r.get("error"),
            case_id=r.get("case_id"),
        )
        for r in sorted_runs[:limit]
    ]
