"""
DeepEval Pega Evaluation Web UI

A comprehensive web interface for:
1. Running DeepEval evaluations with configurable metrics
2. Managing golden datasets (create, upload, select)
"""

import reflex as rx
import json
import os
import subprocess
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from rxconfig import config

# Import structured capture function
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from capture_golden_session import capture_from_structured_output, load_project_config, _normalize_workflows
import llm_oauth

# Available DeepEval metrics for evaluation
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

# Path to golden sessions directory
GOLDEN_SESSIONS_DIR = Path(__file__).parent.parent / "golden_sessions"

# Path to project config template and configs directory
PROJECT_ROOT = Path(__file__).parent.parent
PROJECT_TEMPLATES_DIR = PROJECT_ROOT / "project_templates"
PROJECT_CONFIG_TEMPLATE = PROJECT_TEMPLATES_DIR / "project_config.template.json"

# LLM judge profiles directory and credentials vault
LLM_PROFILES_DIR = PROJECT_ROOT / "llm_profiles"
LLM_CREDENTIALS_FILE = LLM_PROFILES_DIR / ".credentials.json"


class EvaluationResult(BaseModel):
    """Model for evaluation results."""
    metric_name: str
    score: float
    passed: bool
    reason: str
    threshold: float


class GoldenDataset(BaseModel):
    """Model for golden dataset metadata."""
    filename: str
    name: str
    recorded_at: str
    turn_count: int
    conversation_ids: List[str]
    tools_used: List[str]
    tools_count: int = 0
    project_name: str = ""


class ApiClientInfo(BaseModel):
    """Model for OAuth client display info."""
    client_id: str
    description: str = ""
    created_at: str = ""


class State(rx.State):
    """The main app state."""
    
    # --- Navigation ---
    active_tab: str = "evaluation"
    
    # --- Evaluation Section State ---
    selected_metrics: List[str] = ["knowledge_retention", "hallucination"]
    metric_thresholds: Dict[str, float] = {
        "knowledge_retention": 0.5,
        "hallucination": 0.5,
        "conversation_completeness": 0.5,
        "role_adherence": 0.7,
        "pega_tool_correctness": 1.0,
        "business_case_lifecycle": 1.0,
        "contextual_precision": 0.7,
        "contextual_recall": 0.7,
        "toxicity": 0.5,
        "bias": 0.5,
        "business_case_adherence": 1.0,
    }
    selected_dataset: str = ""
    available_datasets: List[GoldenDataset] = []
    evaluation_running: bool = False
    evaluation_results: List[EvaluationResult] = []
    evaluation_log: str = ""
    evaluation_status: str = ""
    evaluation_project_config: str = ""  # Selected project config for evaluation
    evaluation_configs: List[str] = []  # Available project configs for evaluation
    _eval_config_map: dict = {}
    evaluation_case_id: str = ""  # Pega case ID for step agent evaluation
    evaluation_agent_type: str = "conversational"  # Derived from selected project config
    evaluation_project_name: str = ""  # project_name from selected config (for dataset filtering)
    
    # --- Golden Dataset Section State ---
    dataset_creation_mode: str = "capture"  # "capture", "agent_output", "manual", or "upload"
    capture_conversation_id: str = ""
    capture_session_name: str = ""
    capture_project_config: str = ""  # Selected project config for capture
    capture_configs: List[str] = []  # Available project configs for capture
    capture_status: str = ""
    capture_running: bool = False

    # --- Conversation Listing State ---
    capture_conversations: list = []
    capture_conversations_loading: bool = False
    capture_conversations_error: str = ""
    capture_selected_conv_details: dict = {}
    capture_conv_details_loading: bool = False

    # --- Agent Output JSON Mode ---
    agent_output_json: str = ""  # Raw agent output JSON for structured capture
    agent_output_name: str = ""  # Session name for agent output capture
    agent_output_error: str = ""  # Validation error for agent output JSON
    
    manual_json_content: str = ""
    manual_dataset_name: str = ""
    json_validation_error: str = ""
    
    # --- Dataset Preview ---
    preview_dataset_str: str = ""
    show_preview: bool = False

    # --- Dataset Edit ---
    rename_dataset_filename: str = ""
    rename_dataset_new_name: str = ""
    replace_dataset_filename: str = ""
    datasets_filter_project: str = ""
    
    # --- File Upload ---
    uploaded_files: List[str] = []
    
    # --- Project Configuration State ---
    config_project_name: str = ""
    config_version: str = "1.0"
    config_agent_type: str = "conversational"
    config_base_url: str = ""
    config_agent_name: str = ""
    config_a2a_app_path: str = ""
    config_token_url_override: str = ""
    config_pega_client_id: str = ""
    config_pega_client_secret: str = ""
    config_role: str = ""
    config_domain: str = ""
    config_organization: str = ""
    config_off_topic_guidance: str = ""
    config_workflows: str = ""  # JSON string for the workflows array
    config_workflows_validation: str = ""  # feedback from validate_workflows_json
    config_hallucination_context: str = ""  # Newline-separated strings
    config_status: str = ""
    available_configs: List[str] = []
    selected_config: str = ""

    # --- LLM Judge Settings ---
    llm_provider: str = "Google Gemini"        # "Google Gemini", "AWS Bedrock", "OpenAI", "GitHub Copilot", or "Anthropic"
    # Credential input buffers — never pre-populated from .env; cleared after each save
    llm_gemini_api_key: str = ""
    llm_gemini_model_id: str = "gemini-2.5-flash"
    llm_aws_auth_method: str = "Access Keys"   # "Access Keys" or "SSO Profile"
    llm_aws_access_key_id: str = ""
    llm_aws_secret_access_key: str = ""
    llm_aws_profile: str = ""
    llm_aws_region: str = "us-east-1"
    llm_aws_bedrock_model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    llm_openai_api_key: str = ""
    llm_openai_model_id: str = "gpt-4o"
    llm_copilot_api_key: str = ""
    llm_copilot_model_id: str = "openai/gpt-4o"
    llm_anthropic_api_key: str = ""
    llm_anthropic_model_id: str = "claude-sonnet-4-5"
    # Saved-key indicators (safe to transmit — booleans only, never the actual key value)
    llm_gemini_key_saved: bool = False
    llm_aws_access_key_saved: bool = False
    llm_openai_key_saved: bool = False
    llm_copilot_key_saved: bool = False
    llm_anthropic_key_saved: bool = False
    llm_settings_status: str = ""
    llm_test_status: str = ""
    llm_testing: bool = False
    # Model listing state for dynamic dropdowns
    llm_gemini_models: list = []
    llm_gemini_models_loading: bool = False
    llm_gemini_models_error: str = ""
    llm_bedrock_models: list = []
    llm_bedrock_models_loading: bool = False
    llm_bedrock_models_error: str = ""
    llm_openai_models: list = []
    llm_openai_models_loading: bool = False
    llm_openai_models_error: str = ""
    llm_copilot_models: list = []
    llm_copilot_models_loading: bool = False
    llm_copilot_models_error: str = ""
    llm_anthropic_models: list = []
    llm_anthropic_models_loading: bool = False
    llm_anthropic_models_error: str = ""

    # --- OAuth ("Sign in with your account") for OpenAI / Copilot / Anthropic ---
    # Per-provider auth method: "API Key" (default) or "Sign in"
    llm_openai_auth_method: str = "API Key"
    llm_copilot_auth_method: str = "API Key"
    llm_anthropic_auth_method: str = "API Key"
    # Signed-in indicators + human-readable status
    llm_openai_signed_in: bool = False
    llm_copilot_signed_in: bool = False
    llm_anthropic_signed_in: bool = False
    llm_openai_oauth_status: str = ""
    llm_copilot_oauth_status: str = ""
    llm_anthropic_oauth_status: str = ""
    # GitHub Copilot device-code flow
    llm_copilot_user_code: str = ""
    llm_copilot_verification_uri: str = ""
    llm_copilot_login_active: bool = False
    # OpenAI / Anthropic PKCE flow: authorize URL, ephemeral verifier/state, pasted code
    llm_openai_authorize_url: str = ""
    llm_openai_pkce_verifier: str = ""
    llm_openai_pkce_state: str = ""
    llm_openai_code_input: str = ""
    llm_anthropic_authorize_url: str = ""
    llm_anthropic_pkce_verifier: str = ""
    llm_anthropic_pkce_state: str = ""
    llm_anthropic_code_input: str = ""

    # --- LLM Judge Profiles ---
    available_llm_profiles: List[str] = []
    selected_llm_profile: str = ""
    llm_profile_name_input: str = ""
    llm_profile_status: str = ""

    # --- API Server Management ---
    api_server_running: bool = False
    api_server_pid: int = 0
    api_server_port: str = "8100"
    api_server_status: str = ""
    api_clients: List[ApiClientInfo] = []
    api_new_client_description: str = ""
    api_client_status: str = ""
    api_created_client_id: str = ""
    api_created_client_secret: str = ""

    @rx.var
    def filtered_datasets(self) -> List[GoldenDataset]:
        """Return datasets filtered by the selected evaluation project config."""
        if not self.evaluation_project_name:
            return []
        return [d for d in self.available_datasets if d.project_name == self.evaluation_project_name]

    @rx.var
    def gemini_model_options(self) -> list:
        options = list(self.llm_gemini_models)
        current = self.llm_gemini_model_id.strip()
        if current and current not in options:
            options.insert(0, current)
        return options

    @rx.var
    def bedrock_model_options(self) -> list:
        options = list(self.llm_bedrock_models)
        current = self.llm_aws_bedrock_model_id.strip()
        if current and current not in options:
            options.insert(0, current)
        return options

    @rx.var
    def openai_model_options(self) -> list:
        options = list(self.llm_openai_models)
        current = self.llm_openai_model_id.strip()
        if current and current not in options:
            options.insert(0, current)
        return options

    @rx.var
    def copilot_model_options(self) -> list:
        options = list(self.llm_copilot_models)
        current = self.llm_copilot_model_id.strip()
        if current and current not in options:
            options.insert(0, current)
        return options

    @rx.var
    def anthropic_model_options(self) -> list:
        options = list(self.llm_anthropic_models)
        current = self.llm_anthropic_model_id.strip()
        if current and current not in options:
            options.insert(0, current)
        return options

    @rx.var
    def managed_datasets(self) -> List[GoldenDataset]:
        if not self.datasets_filter_project:
            return self.available_datasets
        return [d for d in self.available_datasets if d.project_name == self.datasets_filter_project]

    @rx.var
    def datasets_filter_options(self) -> list:
        projects = sorted(set(
            d.project_name for d in self.available_datasets if d.project_name
        ))
        return ["All Projects"] + projects

    def init_app(self):
        """Initialize the app by loading all required data."""
        self.api_created_client_id = ""
        self.api_created_client_secret = ""
        self.load_available_datasets()
        self.load_evaluation_configs()
        self.load_capture_configs()
        self.load_llm_settings()
        self.load_available_llm_profiles()
        self.load_api_clients()
        self._check_api_server_status()

    def set_active_tab(self, tab: str):
        """Switch between evaluation and dataset management tabs."""
        self.active_tab = tab
        self.api_created_client_id = ""
        self.api_created_client_secret = ""
        if tab == "config":
            self.load_available_configs()
            self.load_config_template()
            self.load_llm_settings()
            self.load_available_llm_profiles()
            self.load_api_clients()
            self._check_api_server_status()
        if tab == "evaluation":
            self.load_available_datasets()
            self.load_evaluation_configs()
    
    def load_evaluation_configs(self):
        """Load list of available project configurations for evaluation."""
        configs = []
        self._eval_config_map = {}
        for f in sorted(PROJECT_TEMPLATES_DIR.glob("project_config.*.json")):
            if "template" not in f.name and "schema" not in f.name:
                try:
                    data = json.loads(f.read_text())
                    display_name = data.get("project_name", f.name)
                except Exception:
                    display_name = f.name
                configs.append(display_name)
                self._eval_config_map[display_name] = f.name
        self.evaluation_configs = configs

    def set_evaluation_project_config(self, display_name: str):
        """Set the project configuration and update environment variable."""
        filename = getattr(self, "_eval_config_map", {}).get(display_name, display_name)
        self.evaluation_project_config = display_name
        self.selected_dataset = ""
        if filename:
            full_path = str(PROJECT_TEMPLATES_DIR / filename)
            os.environ["PROJECT_CONFIG"] = full_path
            self.evaluation_status = ""
            config_path = PROJECT_TEMPLATES_DIR / filename
            if config_path.exists():
                try:
                    data = json.loads(config_path.read_text())
                    self.evaluation_agent_type = data.get("agent_type", "conversational")
                    self.evaluation_project_name = data.get("project_name", "")
                except Exception:
                    self.evaluation_agent_type = "conversational"
                    self.evaluation_project_name = ""
            else:
                self.evaluation_agent_type = "conversational"
                self.evaluation_project_name = ""
        else:
            if "PROJECT_CONFIG" in os.environ:
                del os.environ["PROJECT_CONFIG"]
            self.evaluation_status = ""
            self.evaluation_agent_type = "conversational"
            self.evaluation_project_name = ""
            self.evaluation_case_id = ""

    def set_evaluation_case_id(self, value: str):
        self.evaluation_case_id = value

    def toggle_metric(self, metric_id: str):
        """Toggle a metric selection."""
        if metric_id in self.selected_metrics:
            self.selected_metrics = [m for m in self.selected_metrics if m != metric_id]
        else:
            self.selected_metrics = self.selected_metrics + [metric_id]

    def update_threshold(self, metric_id: str, value: str):
        """Update threshold for a metric."""
        try:
            threshold = float(value)
            if 0 <= threshold <= 1:
                self.metric_thresholds = {**self.metric_thresholds, metric_id: threshold}
        except ValueError:
            pass

    def select_dataset(self, filename: str):
        """Select a golden dataset for evaluation."""
        self.selected_dataset = filename

    def delete_dataset(self, filename: str):
        """Delete a golden dataset file and its companion profile, then refresh the list."""
        path = GOLDEN_SESSIONS_DIR / filename
        profile_name = filename.replace("golden_", "profile_", 1)
        profile_path = GOLDEN_SESSIONS_DIR / profile_name
        try:
            path.unlink(missing_ok=True)
            profile_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"Error deleting {filename}: {e}")
            return
        if self.selected_dataset == filename:
            self.selected_dataset = ""
        self.load_available_datasets()

    def rename_dataset(self, filename: str, new_name: str):
        """Rename a golden dataset (updates the name field inside the JSON)."""
        if not new_name.strip():
            return
        path = GOLDEN_SESSIONS_DIR / filename
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            data["name"] = new_name.strip()
            path.write_text(json.dumps(data, indent=2) + "\n")
            self.load_available_datasets()
        except Exception as e:
            print(f"Error renaming {filename}: {e}")

    def start_replace_dataset(self, filename: str):
        """Enter replace mode for a dataset — creation methods will overwrite this file."""
        self.replace_dataset_filename = filename

    def cancel_replace_dataset(self):
        """Exit replace mode."""
        self.replace_dataset_filename = ""

    def set_rename_dataset_new_name(self, value: str):
        self.rename_dataset_new_name = value

    def set_datasets_filter_project(self, value: str):
        self.datasets_filter_project = "" if value == "All Projects" else value

    def _replace_file_with_data(self, new_data: dict):
        """Overwrite the target golden file with new data, preserving metadata."""
        path = GOLDEN_SESSIONS_DIR / self.replace_dataset_filename
        if not path.exists():
            return
        original = json.loads(path.read_text())
        if "name" not in new_data:
            new_data["name"] = original.get("name", path.stem)
        if "recorded_at" not in new_data:
            new_data["recorded_at"] = original.get("recorded_at", "")
        if "profile" not in new_data:
            new_data["profile"] = original.get("profile", "")
        if "turns" in new_data:
            all_tools = []
            for turn in new_data["turns"]:
                all_tools.extend(turn.get("expected_tools", []))
                all_tools.extend((turn.get("insight") or {}).get("tools_detected", []))
            new_data.setdefault("summary", {
                "turn_count": len(new_data["turns"]),
                "all_tools_used": sorted(set(all_tools)),
            })
        path.write_text(json.dumps(new_data, indent=2) + "\n")
        self.replace_dataset_filename = ""

    def load_available_datasets(self):
        """Load list of available golden datasets."""
        datasets = []
        if GOLDEN_SESSIONS_DIR.exists():
            for f in sorted(GOLDEN_SESSIONS_DIR.glob("golden_*.json"), reverse=True):
                try:
                    with open(f) as fp:
                        data = json.load(fp)
                        tools = data.get("summary", {}).get("all_tools_used", [])
                        # Look up project_name from companion profile file
                        profile_name = f.name.replace("golden_", "profile_", 1)
                        profile_path = GOLDEN_SESSIONS_DIR / profile_name
                        project_name = ""
                        if profile_path.exists():
                            try:
                                with open(profile_path) as pf:
                                    project_name = json.load(pf).get("project_name", "")
                            except Exception:
                                pass
                        datasets.append(GoldenDataset(
                            filename=f.name,
                            name=data.get("name", f.stem),
                            recorded_at=data.get("recorded_at", "Unknown"),
                            turn_count=len(data.get("turns", [])),
                            conversation_ids=data.get("conversation_ids", []),
                            tools_used=tools,
                            tools_count=len(tools),
                            project_name=project_name,
                        ))
                except Exception as e:
                    print(f"Error loading {f}: {e}")
        self.available_datasets = datasets

    def preview_selected_dataset(self):
        """Load and preview the selected dataset."""
        if not self.selected_dataset:
            return
        try:
            filepath = GOLDEN_SESSIONS_DIR / self.selected_dataset
            with open(filepath) as f:
                data = json.load(f)
            self.preview_dataset_str = json.dumps(data, indent=2)
            self.show_preview = True
        except Exception as e:
            self.evaluation_status = f"Error loading dataset: {e}"

    def close_preview(self):
        """Close the dataset preview modal."""
        self.show_preview = False
        self.preview_dataset_str = ""

    async def run_evaluation(self):
        """Run the DeepEval evaluation with selected metrics and dataset.
        
        Uses async subprocess and yields to update state in the UI during execution.
        """
        if not self.selected_dataset:
            self.evaluation_status = "Please select a golden dataset first."
            yield
            return
        
        if not self.selected_metrics:
            self.evaluation_status = "Please select at least one metric."
            yield
            return

        if self.evaluation_agent_type == "step_agent" and not self.evaluation_case_id.strip():
            self.evaluation_status = "Step agent requires a Pega Case ID. Please provide one."
            yield
            return
        
        self.evaluation_results = []       # clear tiles before any yield
        self.evaluation_running = True
        self.evaluation_log = "Starting evaluation...\n"
        self.evaluation_status = "Running evaluation..."
        yield  # Update UI to show running state
        
        try:
            # Build the pytest command
            golden_path = str(GOLDEN_SESSIONS_DIR / self.selected_dataset)
            
            # Run the test_golden_session.py with the selected golden file
            cmd = [
                "python", "-m", "pytest",
                "test_golden_session.py",
                f"--golden={golden_path}",
                f"--metrics={','.join(self.selected_metrics)}",
                "-v", "-s",
                "--tb=short",
            ]
            if self.evaluation_project_config:
                cmd.append(
                    f"--project-config={PROJECT_TEMPLATES_DIR / self.evaluation_project_config}"
                )
            if self.evaluation_case_id.strip():
                cmd.append(f"--case-id={self.evaluation_case_id.strip()}")

            # Pass metric thresholds as environment variables so test_golden_session.py
            # can pick them up when instantiating DeepEval metrics.
            # Format: EVAL_THRESHOLD_<METRIC_ID_UPPER> = float value
            env = {**os.environ}
            for metric_id, threshold in self.metric_thresholds.items():
                env[f"EVAL_THRESHOLD_{metric_id.upper()}"] = str(threshold)

            self.evaluation_log += f"Running: {' '.join(cmd)}\n\n"
            yield  # Update UI with command

            # Execute the evaluation asynchronously
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(GOLDEN_SESSIONS_DIR.parent),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            
            # Wait for the process to complete with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=300  # 5 minute timeout
                )
                output = stdout.decode() + stderr.decode()
                self.evaluation_log += output
                
                # pytest exit codes: 0 = all passed, 1 = some tests failed (normal),
                # 2+ = internal/infrastructure error (interrupted, bad CLI, etc.)
                if process.returncode in (0, 1):
                    # Tests ran — parse actual pass/fail from output
                    # Map test function name → metric id
                    test_to_metric = {
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
                    passed_tests: set = set()
                    failed_tests: set = set()
                    # pytest emits outcomes in two formats:
                    #   inline:    "test_golden_session.py::test_role_adherence PASSED"
                    #   separated: "test_golden_session.py::test_role_adherence\nPASSED"
                    # We track the last test name seen so the second format works too.
                    last_test_seen: str = ""
                    for line in output.splitlines():
                        stripped = line.lstrip()
                        # Detect test name on this line
                        for test_name in test_to_metric:
                            if test_name in line:
                                last_test_seen = test_name
                                # Inline outcome on the same line
                                if "PASSED" in line:
                                    passed_tests.add(test_name)
                                    last_test_seen = ""
                                elif "FAILED" in line:
                                    failed_tests.add(test_name)
                                    last_test_seen = ""
                                break
                        else:
                            # No test name on this line — check for standalone outcome
                            if last_test_seen:
                                if stripped.startswith("PASSED"):
                                    passed_tests.add(last_test_seen)
                                    last_test_seen = ""
                                elif stripped.startswith("FAILED"):
                                    failed_tests.add(last_test_seen)
                                    last_test_seen = ""

                    for metric_id in self.selected_metrics:
                        metric_info = next((m for m in AVAILABLE_METRICS if m["id"] == metric_id), None)
                        if not metric_info:
                            continue
                        test_name = next((t for t, m in test_to_metric.items() if m == metric_id), None)
                        threshold = self.metric_thresholds.get(metric_id, metric_info["default_threshold"])
                        if test_name in passed_tests:
                            passed = True
                            reason = "Passed threshold"
                        elif test_name in failed_tests:
                            passed = False
                            reason = "Did not meet threshold"
                        else:
                            # Test was skipped or not in selected set — omit tile
                            continue
                        self.evaluation_results.append(EvaluationResult(
                            metric_name=metric_info["name"],
                            score=threshold if passed else 0.0,
                            passed=passed,
                            reason=reason,
                            threshold=threshold,
                        ))

                    total = len(passed_tests) + len(failed_tests)
                    if process.returncode == 0 or not failed_tests:
                        self.evaluation_status = f"✅ Evaluation complete — {len(passed_tests)}/{total} tests passed"
                    else:
                        self.evaluation_status = f"⚠️ Evaluation complete — {len(failed_tests)} test(s) failed, {len(passed_tests)} passed"
                else:
                    self.evaluation_status = f"❌ Evaluation error (exit code: {process.returncode})"
                
            except asyncio.TimeoutError:
                process.kill()
                self.evaluation_log += "\nError: Evaluation timed out after 5 minutes"
                self.evaluation_status = "❌ Evaluation timed out"
                
        except Exception as e:
            self.evaluation_log += f"\nError: {str(e)}"
            self.evaluation_status = f"❌ Error running evaluation: {str(e)}"
        
        finally:
            self.evaluation_running = False
            yield  # Final UI update

    def set_creation_mode(self, mode: str):
        """Set dataset creation mode."""
        self.dataset_creation_mode = mode
        self.capture_status = ""
        self.json_validation_error = ""

    def set_conversation_id(self, value: str):
        """Set the conversation ID for capture."""
        self.capture_conversation_id = value

    def set_session_name(self, value: str):
        """Set the session name for capture."""
        self.capture_session_name = value

    def load_capture_configs(self):
        """Load list of available project configurations for capture."""
        configs = []
        for f in sorted(PROJECT_TEMPLATES_DIR.glob("project_config.*.json")):
            if "template" not in f.name and "schema" not in f.name:
                configs.append(f.name)
        self.capture_configs = configs

    def set_capture_project_config(self, filename: str):
        """Set the project configuration for capture, update env var, and persist to .env."""
        self._clear_conversations()
        self.capture_project_config = filename
        if filename:
            os.environ["PROJECT_CONFIG"] = filename
            env_updates: dict = {"PROJECT_CONFIG": filename}

            # Load connection values from the config and sync them to .env
            config_path = PROJECT_TEMPLATES_DIR / filename
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                conn = cfg.get("connection", {})
                if conn.get("base_url"):
                    env_updates["AGENTX_BASE_URL"] = conn["base_url"]
                    os.environ["AGENTX_BASE_URL"] = conn["base_url"]
                if conn.get("agent_name"):
                    env_updates["AGENT_NAME"] = conn["agent_name"]
                    os.environ["AGENT_NAME"] = conn["agent_name"]
                self._write_env_keys(env_updates)
            except Exception as e:
                self.capture_status = f"⚠️ Config selected but .env update failed: {e}"
        else:
            if "PROJECT_CONFIG" in os.environ:
                del os.environ["PROJECT_CONFIG"]
            self.capture_status = ""

    async def capture_golden_session(self):
        """Run the capture_golden_session.py script.

        Uses async subprocess and yields to update state in the UI during execution.
        If in replace mode, overwrites the target dataset with captured data.
        """
        if not self.capture_conversation_id:
            self.capture_status = "Please enter a conversation ID (e.g., PXCONV-12345)"
            yield
            return

        self.capture_running = True
        self.capture_status = "Capturing golden session..."
        yield

        try:
            cmd = [
                "python", "capture_golden_session.py",
                self.capture_conversation_id,
            ]

            if self.capture_session_name:
                cmd.extend(["--name", self.capture_session_name])

            # Snapshot existing files to detect the new one
            existing = set(GOLDEN_SESSIONS_DIR.glob("golden_*.json")) if GOLDEN_SESSIONS_DIR.exists() else set()

            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(GOLDEN_SESSIONS_DIR.parent),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=120
                )
                output = stdout.decode() + stderr.decode()

                if process.returncode == 0:
                    if self.replace_dataset_filename:
                        # Find the newly created file
                        current = set(GOLDEN_SESSIONS_DIR.glob("golden_*.json"))
                        new_files = current - existing
                        if new_files:
                            new_file = max(new_files, key=lambda f: f.stat().st_mtime)
                            new_data = json.loads(new_file.read_text())
                            self._replace_file_with_data(new_data)
                            new_file.unlink(missing_ok=True)
                            profile_f = GOLDEN_SESSIONS_DIR / new_file.name.replace("golden_", "profile_", 1)
                            profile_f.unlink(missing_ok=True)
                        self.capture_status = "✅ Dataset replaced successfully"
                    else:
                        self.capture_status = f"✅ Golden session captured successfully!\n\n{output[-500:]}"
                    self.capture_conversation_id = ""
                    self.capture_session_name = ""
                else:
                    self.capture_status = f"❌ Capture failed:\n{output[-1000:]}"

            except asyncio.TimeoutError:
                process.kill()
                self.capture_status = "❌ Capture timed out after 2 minutes"

        except Exception as e:
            self.capture_status = f"❌ Error: {str(e)}"

        finally:
            self.capture_running = False
            self.load_available_datasets()
            yield

    def _clear_conversations(self):
        """Reset conversation listing state."""
        self.capture_conversations = []
        self.capture_conversations_error = ""
        self.capture_selected_conv_details = {}
        self.capture_conversation_id = ""

    async def load_conversations(self):
        """Fetch conversation list from Pega DX API data view."""
        if not self.capture_project_config:
            self.capture_conversations_error = "Select a project configuration first."
            yield
            return

        self.capture_conversations_loading = True
        self.capture_conversations_error = ""
        self.capture_conversations = []
        self.capture_selected_conv_details = {}
        yield

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._fetch_conversations_sync
            )
            self.capture_conversations = result
            if not result:
                self.capture_conversations_error = "No conversations found for this agent."
        except Exception as e:
            self.capture_conversations_error = f"Failed to load conversations: {e}"
        finally:
            self.capture_conversations_loading = False
        yield

    def _fetch_conversations_sync(self) -> list:
        """Synchronous helper: authenticate + list conversations via data view query."""
        import requests as _requests
        from urllib.parse import quote as _quote

        base_url = os.environ.get("AGENTX_BASE_URL", "").rstrip("/")
        client_id = os.environ.get("PEGA_CLIENT_ID", "")
        client_secret = os.environ.get("PEGA_CLIENT_SECRET", "")

        if not all([base_url, client_id, client_secret]):
            raise ValueError("Missing connection credentials. Check project config and .env file.")

        token_url = f"{base_url}/prweb/PRRestService/oauth2/v1/token"
        auth_resp = _requests.post(token_url, data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }, verify=False, timeout=15)
        if auth_resp.status_code == 503:
            raise RuntimeError("Pega server unavailable (HTTP 503). The instance may be starting up — try again in a moment.")
        if auth_resp.status_code in (401, 403):
            raise RuntimeError(f"Authentication failed (HTTP {auth_resp.status_code}). Verify PEGA_CLIENT_ID and PEGA_CLIENT_SECRET.")
        if auth_resp.status_code != 200:
            raise RuntimeError(f"Authentication failed (HTTP {auth_resp.status_code}): {auth_resp.text[:200]}")
        access_token = auth_resp.json()["access_token"]

        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
        dx_base = f"{base_url}/prweb/api/application/v2"

        data_view_names = ["D_pxAutopilotConversations", "D_pxAutopilotConversationList"]

        # Load custom data view name from project config if configured
        config_path = PROJECT_TEMPLATES_DIR / self.capture_project_config
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            custom_dv = cfg.get("connection", {}).get("conversation_list_data_view")
            if custom_dv:
                data_view_names.insert(0, custom_dv)
        except Exception:
            pass

        last_error = None
        for dv_name in data_view_names:
            url = f"{dx_base}/data_views/{dv_name}"
            resp = _requests.post(url, headers=headers, json={"includeTotalCount": True}, verify=False, timeout=30)
            if resp.status_code in (404, 400, 503):
                last_error = f"Data view '{dv_name}' returned HTTP {resp.status_code}"
                continue
            if resp.status_code != 200:
                last_error = f"Data view '{dv_name}' returned HTTP {resp.status_code}: {resp.text[:100]}"
                continue
            data = resp.json()
            items = data.get("data", data.get("results", []))
            if isinstance(data, list):
                items = data
            conversations = []
            for item in items:
                py_id = item.get("pyID", item.get("pxInsName", ""))
                if not py_id:
                    continue
                created = item.get("pxCreateDateTime", "")[:10]
                status = item.get("pyStatusWork", "")
                creator = item.get("pxCreateOpName", item.get("pxCreateOperator", ""))
                label = f"{py_id} ({status}) — {created}"
                if creator:
                    label += f" — {creator}"
                conversations.append(label)
            conversations.sort(key=lambda x: x.split("—")[1].strip() if "—" in x else "", reverse=True)
            return conversations

        raise RuntimeError(last_error or "No suitable data view found for listing conversations.")

    async def select_conversation(self, value: str):
        """Handle conversation selection from dropdown and fetch details."""
        conv_id = value.split(" ")[0].strip() if value else ""
        self.capture_conversation_id = conv_id
        if not conv_id:
            self.capture_selected_conv_details = {}
            yield
            return

        self.capture_conv_details_loading = True
        self.capture_selected_conv_details = {}
        yield

        try:
            loop = asyncio.get_event_loop()
            details = await loop.run_in_executor(
                None, lambda: self._fetch_conversation_details_sync(conv_id)
            )
            self.capture_selected_conv_details = details
        except Exception as e:
            self.capture_selected_conv_details = {"error": str(e)}
        finally:
            self.capture_conv_details_loading = False
        yield

    def _fetch_conversation_details_sync(self, conversation_id: str) -> dict:
        """Synchronous helper: fetch conversation details via D_pxAutopilotConversation."""
        import requests as _requests
        from urllib.parse import quote as _quote
        import html as _html

        base_url = os.environ.get("AGENTX_BASE_URL", "").rstrip("/")
        client_id = os.environ.get("PEGA_CLIENT_ID", "")
        client_secret = os.environ.get("PEGA_CLIENT_SECRET", "")

        token_url = f"{base_url}/prweb/PRRestService/oauth2/v1/token"
        auth_resp = _requests.post(token_url, data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }, verify=False, timeout=15)
        auth_resp.raise_for_status()
        access_token = auth_resp.json()["access_token"]

        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
        params_json = json.dumps({"InteractionID": conversation_id})
        url = f"{base_url}/prweb/api/application/v2/data_views/D_pxAutopilotConversation?dataViewParameters={_quote(params_json)}"

        resp = _requests.get(url, headers=headers, verify=False, timeout=30)
        resp.raise_for_status()
        raw = resp.json()

        messages = raw.get("pyMessages", [])
        user_msgs = [m for m in messages if m.get("pyRole") == "USER"]
        assistant_msgs = [m for m in messages if m.get("pyRole") == "ASSISTANT"]

        first_user_msg = ""
        if user_msgs:
            raw_content = user_msgs[0].get("pyContent", "")
            first_user_msg = _html.unescape(raw_content)[:120]

        return {
            "id": conversation_id,
            "turn_count": str(len(user_msgs)),
            "total_messages": str(len(messages)),
            "status": raw.get("pyStatusWork", "Unknown"),
            "first_user_msg": first_user_msg,
        }

    def set_agent_output_json(self, value: str):
        """Update agent output JSON content and validate."""
        self.agent_output_json = value
        self.validate_agent_output()

    def set_agent_output_name(self, value: str):
        """Set the session name for agent output capture."""
        self.agent_output_name = value

    def validate_agent_output(self):
        """Validate the agent output JSON has required fields."""
        if not self.agent_output_json.strip():
            self.agent_output_error = ""
            return
        
        try:
            data = json.loads(self.agent_output_json)
            
            # Check for required fields in agent output format
            if "conversation_history" not in data:
                self.agent_output_error = "Missing 'conversation_history' - this doesn't look like a Pega agent output"
                return
            
            # Check conversation_history has history entries
            conv_hist = data.get("conversation_history", [])
            if not conv_hist or not any(c.get("history") for c in conv_hist):
                self.agent_output_error = "conversation_history is empty or has no history entries"
                return
            
            self.agent_output_error = ""
            
        except json.JSONDecodeError as e:
            self.agent_output_error = f"Invalid JSON: {str(e)}"

    async def capture_from_agent_output(self):
        """Capture golden session from structured Pega agent output JSON.

        If in replace mode, overwrites the target dataset with captured data.
        """
        if not self.agent_output_json.strip():
            self.capture_status = "Please paste the Pega agent output JSON"
            yield
            return

        if self.agent_output_error:
            self.capture_status = f"❌ {self.agent_output_error}"
            yield
            return

        self.capture_running = True
        self.capture_status = "Processing agent output..."
        yield

        try:
            agent_output = json.loads(self.agent_output_json)
            session_name = self.agent_output_name or agent_output.get("name", "agent_session")

            pcfg = load_project_config(
                str(PROJECT_TEMPLATES_DIR / self.capture_project_config)
                if self.capture_project_config else None
            )

            golden_path, profile_path = capture_from_structured_output(
                agent_output=agent_output,
                session_name=session_name,
                output_dir=str(GOLDEN_SESSIONS_DIR),
                project_config=pcfg,
            )

            if self.replace_dataset_filename:
                new_data = json.loads(golden_path.read_text())
                self._replace_file_with_data(new_data)
                golden_path.unlink(missing_ok=True)
                profile_path.unlink(missing_ok=True)
                self.capture_status = "✅ Dataset replaced successfully"
            else:
                self.capture_status = (
                    f"✅ Golden session captured successfully!\n\n"
                    f"Golden session: {golden_path.name}\n"
                    f"Profile: {profile_path.name}\n\n"
                    f"Tools detected from structured data (tool_calls)"
                )

            self.agent_output_json = ""
            self.agent_output_name = ""
            self.load_available_datasets()

        except Exception as e:
            self.capture_status = f"❌ Error: {str(e)}"

        finally:
            self.capture_running = False
            yield

    def set_manual_json(self, value: str):
        """Update manual JSON content and validate."""
        self.manual_json_content = value
        self.validate_json()

    def set_manual_name(self, value: str):
        """Set manual dataset name."""
        self.manual_dataset_name = value

    def validate_json(self):
        """Validate the manually entered JSON."""
        if not self.manual_json_content.strip():
            self.json_validation_error = ""
            return
        
        try:
            data = json.loads(self.manual_json_content)
            
            # Check required fields
            required = ["turns"]
            missing = [f for f in required if f not in data]
            if missing:
                self.json_validation_error = f"Missing required fields: {', '.join(missing)}"
                return
            
            # Validate turns structure
            turns = data.get("turns", [])
            if not isinstance(turns, list):
                self.json_validation_error = "'turns' must be a list"
                return
            
            if len(turns) == 0:
                self.json_validation_error = "At least one turn is required"
                return
            
            for i, turn in enumerate(turns):
                if "input" not in turn:
                    self.json_validation_error = f"Turn {i+1} missing 'input' field"
                    return
                if "response" not in turn:
                    self.json_validation_error = f"Turn {i+1} missing 'response' field"
                    return
            
            self.json_validation_error = ""
            
        except json.JSONDecodeError as e:
            self.json_validation_error = f"Invalid JSON: {str(e)}"

    def save_manual_dataset(self):
        """Save manually created golden dataset, or replace target if in replace mode."""
        if not self.manual_json_content.strip():
            self.json_validation_error = "Please enter JSON content"
            return

        if self.json_validation_error:
            return

        try:
            data = json.loads(self.manual_json_content)

            if self.replace_dataset_filename:
                self._replace_file_with_data(data)
                self.capture_status = f"✅ Dataset replaced successfully"
            else:
                data["recorded_at"] = datetime.now().isoformat()
                data["capture_method"] = "manual_entry"
                data["name"] = self.manual_dataset_name or "Manual Dataset"
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_name = "".join(c for c in data["name"] if c.isalnum() or c in "-_ ")[:30].strip().replace(" ", "_")
                filename = f"golden_{safe_name}_{ts}.json"
                filepath = GOLDEN_SESSIONS_DIR / filename
                GOLDEN_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2)
                self.capture_status = f"✅ Golden dataset saved: {filename}"

            self.manual_json_content = ""
            self.manual_dataset_name = ""
            self.load_available_datasets()

        except Exception as e:
            self.json_validation_error = f"Error saving: {str(e)}"

    def load_template_json(self):
        """Load a template JSON for manual dataset creation."""
        template = {
            "name": "My Golden Dataset",
            "conversation_ids": ["PXCONV-XXXXX"],
            "turns": [
                {
                    "turn": 1,
                    "input": "User message here",
                    "description": "Description of this turn",
                    "expected_tools": [],
                    "response": {
                        "text": "Expected agent response",
                        "context_id": "PXCONV-XXXXX",
                        "message_id": "MSG-XXXXX",
                        "latency_ms": 0
                    },
                    "insight": {
                        "tools_detected": [],
                        "step_agents": []
                    }
                }
            ],
            "summary": {
                "total_turns": 1,
                "all_tools_used": [],
                "step_agent_count": 0
            }
        }
        self.manual_json_content = json.dumps(template, indent=2)
        self.validate_json()

    async def handle_upload(self, files: list[rx.UploadFile]):
        """Handle golden dataset file upload, or replace target if in replace mode."""
        for file in files:
            try:
                upload_data = await file.read()
                data = json.loads(upload_data.decode("utf-8"))

                if "turns" not in data:
                    self.capture_status = f"❌ Invalid golden session file: missing 'turns'"
                    continue

                if self.replace_dataset_filename:
                    self._replace_file_with_data(data)
                    self.capture_status = f"✅ Dataset replaced successfully"
                else:
                    filename = file.filename or "uploaded_dataset.json"
                    if not filename.startswith("golden_"):
                        filename = f"golden_{filename}"
                    if not filename.endswith(".json"):
                        filename = f"{filename}.json"
                    filepath = GOLDEN_SESSIONS_DIR / filename
                    GOLDEN_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
                    with open(filepath, "w") as f:
                        json.dump(data, f, indent=2)
                    self.capture_status = f"✅ Uploaded: {filename}"
                    self.uploaded_files = self.uploaded_files + [filename]

            except json.JSONDecodeError:
                self.capture_status = f"❌ Invalid JSON in uploaded file"
            except Exception as e:
                self.capture_status = f"❌ Upload error: {str(e)}"

        self.load_available_datasets()

    # --- Project Configuration Methods ---
    
    def _load_pega_credentials_from_env(self):
        """Read PEGA_CLIENT_ID and PEGA_CLIENT_SECRET from the .env file."""
        env_path = PROJECT_ROOT / ".env"
        env_data: dict = {}
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    key, _, value = stripped.partition("=")
                    env_data[key.strip()] = value.strip()
        self.config_pega_client_id = env_data.get("PEGA_CLIENT_ID", "")
        self.config_pega_client_secret = env_data.get("PEGA_CLIENT_SECRET", "")

    def load_config_template(self):
        """Load the project config template as defaults."""
        try:
            if PROJECT_CONFIG_TEMPLATE.exists():
                with open(PROJECT_CONFIG_TEMPLATE) as f:
                    template = json.load(f)

                # Populate form fields from template
                self.config_project_name = template.get("project_name", "My Agent Project")
                self.config_version = template.get("version", "1.0")
                self.config_agent_type = template.get("agent_type", "conversational")

                conn = template.get("connection", {})
                self.config_base_url = conn.get("base_url", "")
                self.config_agent_name = conn.get("agent_name", "")
                self.config_a2a_app_path = conn.get("a2a_app_path", "")
                self.config_token_url_override = conn.get("token_url_override") or ""
                self._load_pega_credentials_from_env()
                
                identity = template.get("agent_identity", {})
                self.config_role = identity.get("role", "")
                self.config_domain = identity.get("domain", "")
                self.config_organization = identity.get("organization", "")
                self.config_off_topic_guidance = identity.get("off_topic_guidance", "")
                
                self.config_workflows = json.dumps(_normalize_workflows(template), indent=2)
                
                hallucination = template.get("hallucination_context", [])
                self.config_hallucination_context = "\n".join(hallucination)
                
                self.config_status = "Template loaded"
        except Exception as e:
            self.config_status = f"❌ Error loading template: {str(e)}"
    
    def load_available_configs(self):
        """Load list of existing project configuration files."""
        configs = []
        for f in sorted(PROJECT_TEMPLATES_DIR.glob("project_config.*.json")):
            if "template" not in f.name and "schema" not in f.name:
                configs.append(f.name)
        self.available_configs = configs
    
    def load_existing_config(self, filename: str):
        """Load an existing project configuration file."""
        try:
            filepath = PROJECT_TEMPLATES_DIR / filename
            if filepath.exists():
                with open(filepath) as f:
                    config_data = json.load(f)
                
                self.config_project_name = config_data.get("project_name", "")
                self.config_version = config_data.get("version", "1.0")
                self.config_agent_type = config_data.get("agent_type", "conversational")

                conn = config_data.get("connection", {})
                self.config_base_url = conn.get("base_url", "")
                self.config_agent_name = conn.get("agent_name", "")
                self.config_a2a_app_path = conn.get("a2a_app_path", "")
                self.config_token_url_override = conn.get("token_url_override") or ""
                self._load_pega_credentials_from_env()

                identity = config_data.get("agent_identity", {})
                self.config_role = identity.get("role", "")
                self.config_domain = identity.get("domain", "")
                self.config_organization = identity.get("organization", "")
                self.config_off_topic_guidance = identity.get("off_topic_guidance", "")

                self.config_workflows = json.dumps(_normalize_workflows(config_data), indent=2)
                
                hallucination = config_data.get("hallucination_context", [])
                self.config_hallucination_context = "\n".join(hallucination)
                
                self.selected_config = filename
                self.config_status = f"✅ Loaded: {filename}"
        except Exception as e:
            self.config_status = f"❌ Error loading config: {str(e)}"
    
    def save_project_config(self):
        """Save the project configuration to a file."""
        if not self.config_project_name.strip():
            self.config_status = "❌ Project name is required"
            return
        
        try:
            # Parse workflows JSON
            try:
                workflows = json.loads(self.config_workflows) if self.config_workflows.strip() else []
            except json.JSONDecodeError:
                self.config_status = "❌ Invalid JSON in workflows"
                return
            
            # Build configuration object
            config_data = {
                "project_name": self.config_project_name.strip(),
                "version": self.config_version.strip() or "1.0",
                "agent_type": self.config_agent_type,
                "connection": {
                    "base_url": self.config_base_url.strip(),
                    "agent_name": self.config_agent_name.strip(),
                    "a2a_app_path": self.config_a2a_app_path.strip(),
                    "token_url_override": self.config_token_url_override.strip() or None
                },
                "agent_identity": {
                    "role": self.config_role.strip(),
                    "domain": self.config_domain.strip(),
                    "organization": self.config_organization.strip(),
                    "off_topic_guidance": self.config_off_topic_guidance.strip()
                },
                "workflows": workflows,
                "hallucination_context": [
                    line.strip() for line in self.config_hallucination_context.split("\n")
                    if line.strip()
                ],
                "tool_patterns": {"patterns": [], "labels": {}},
                "step_agent_patterns": {"patterns": []},
                "silent_upload_patterns": {"patterns": []}
            }
            
            # Generate filename from project name
            safe_name = "".join(
                c for c in self.config_project_name.strip().lower()
                if c.isalnum() or c in "-_"
            ).replace(" ", "_")[:30]
            filename = f"project_config.{safe_name}.json"
            filepath = PROJECT_TEMPLATES_DIR / filename
            
            with open(filepath, "w") as f:
                json.dump(config_data, f, indent=2)

            # Write OAuth credentials to .env
            env_updates: dict = {}
            if self.config_pega_client_id.strip():
                env_updates["PEGA_CLIENT_ID"] = self.config_pega_client_id.strip()
            if self.config_pega_client_secret.strip():
                env_updates["PEGA_CLIENT_SECRET"] = self.config_pega_client_secret.strip()
            if env_updates:
                self._write_env_keys(env_updates)
                for key, value in env_updates.items():
                    os.environ[key] = value

            self.config_status = f"✅ Saved: {filename}"
            self.selected_config = filename
            self.load_available_configs()
            
        except Exception as e:
            self.config_status = f"❌ Error saving: {str(e)}"

    def delete_project_config(self):
        """Delete the selected project config and any associated profile/golden session files."""
        if not self.selected_config:
            self.config_status = "❌ No configuration selected to delete"
            return

        config_path = PROJECT_TEMPLATES_DIR / self.selected_config

        try:
            # Read config first to get project_name for finding associated files
            project_name = ""
            if config_path.exists():
                with open(config_path) as f:
                    project_name = json.load(f).get("project_name", "")

            # Delete associated profile and golden session files
            extra_deleted = 0
            if project_name and GOLDEN_SESSIONS_DIR.exists():
                for profile_path in list(GOLDEN_SESSIONS_DIR.glob("profile_*.json")):
                    try:
                        with open(profile_path) as f:
                            profile_data = json.load(f)
                        if profile_data.get("project_name") == project_name:
                            golden_ref = profile_data.get("golden_session")
                            if golden_ref:
                                golden_path = GOLDEN_SESSIONS_DIR / golden_ref
                                if golden_path.exists():
                                    golden_path.unlink()
                                    extra_deleted += 1
                            profile_path.unlink()
                            extra_deleted += 1
                    except Exception:
                        pass

            deleted_name = self.selected_config
            config_path.unlink(missing_ok=True)

            self.selected_config = ""
            self.config_status = (
                f"✅ Deleted: {deleted_name}"
                + (f" and {extra_deleted} associated file(s)" if extra_deleted else "")
            )
            self.load_available_configs()
            self.load_config_template()

        except Exception as e:
            self.config_status = f"❌ Error deleting config: {str(e)}"

    def set_config_project_name(self, value: str):
        self.config_project_name = value
    
    def set_config_version(self, value: str):
        self.config_version = value

    def set_config_agent_type(self, value: str):
        self.config_agent_type = value

    def set_config_base_url(self, value: str):
        self.config_base_url = value
    
    def set_config_agent_name(self, value: str):
        self.config_agent_name = value
    
    def set_config_a2a_app_path(self, value: str):
        self.config_a2a_app_path = value
    
    def set_config_token_url_override(self, value: str):
        self.config_token_url_override = value

    def set_config_pega_client_id(self, value: str):
        self.config_pega_client_id = value

    def set_config_pega_client_secret(self, value: str):
        self.config_pega_client_secret = value

    def set_config_role(self, value: str):
        self.config_role = value
    
    def set_config_domain(self, value: str):
        self.config_domain = value
    
    def set_config_organization(self, value: str):
        self.config_organization = value
    
    def set_config_off_topic_guidance(self, value: str):
        self.config_off_topic_guidance = value
    
    def set_config_workflows(self, value: str):
        self.config_workflows = value
        self.config_workflows_validation = ""

    def validate_workflows_json(self):
        """Validate the workflows JSON: must be a non-empty array of objects each with an 'id'."""
        raw = self.config_workflows.strip()
        if not raw:
            self.config_workflows_validation = "⚠️ Workflows field is empty."
            return
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            self.config_workflows_validation = f"❌ Invalid JSON — {e}"
            return
        if not isinstance(parsed, list):
            self.config_workflows_validation = "❌ Must be a JSON array ([ ... ])."
            return
        if len(parsed) == 0:
            self.config_workflows_validation = "⚠️ Array is empty — add at least one workflow entry."
            return
        errors = []
        for i, entry in enumerate(parsed):
            if not isinstance(entry, dict):
                errors.append(f"Entry {i}: must be an object.")
                continue
            if not entry.get("id"):
                errors.append(f"Entry {i}: missing required 'id' field.")
            if "stages" in entry and not isinstance(entry["stages"], list):
                errors.append(f"Entry {i}: 'stages' must be an array.")
        if errors:
            self.config_workflows_validation = "❌ " + "  |  ".join(errors)
        else:
            ids = [e.get("id") for e in parsed]
            self.config_workflows_validation = f"✅ Valid — {len(parsed)} workflow(s): {', '.join(ids)}"
    
    def set_config_hallucination_context(self, value: str):
        self.config_hallucination_context = value

    # --- LLM Judge Settings Methods ---

    def set_llm_provider(self, value: str):
        self.llm_provider = value
        self.llm_settings_status = ""
        self.llm_test_status = ""

    def set_llm_gemini_api_key(self, value: str):
        self.llm_gemini_api_key = value

    def set_llm_gemini_model_id(self, value: str):
        self.llm_gemini_model_id = value

    def set_llm_aws_auth_method(self, value: str):
        self.llm_aws_auth_method = value
        self.llm_test_status = ""

    def set_llm_aws_access_key_id(self, value: str):
        self.llm_aws_access_key_id = value

    def set_llm_aws_secret_access_key(self, value: str):
        self.llm_aws_secret_access_key = value

    def set_llm_aws_profile(self, value: str):
        self.llm_aws_profile = value

    def set_llm_aws_region(self, value: str):
        self.llm_aws_region = value

    def set_llm_aws_bedrock_model_id(self, value: str):
        self.llm_aws_bedrock_model_id = value

    def set_llm_openai_api_key(self, value: str):
        self.llm_openai_api_key = value

    def set_llm_openai_model_id(self, value: str):
        self.llm_openai_model_id = value

    def set_llm_copilot_api_key(self, value: str):
        self.llm_copilot_api_key = value

    def set_llm_copilot_model_id(self, value: str):
        self.llm_copilot_model_id = value.lstrip("/")

    def set_llm_anthropic_api_key(self, value: str):
        self.llm_anthropic_api_key = value

    def set_llm_anthropic_model_id(self, value: str):
        self.llm_anthropic_model_id = value

    # --- OAuth auth-method setters ---

    def set_llm_openai_auth_method(self, value: str):
        self.llm_openai_auth_method = value
        self.llm_test_status = ""

    def set_llm_copilot_auth_method(self, value: str):
        self.llm_copilot_auth_method = value
        self.llm_test_status = ""

    def set_llm_anthropic_auth_method(self, value: str):
        self.llm_anthropic_auth_method = value
        self.llm_test_status = ""

    def set_llm_openai_code_input(self, value: str):
        self.llm_openai_code_input = value

    def set_llm_anthropic_code_input(self, value: str):
        self.llm_anthropic_code_input = value

    def refresh_oauth_status(self):
        """Refresh signed-in indicators from the OAuth credential vault."""
        self.llm_openai_signed_in = llm_oauth.is_signed_in("openai")
        self.llm_copilot_signed_in = llm_oauth.is_signed_in("copilot")
        self.llm_anthropic_signed_in = llm_oauth.is_signed_in("anthropic")
        self.llm_openai_oauth_status = llm_oauth.status_label("openai")
        self.llm_copilot_oauth_status = llm_oauth.status_label("copilot")
        self.llm_anthropic_oauth_status = llm_oauth.status_label("anthropic")

    # --- GitHub Copilot device-code sign-in ---

    @rx.event(background=True)
    async def start_copilot_login(self):
        """Begin the GitHub device-code flow and poll until authorized."""
        loop = asyncio.get_event_loop()
        try:
            data = await loop.run_in_executor(None, llm_oauth.copilot_start_device_login)
        except Exception as e:
            async with self:
                self.llm_copilot_oauth_status = f"❌ Could not start sign-in: {e}"
            return
        device_code = data.get("device_code", "")
        interval = int(data.get("interval", 5)) or 5
        expires_in = int(data.get("expires_in", 900)) or 900
        async with self:
            self.llm_copilot_login_active = True
            self.llm_copilot_user_code = data.get("user_code", "")
            self.llm_copilot_verification_uri = data.get(
                "verification_uri", "https://github.com/login/device"
            )
            self.llm_copilot_oauth_status = "Waiting for you to authorize on GitHub..."

        waited = 0
        while waited < expires_in:
            await asyncio.sleep(interval)
            waited += interval
            try:
                status, err = await loop.run_in_executor(
                    None, lambda: llm_oauth.copilot_poll_once(device_code)
                )
            except Exception as e:
                status, err = "error", str(e)
            if status == "ok":
                async with self:
                    self.llm_copilot_login_active = False
                    self.llm_copilot_user_code = ""
                    self.llm_copilot_verification_uri = ""
                    self.refresh_oauth_status()
                    self.llm_copilot_oauth_status = "✅ Signed in to GitHub Copilot"
                return
            if status == "slow_down":
                interval += 5
                continue
            if status == "error":
                async with self:
                    self.llm_copilot_login_active = False
                    self.llm_copilot_oauth_status = f"❌ Sign-in failed: {err}"
                return
        async with self:
            self.llm_copilot_login_active = False
            self.llm_copilot_oauth_status = "❌ Sign-in timed out — please try again"

    def sign_out_copilot(self):
        llm_oauth.sign_out("copilot")
        self.llm_copilot_login_active = False
        self.llm_copilot_user_code = ""
        self.llm_copilot_verification_uri = ""
        self.refresh_oauth_status()
        self.llm_copilot_oauth_status = "Signed out of GitHub Copilot"

    # --- OpenAI (ChatGPT) PKCE sign-in ---

    def start_openai_login(self):
        url, state, verifier = llm_oauth.openai_build_authorize_url()
        self.llm_openai_authorize_url = url
        self.llm_openai_pkce_state = state
        self.llm_openai_pkce_verifier = verifier
        self.llm_openai_oauth_status = (
            "Open the sign-in link, then paste the resulting code (or full redirect URL) below."
        )

    async def complete_openai_login(self):
        code, _state = llm_oauth.parse_code_input(self.llm_openai_code_input)
        if not code:
            self.llm_openai_oauth_status = "❌ Paste the authorization code first"
            return
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None, lambda: llm_oauth.openai_complete_login(code)
            )
        except Exception as e:
            self.llm_openai_oauth_status = f"❌ Sign-in failed: {e}"
            return
        self.llm_openai_authorize_url = ""
        self.llm_openai_code_input = ""
        self.llm_openai_pkce_verifier = ""
        self.llm_openai_pkce_state = ""
        self.refresh_oauth_status()
        self.llm_openai_oauth_status = "✅ Signed in with ChatGPT"

    def sign_out_openai(self):
        llm_oauth.sign_out("openai")
        self.llm_openai_authorize_url = ""
        self.refresh_oauth_status()
        self.llm_openai_oauth_status = "Signed out of ChatGPT"

    # --- Anthropic (Claude) PKCE sign-in ---

    def start_anthropic_login(self):
        url, state, verifier = llm_oauth.anthropic_build_authorize_url()
        self.llm_anthropic_authorize_url = url
        self.llm_anthropic_pkce_state = state
        self.llm_anthropic_pkce_verifier = verifier
        self.llm_anthropic_oauth_status = (
            "Open the sign-in link, authorize, then paste the code (code#state) below."
        )

    async def complete_anthropic_login(self):
        code, pasted_state = llm_oauth.parse_code_input(self.llm_anthropic_code_input)
        if not code:
            self.llm_anthropic_oauth_status = "❌ Paste the authorization code first"
            return
        loop = asyncio.get_event_loop()
        try:
            # Prefer the state that came bound to the pasted code; fall back to
            # the server-side pending state inside anthropic_complete_login.
            await loop.run_in_executor(
                None, lambda: llm_oauth.anthropic_complete_login(code, state=pasted_state)
            )
        except Exception as e:
            self.llm_anthropic_oauth_status = f"❌ Sign-in failed: {e}"
            return
        self.llm_anthropic_authorize_url = ""
        self.llm_anthropic_code_input = ""
        self.llm_anthropic_pkce_verifier = ""
        self.llm_anthropic_pkce_state = ""
        self.refresh_oauth_status()
        self.llm_anthropic_oauth_status = "✅ Signed in with Claude"

    def sign_out_anthropic(self):
        llm_oauth.sign_out("anthropic")
        self.llm_anthropic_authorize_url = ""
        self.refresh_oauth_status()
        self.llm_anthropic_oauth_status = "Signed out of Claude"

    def load_llm_settings(self):
        """Load non-sensitive LLM settings from .env; set saved-key indicators.

        Raw credential values are intentionally NOT loaded into state to prevent
        them from being transmitted to the browser over the Reflex WebSocket.
        """
        _PROVIDER_MAP = {"gemini": "Google Gemini", "bedrock": "AWS Bedrock", "openai": "OpenAI", "copilot": "GitHub Copilot", "anthropic": "Anthropic"}
        _AUTH_MAP = {"access_keys": "Access Keys", "sso_profile": "SSO Profile"}
        _OAUTH_LABEL = {"api_key": "API Key", "oauth": "Sign in"}
        env_data = self._read_env_creds()
        raw_provider = env_data.get("LLM_PROVIDER", "gemini").lower()
        self.llm_provider = _PROVIDER_MAP.get(raw_provider, "Google Gemini")
        raw_auth = env_data.get("AWS_AUTH_METHOD", "access_keys").lower()
        self.llm_aws_auth_method = _AUTH_MAP.get(raw_auth, "Access Keys")
        # Per-provider OAuth vs API-key method
        self.llm_openai_auth_method = _OAUTH_LABEL.get(
            env_data.get("OPENAI_AUTH_METHOD", "api_key").lower(), "API Key"
        )
        self.llm_copilot_auth_method = _OAUTH_LABEL.get(
            env_data.get("COPILOT_AUTH_METHOD", "api_key").lower(), "API Key"
        )
        self.llm_anthropic_auth_method = _OAUTH_LABEL.get(
            env_data.get("ANTHROPIC_AUTH_METHOD", "api_key").lower(), "API Key"
        )
        self.refresh_oauth_status()
        # Non-sensitive settings — safe to store in state
        self.llm_aws_profile = env_data.get("AWS_PROFILE", "")
        self.llm_aws_region = env_data.get("AWS_REGION", "us-east-1")
        self.llm_aws_bedrock_model_id = env_data.get(
            "AWS_BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"
        )
        self.llm_gemini_model_id = env_data.get("GEMINI_MODEL_ID", "gemini-2.5-flash")
        self.llm_openai_model_id = env_data.get("OPENAI_MODEL_ID", "gpt-4o")
        self.llm_copilot_model_id = env_data.get("GITHUB_COPILOT_MODEL_ID", "openai/gpt-4o")
        self.llm_anthropic_model_id = env_data.get("ANTHROPIC_MODEL_ID", "claude-sonnet-4-5")
        # Boolean indicators only — actual key values stay in .env, off the WebSocket
        self.llm_gemini_key_saved = bool(env_data.get("GEMINI_API_KEY", "").strip())
        self.llm_aws_access_key_saved = bool(env_data.get("AWS_ACCESS_KEY_ID", "").strip())
        self.llm_openai_key_saved = bool(env_data.get("OPENAI_API_KEY", "").strip())
        self.llm_copilot_key_saved = bool(env_data.get("GITHUB_COPILOT_TOKEN", "").strip())
        self.llm_anthropic_key_saved = bool(env_data.get("ANTHROPIC_API_KEY", "").strip())
        # Ensure credential input buffers are always empty on load
        self.llm_gemini_api_key = ""
        self.llm_aws_access_key_id = ""
        self.llm_aws_secret_access_key = ""
        self.llm_openai_api_key = ""
        self.llm_copilot_api_key = ""
        self.llm_anthropic_api_key = ""

    def _read_env_creds(self) -> dict:
        """Read all .env key-value pairs server-side without touching state."""
        env_path = PROJECT_ROOT / ".env"
        env_data: dict = {}
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    key, _, value = stripped.partition("=")
                    env_data[key.strip()] = value.strip()
        return env_data

    def save_llm_settings(self):
        """Persist LLM provider and credentials to the .env file."""
        _PROVIDER_INTERNAL = {"Google Gemini": "gemini", "AWS Bedrock": "bedrock", "OpenAI": "openai", "GitHub Copilot": "copilot", "Anthropic": "anthropic"}
        _AUTH_INTERNAL = {"Access Keys": "access_keys", "SSO Profile": "sso_profile"}
        try:
            internal = _PROVIDER_INTERNAL.get(self.llm_provider, "gemini")
            updates: dict = {"LLM_PROVIDER": internal}

            if self.llm_provider == "Google Gemini":
                if self.llm_gemini_api_key.strip():
                    updates["GEMINI_API_KEY"] = self.llm_gemini_api_key.strip()
                if self.llm_gemini_model_id.strip():
                    updates["GEMINI_MODEL_ID"] = self.llm_gemini_model_id.strip()
            elif self.llm_provider == "AWS Bedrock":
                auth_internal = _AUTH_INTERNAL.get(self.llm_aws_auth_method, "access_keys")
                updates["AWS_AUTH_METHOD"] = auth_internal
                if self.llm_aws_auth_method == "SSO Profile":
                    if self.llm_aws_profile.strip():
                        updates["AWS_PROFILE"] = self.llm_aws_profile.strip()
                else:
                    if self.llm_aws_access_key_id.strip():
                        updates["AWS_ACCESS_KEY_ID"] = self.llm_aws_access_key_id.strip()
                    if self.llm_aws_secret_access_key.strip():
                        updates["AWS_SECRET_ACCESS_KEY"] = self.llm_aws_secret_access_key.strip()
                if self.llm_aws_region.strip():
                    updates["AWS_REGION"] = self.llm_aws_region.strip()
                if self.llm_aws_bedrock_model_id.strip():
                    updates["AWS_BEDROCK_MODEL_ID"] = self.llm_aws_bedrock_model_id.strip()
            elif self.llm_provider == "OpenAI":
                updates["OPENAI_AUTH_METHOD"] = (
                    "oauth" if self.llm_openai_auth_method == "Sign in" else "api_key"
                )
                if self.llm_openai_api_key.strip():
                    updates["OPENAI_API_KEY"] = self.llm_openai_api_key.strip()
                if self.llm_openai_model_id.strip():
                    updates["OPENAI_MODEL_ID"] = self.llm_openai_model_id.strip()
            elif self.llm_provider == "GitHub Copilot":
                updates["COPILOT_AUTH_METHOD"] = (
                    "oauth" if self.llm_copilot_auth_method == "Sign in" else "api_key"
                )
                raw_token = self.llm_copilot_api_key.strip()
                if raw_token:
                    updates["GITHUB_COPILOT_TOKEN"] = raw_token.encode("ascii", errors="ignore").decode("ascii").strip()
                if self.llm_copilot_model_id.strip():
                    updates["GITHUB_COPILOT_MODEL_ID"] = self.llm_copilot_model_id.strip()
            elif self.llm_provider == "Anthropic":
                updates["ANTHROPIC_AUTH_METHOD"] = (
                    "oauth" if self.llm_anthropic_auth_method == "Sign in" else "api_key"
                )
                if self.llm_anthropic_api_key.strip():
                    updates["ANTHROPIC_API_KEY"] = self.llm_anthropic_api_key.strip()
                if self.llm_anthropic_model_id.strip():
                    updates["ANTHROPIC_MODEL_ID"] = self.llm_anthropic_model_id.strip()

            self._write_env_keys(updates)

            # Also update the running process so pytest subprocess picks them up
            for key, value in updates.items():
                os.environ[key] = value

            # Clear credential buffers from state so values are not retained in
            # WebSocket state after the save completes
            self.llm_gemini_api_key = ""
            self.llm_aws_access_key_id = ""
            self.llm_aws_secret_access_key = ""
            self.llm_openai_api_key = ""
            self.llm_copilot_api_key = ""
            self.llm_anthropic_api_key = ""
            # Refresh saved-key indicators
            self.llm_gemini_key_saved = bool(updates.get("GEMINI_API_KEY", "").strip() or os.environ.get("GEMINI_API_KEY", "").strip())
            self.llm_aws_access_key_saved = bool(updates.get("AWS_ACCESS_KEY_ID", "").strip() or os.environ.get("AWS_ACCESS_KEY_ID", "").strip())
            self.llm_openai_key_saved = bool(updates.get("OPENAI_API_KEY", "").strip() or os.environ.get("OPENAI_API_KEY", "").strip())
            self.llm_copilot_key_saved = bool(updates.get("GITHUB_COPILOT_TOKEN", "").strip() or os.environ.get("GITHUB_COPILOT_TOKEN", "").strip())
            self.llm_anthropic_key_saved = bool(updates.get("ANTHROPIC_API_KEY", "").strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip())

            self.llm_settings_status = "✅ LLM settings saved to .env"
        except Exception as e:
            self.llm_settings_status = f"❌ Error saving LLM settings: {str(e)}"

    # --- LLM Judge Profile Management ---

    def load_available_llm_profiles(self):
        """Scan llm_profiles/ for saved profile JSON files."""
        profiles: list = []
        if LLM_PROFILES_DIR.exists():
            for f in sorted(LLM_PROFILES_DIR.glob("llm_profile.*.json")):
                try:
                    data = json.loads(f.read_text())
                    name = data.get("name", f.stem)
                    if name not in profiles:
                        profiles.append(name)
                except Exception:
                    pass
        self.available_llm_profiles = profiles

    def _profile_safe_key(self, name: str) -> str:
        """Convert a display name to a safe filesystem key."""
        import re
        return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

    def _profile_path(self, safe_key: str) -> Path:
        return LLM_PROFILES_DIR / f"llm_profile.{safe_key}.json"

    def _read_llm_credentials(self, profile_key: str) -> dict:
        """Read credentials for a profile from the vault file."""
        if not LLM_CREDENTIALS_FILE.exists():
            return {}
        try:
            vault = json.loads(LLM_CREDENTIALS_FILE.read_text())
            return vault.get(profile_key, {})
        except Exception:
            return {}

    def _write_llm_credentials(self, profile_key: str, creds: dict):
        """Write credentials for a profile to the vault file."""
        LLM_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        vault: dict = {}
        if LLM_CREDENTIALS_FILE.exists():
            try:
                vault = json.loads(LLM_CREDENTIALS_FILE.read_text())
            except Exception:
                vault = {}
        vault[profile_key] = creds
        LLM_CREDENTIALS_FILE.write_text(json.dumps(vault, indent=2) + "\n")

    def _delete_llm_credentials(self, profile_key: str):
        """Remove a profile's credentials from the vault file."""
        if not LLM_CREDENTIALS_FILE.exists():
            return
        try:
            vault = json.loads(LLM_CREDENTIALS_FILE.read_text())
            vault.pop(profile_key, None)
            LLM_CREDENTIALS_FILE.write_text(json.dumps(vault, indent=2) + "\n")
        except Exception:
            pass

    def save_llm_profile(self):
        """Save the current LLM form state as a named profile."""
        _PROVIDER_INTERNAL = {"Google Gemini": "gemini", "AWS Bedrock": "bedrock", "OpenAI": "openai", "GitHub Copilot": "copilot", "Anthropic": "anthropic"}
        _AUTH_INTERNAL = {"Access Keys": "access_keys", "SSO Profile": "sso_profile"}

        name = self.llm_profile_name_input.strip()
        if not name:
            self.llm_profile_status = "❌ Enter a profile name"
            return
        try:
            safe_key = self._profile_safe_key(name)
            LLM_PROFILES_DIR.mkdir(parents=True, exist_ok=True)

            internal_provider = _PROVIDER_INTERNAL.get(self.llm_provider, "gemini")
            auth_internal = _AUTH_INTERNAL.get(self.llm_aws_auth_method, "access_keys")

            profile_data = {
                "name": name,
                "provider": internal_provider,
                "gemini_model_id": self.llm_gemini_model_id.strip() or None,
                "aws_auth_method": auth_internal if internal_provider == "bedrock" else None,
                "aws_profile": self.llm_aws_profile.strip() or None,
                "aws_region": self.llm_aws_region.strip() or None,
                "aws_bedrock_model_id": self.llm_aws_bedrock_model_id.strip() or None,
                "openai_model_id": self.llm_openai_model_id.strip() or None,
                "copilot_model_id": self.llm_copilot_model_id.strip() or None,
                "anthropic_model_id": self.llm_anthropic_model_id.strip() or None,
                "openai_auth_method": "oauth" if self.llm_openai_auth_method == "Sign in" else "api_key",
                "copilot_auth_method": "oauth" if self.llm_copilot_auth_method == "Sign in" else "api_key",
                "anthropic_auth_method": "oauth" if self.llm_anthropic_auth_method == "Sign in" else "api_key",
                "created_at": datetime.now().isoformat(),
            }
            self._profile_path(safe_key).write_text(json.dumps(profile_data, indent=2) + "\n")

            env_data = self._read_env_creds()
            creds: dict = {}
            gemini_key = self.llm_gemini_api_key.strip() or env_data.get("GEMINI_API_KEY", "")
            if gemini_key:
                creds["GEMINI_API_KEY"] = gemini_key
            aws_key_id = self.llm_aws_access_key_id.strip() or env_data.get("AWS_ACCESS_KEY_ID", "")
            aws_secret = self.llm_aws_secret_access_key.strip() or env_data.get("AWS_SECRET_ACCESS_KEY", "")
            if aws_key_id:
                creds["AWS_ACCESS_KEY_ID"] = aws_key_id
            if aws_secret:
                creds["AWS_SECRET_ACCESS_KEY"] = aws_secret
            openai_key = self.llm_openai_api_key.strip() or env_data.get("OPENAI_API_KEY", "")
            if openai_key:
                creds["OPENAI_API_KEY"] = openai_key
            copilot_token = self.llm_copilot_api_key.strip() or env_data.get("GITHUB_COPILOT_TOKEN", "")
            if copilot_token:
                creds["GITHUB_COPILOT_TOKEN"] = copilot_token
            anthropic_key = self.llm_anthropic_api_key.strip() or env_data.get("ANTHROPIC_API_KEY", "")
            if anthropic_key:
                creds["ANTHROPIC_API_KEY"] = anthropic_key
            self._write_llm_credentials(safe_key, creds)

            self.save_llm_settings()

            self.llm_profile_name_input = ""
            self.selected_llm_profile = name
            self.load_available_llm_profiles()
            self.llm_profile_status = f"✅ Profile saved: {name}"
        except Exception as e:
            self.llm_profile_status = f"❌ Error saving profile: {str(e)}"

    def load_llm_profile(self, profile_display_name: str):
        """Load a saved profile by display name and activate it in .env."""
        _PROVIDER_MAP = {"gemini": "Google Gemini", "bedrock": "AWS Bedrock", "openai": "OpenAI", "copilot": "GitHub Copilot", "anthropic": "Anthropic"}
        _AUTH_MAP = {"access_keys": "Access Keys", "sso_profile": "SSO Profile"}

        if not profile_display_name:
            return
        try:
            target_path = None
            safe_key = None
            if LLM_PROFILES_DIR.exists():
                for f in LLM_PROFILES_DIR.glob("llm_profile.*.json"):
                    try:
                        data = json.loads(f.read_text())
                        if data.get("name") == profile_display_name:
                            target_path = f
                            safe_key = f.stem.replace("llm_profile.", "")
                            break
                    except Exception:
                        continue
            if not target_path:
                self.llm_profile_status = f"❌ Profile not found: {profile_display_name}"
                return

            data = json.loads(target_path.read_text())
            provider = data.get("provider", "gemini")
            self.llm_provider = _PROVIDER_MAP.get(provider, "Google Gemini")
            auth = data.get("aws_auth_method", "access_keys")
            self.llm_aws_auth_method = _AUTH_MAP.get(auth, "Access Keys")
            self.llm_gemini_model_id = data.get("gemini_model_id") or "gemini-2.5-flash"
            self.llm_aws_profile = data.get("aws_profile") or ""
            self.llm_aws_region = data.get("aws_region") or "us-east-1"
            self.llm_aws_bedrock_model_id = data.get("aws_bedrock_model_id") or "anthropic.claude-3-5-sonnet-20241022-v2:0"
            self.llm_openai_model_id = data.get("openai_model_id") or "gpt-4o"
            self.llm_copilot_model_id = data.get("copilot_model_id") or "openai/gpt-4o"
            self.llm_anthropic_model_id = data.get("anthropic_model_id") or "claude-sonnet-4-5"
            _OAUTH_LABEL = {"api_key": "API Key", "oauth": "Sign in"}
            openai_auth = data.get("openai_auth_method", "api_key")
            copilot_auth = data.get("copilot_auth_method", "api_key")
            anthropic_auth = data.get("anthropic_auth_method", "api_key")
            self.llm_openai_auth_method = _OAUTH_LABEL.get(openai_auth, "API Key")
            self.llm_copilot_auth_method = _OAUTH_LABEL.get(copilot_auth, "API Key")
            self.llm_anthropic_auth_method = _OAUTH_LABEL.get(anthropic_auth, "API Key")
            self.refresh_oauth_status()

            creds = self._read_llm_credentials(safe_key)

            env_updates: dict = {"LLM_PROVIDER": provider}
            if provider == "gemini":
                if data.get("gemini_model_id"):
                    env_updates["GEMINI_MODEL_ID"] = data["gemini_model_id"]
            elif provider == "bedrock":
                env_updates["AWS_AUTH_METHOD"] = auth
                if data.get("aws_profile"):
                    env_updates["AWS_PROFILE"] = data["aws_profile"]
                if data.get("aws_region"):
                    env_updates["AWS_REGION"] = data["aws_region"]
                if data.get("aws_bedrock_model_id"):
                    env_updates["AWS_BEDROCK_MODEL_ID"] = data["aws_bedrock_model_id"]
            elif provider == "openai":
                env_updates["OPENAI_AUTH_METHOD"] = openai_auth
                if data.get("openai_model_id"):
                    env_updates["OPENAI_MODEL_ID"] = data["openai_model_id"]
            elif provider == "copilot":
                env_updates["COPILOT_AUTH_METHOD"] = copilot_auth
                if data.get("copilot_model_id"):
                    env_updates["GITHUB_COPILOT_MODEL_ID"] = data["copilot_model_id"]
            elif provider == "anthropic":
                env_updates["ANTHROPIC_AUTH_METHOD"] = anthropic_auth
                if data.get("anthropic_model_id"):
                    env_updates["ANTHROPIC_MODEL_ID"] = data["anthropic_model_id"]

            for k, v in creds.items():
                if v:
                    env_updates[k] = v

            self._write_env_keys(env_updates)
            for k, v in env_updates.items():
                os.environ[k] = v

            self.llm_gemini_api_key = ""
            self.llm_aws_access_key_id = ""
            self.llm_aws_secret_access_key = ""
            self.llm_openai_api_key = ""
            self.llm_copilot_api_key = ""
            self.llm_anthropic_api_key = ""
            self.llm_gemini_key_saved = bool(creds.get("GEMINI_API_KEY", ""))
            self.llm_aws_access_key_saved = bool(creds.get("AWS_ACCESS_KEY_ID", ""))
            self.llm_openai_key_saved = bool(creds.get("OPENAI_API_KEY", ""))
            self.llm_copilot_key_saved = bool(creds.get("GITHUB_COPILOT_TOKEN", ""))
            self.llm_anthropic_key_saved = bool(creds.get("ANTHROPIC_API_KEY", ""))

            self.selected_llm_profile = profile_display_name
            self.llm_profile_status = f"✅ Activated: {profile_display_name}"
            self.llm_settings_status = ""
        except Exception as e:
            self.llm_profile_status = f"❌ Error loading profile: {str(e)}"

    def delete_llm_profile(self):
        """Delete the currently selected LLM profile."""
        if not self.selected_llm_profile:
            return
        try:
            if LLM_PROFILES_DIR.exists():
                for f in LLM_PROFILES_DIR.glob("llm_profile.*.json"):
                    try:
                        data = json.loads(f.read_text())
                        if data.get("name") == self.selected_llm_profile:
                            safe_key = f.stem.replace("llm_profile.", "")
                            f.unlink()
                            self._delete_llm_credentials(safe_key)
                            break
                    except Exception:
                        continue
            deleted_name = self.selected_llm_profile
            self.selected_llm_profile = ""
            self.load_available_llm_profiles()
            self.llm_profile_status = f"✅ Deleted: {deleted_name}"
        except Exception as e:
            self.llm_profile_status = f"❌ Error deleting profile: {str(e)}"

    def set_llm_profile_name_input(self, value: str):
        self.llm_profile_name_input = value

    # --- API Server Management ---

    def load_api_clients(self):
        """Load registered OAuth clients from api_clients.json."""
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from api.auth import list_clients
            self.api_clients = [
                ApiClientInfo(
                    client_id=c["client_id"],
                    description=c.get("description", ""),
                    created_at=c.get("created_at", ""),
                )
                for c in list_clients()
            ]
        except Exception:
            self.api_clients = []

    def register_api_client(self):
        """Register a new OAuth client and display the generated credentials."""
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from api.auth import register_client
            result = register_client(description=self.api_new_client_description.strip())
            self.api_created_client_id = result["client_id"]
            self.api_created_client_secret = result["client_secret"]
            self.api_new_client_description = ""
            self.api_client_status = "✅ Client registered — copy the secret now, it cannot be shown again"
            self.load_api_clients()
        except Exception as e:
            self.api_client_status = f"❌ Error registering client: {e}"
            self.api_created_client_id = ""
            self.api_created_client_secret = ""

    def delete_api_client(self, client_id: str):
        """Delete an OAuth client by client_id."""
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from api.auth import delete_client
            if delete_client(client_id):
                self.api_client_status = f"✅ Deleted client: {client_id}"
            else:
                self.api_client_status = f"❌ Client not found: {client_id}"
            self.load_api_clients()
        except Exception as e:
            self.api_client_status = f"❌ Error deleting client: {e}"

    def dismiss_client_credentials(self):
        """Clear the displayed credentials after the user has copied them."""
        self.api_created_client_id = ""
        self.api_created_client_secret = ""
        self.api_client_status = ""

    def download_client_credentials(self):
        """Download the newly created client credentials as a text file."""
        content = (
            f"# DeepEval Pega API Client Credentials\n"
            f"# Generated: {datetime.now().isoformat()}\n\n"
            f"Client ID: {self.api_created_client_id}\n"
            f"Client Secret: {self.api_created_client_secret}\n"
        )
        filename = f"{self.api_created_client_id}.txt"
        return rx.download(data=content, filename=filename)

    def set_api_new_client_description(self, value: str):
        self.api_new_client_description = value

    def set_api_server_port(self, value: str):
        self.api_server_port = value

    def _find_api_server_pid(self) -> int:
        """Find the PID of a running API server on the configured port."""
        port = self.api_server_port.strip() or "8100"
        try:
            result = subprocess.run(
                ["lsof", "-ti", f"tcp:{port}"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip().splitlines()[0])
        except Exception:
            pass
        return 0

    def _check_api_server_status(self):
        """Check if the API server process is still running."""
        if self.api_server_pid:
            try:
                os.kill(self.api_server_pid, 0)
                self.api_server_running = True
                return
            except OSError:
                self.api_server_pid = 0
        found_pid = self._find_api_server_pid()
        if found_pid:
            self.api_server_pid = found_pid
            self.api_server_running = True
        else:
            self.api_server_running = False
            self.api_server_pid = 0
            self.api_server_status = ""

    def start_api_server(self):
        """Start the FastAPI server as a background subprocess."""
        self._check_api_server_status()
        if self.api_server_running:
            self.api_server_status = f"Server is already running (PID {self.api_server_pid})"
            return
        try:
            port = self.api_server_port.strip() or "8100"
            proc = subprocess.Popen(
                [
                    "python", "-m", "uvicorn",
                    "api.app:create_app", "--factory",
                    "--host", "0.0.0.0",
                    "--port", port,
                ],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.api_server_pid = proc.pid
            self.api_server_running = True
            self.api_server_status = f"✅ API server started on port {port} (PID {proc.pid})"
        except Exception as e:
            self.api_server_status = f"❌ Failed to start server: {e}"

    def stop_api_server(self):
        """Stop the API server subprocess."""
        self._check_api_server_status()
        if not self.api_server_pid:
            self.api_server_running = False
            self.api_server_status = "No server running"
            return
        try:
            os.kill(self.api_server_pid, 15)  # SIGTERM
            self.api_server_running = False
            self.api_server_status = f"✅ API server stopped (PID {self.api_server_pid})"
            self.api_server_pid = 0
        except OSError:
            self.api_server_running = False
            self.api_server_pid = 0
            self.api_server_status = "Server process already terminated"

    def _write_env_keys(self, updates: dict):
        """Update specific keys in the .env file, preserving all other lines."""
        env_path = PROJECT_ROOT / ".env"
        lines = env_path.read_text().splitlines(keepends=True) if env_path.exists() else []

        updated_keys: set = set()
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key = stripped.split("=", 1)[0].strip()
                if key in updates:
                    new_lines.append(f"{key}={updates[key]}\n")
                    updated_keys.add(key)
                    continue
            new_lines.append(line)

        # Append keys not already present in the file
        remaining = {k: v for k, v in updates.items() if k not in updated_keys}
        if remaining:
            if new_lines and new_lines[-1].strip():
                new_lines.append("\n")
            for key, value in remaining.items():
                new_lines.append(f"{key}={value}\n")

        env_path.write_text("".join(new_lines))

    async def test_llm_connection(self):
        """Make a minimal API call with the current form credentials to verify they work.

        Credentials are resolved from the state buffer if the user just typed a new
        value, otherwise fall back to the saved value in .env — so the user can test
        without re-entering a key they already saved.
        """
        # Resolve credentials server-side without storing them back in state
        env_data = self._read_env_creds()
        gemini_key = self.llm_gemini_api_key.strip() or env_data.get("GEMINI_API_KEY", "")
        aws_key_id = self.llm_aws_access_key_id.strip() or env_data.get("AWS_ACCESS_KEY_ID", "")
        aws_secret = self.llm_aws_secret_access_key.strip() or env_data.get("AWS_SECRET_ACCESS_KEY", "")
        openai_key = self.llm_openai_api_key.strip() or env_data.get("OPENAI_API_KEY", "")
        copilot_token = self.llm_copilot_api_key.strip() or env_data.get("GITHUB_COPILOT_TOKEN", "")
        # API tokens are always ASCII; strip any stray non-ASCII from paste artifacts
        copilot_token = copilot_token.encode("ascii", errors="ignore").decode("ascii").strip()
        anthropic_key = self.llm_anthropic_api_key.strip() or env_data.get("ANTHROPIC_API_KEY", "")

        # Validate required fields before attempting
        if self.llm_provider == "Google Gemini":
            if not gemini_key:
                self.llm_test_status = "❌ Enter a Gemini API key first"
                yield
                return
        elif self.llm_provider == "AWS Bedrock":
            if self.llm_aws_auth_method == "SSO Profile":
                if not self.llm_aws_profile.strip():
                    self.llm_test_status = "❌ Enter an SSO profile name first"
                    yield
                    return
            else:
                if not aws_key_id or not aws_secret:
                    self.llm_test_status = "❌ Enter AWS Access Key ID and Secret Access Key first"
                    yield
                    return
        elif self.llm_provider == "OpenAI":
            if self.llm_openai_auth_method == "Sign in":
                if not self.llm_openai_signed_in:
                    self.llm_test_status = "❌ Sign in with ChatGPT first"
                    yield
                    return
            elif not openai_key:
                self.llm_test_status = "❌ Enter an OpenAI API key first"
                yield
                return
        elif self.llm_provider == "GitHub Copilot":
            if self.llm_copilot_auth_method == "Sign in":
                if not self.llm_copilot_signed_in:
                    self.llm_test_status = "❌ Sign in with GitHub first"
                    yield
                    return
            elif not copilot_token:
                self.llm_test_status = "❌ Enter a GitHub token first"
                yield
                return
        elif self.llm_provider == "Anthropic":
            if self.llm_anthropic_auth_method == "Sign in":
                if not self.llm_anthropic_signed_in:
                    self.llm_test_status = "❌ Sign in with Claude first"
                    yield
                    return
            elif not anthropic_key:
                self.llm_test_status = "❌ Enter an Anthropic API key first"
                yield
                return

        self.llm_testing = True
        self.llm_test_status = "Testing connection..."
        yield

        try:
            loop = asyncio.get_event_loop()
            if self.llm_provider == "Google Gemini":
                result = await loop.run_in_executor(
                    None, lambda: self._test_gemini(gemini_key)
                )
            elif self.llm_provider == "AWS Bedrock":
                result = await loop.run_in_executor(
                    None, lambda: self._test_bedrock(aws_key_id, aws_secret)
                )
            elif self.llm_provider == "GitHub Copilot":
                if self.llm_copilot_auth_method == "Sign in":
                    result = await loop.run_in_executor(None, self._test_copilot_oauth)
                else:
                    result = await loop.run_in_executor(
                        None, lambda: self._test_copilot(copilot_token)
                    )
            elif self.llm_provider == "Anthropic":
                if self.llm_anthropic_auth_method == "Sign in":
                    result = await loop.run_in_executor(None, self._test_anthropic_oauth)
                else:
                    result = await loop.run_in_executor(
                        None, lambda: self._test_anthropic(anthropic_key)
                    )
            else:
                if self.llm_openai_auth_method == "Sign in":
                    result = await loop.run_in_executor(None, self._test_openai_oauth)
                else:
                    result = await loop.run_in_executor(
                        None, lambda: self._test_openai(openai_key)
                    )
            self.llm_test_status = result
        except Exception as e:
            self.llm_test_status = f"❌ Connection failed: {e}"
        finally:
            self.llm_testing = False
        yield

    def _test_gemini(self, api_key: str) -> str:
        """Synchronous Gemini connectivity check (run in executor)."""
        from google import genai as _genai
        model = self.llm_gemini_model_id.strip() or "gemini-2.5-flash"
        client = _genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model,
            contents="Reply with just the word OK.",
        )
        preview = (resp.text or "").strip()[:60]
        return f"✅ Gemini connection successful ({model}) — model replied: {preview}"

    def _test_bedrock(self, aws_key_id: str, aws_secret: str) -> str:
        """Synchronous Bedrock connectivity check (run in executor)."""
        import boto3 as _boto3
        model_id = self.llm_aws_bedrock_model_id.strip() or "anthropic.claude-3-5-sonnet-20241022-v2:0"
        region = self.llm_aws_region.strip() or "us-east-1"
        if self.llm_aws_auth_method == "SSO Profile":
            session = _boto3.Session(profile_name=self.llm_aws_profile.strip())
            client = session.client(service_name="bedrock-runtime", region_name=region)
        else:
            client = _boto3.client(
                service_name="bedrock-runtime",
                region_name=region,
                aws_access_key_id=aws_key_id,
                aws_secret_access_key=aws_secret,
            )
        if "anthropic" in model_id:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Reply with just the word OK."}],
            })
            resp_body = json.loads(client.invoke_model(
                modelId=model_id, body=body,
                contentType="application/json", accept="application/json",
            )["body"].read())
            text = resp_body.get("content", [{}])[0].get("text", "")
        elif "amazon" in model_id:
            body = json.dumps({
                "inputText": "Reply with just the word OK.",
                "textGenerationConfig": {"maxTokenCount": 10},
            })
            resp_body = json.loads(client.invoke_model(
                modelId=model_id, body=body,
                contentType="application/json", accept="application/json",
            )["body"].read())
            text = resp_body.get("results", [{}])[0].get("outputText", "")
        elif "meta" in model_id:
            body = json.dumps({
                "prompt": (
                    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
                    "Reply with just the word OK."
                    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                ),
                "max_gen_len": 10,
            })
            resp_body = json.loads(client.invoke_model(
                modelId=model_id, body=body,
                contentType="application/json", accept="application/json",
            )["body"].read())
            text = resp_body.get("generation", "")
        else:
            # Fallback: try Anthropic-style for unknown prefixes
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Reply with just the word OK."}],
            })
            resp_body = json.loads(client.invoke_model(
                modelId=model_id, body=body,
                contentType="application/json", accept="application/json",
            )["body"].read())
            text = resp_body.get("content", [{}])[0].get("text", "")
        preview = text.strip()[:60]
        return f"✅ Bedrock connection successful ({model_id}) — model replied: {preview}"

    def _test_openai(self, api_key: str) -> str:
        """Synchronous OpenAI connectivity check (run in executor)."""
        from openai import OpenAI
        model = self.llm_openai_model_id.strip() or "gpt-4o"
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with just the word OK."}],
            max_tokens=10,
        )
        preview = (response.choices[0].message.content or "").strip()[:60]
        return f"✅ OpenAI connection successful ({model}) — model replied: {preview}"

    def _test_copilot(self, token: str) -> str:
        """Synchronous GitHub Copilot connectivity check (run in executor)."""
        from openai import OpenAI
        model = (self.llm_copilot_model_id.strip() or "openai/gpt-4o").lstrip("/")
        client = OpenAI(
            api_key=token,
            base_url="https://models.github.ai/inference",
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with just the word OK."}],
            max_completion_tokens=10,
        )
        preview = (response.choices[0].message.content or "").strip()[:60]
        return f"✅ GitHub Copilot connection successful ({model}) — model replied: {preview}"

    def _test_anthropic(self, api_key: str) -> str:
        """Synchronous Anthropic connectivity check (run in executor)."""
        from anthropic import Anthropic
        model = self.llm_anthropic_model_id.strip() or "claude-sonnet-4-5"
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=10,
            messages=[{"role": "user", "content": "Reply with just the word OK."}],
        )
        preview = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()[:60]
        return f"✅ Anthropic connection successful ({model}) — model replied: {preview}"

    def _test_openai_oauth(self) -> str:
        """Connectivity check for the ChatGPT subscription (OAuth) path."""
        model = self.llm_openai_model_id.strip() or "gpt-5"
        text = llm_oauth.openai_chatgpt_generate(
            model, "You are a connectivity test.", "Reply with just the word OK."
        )
        return f"✅ ChatGPT (OAuth) connection successful ({model}) — model replied: {text.strip()[:60]}"

    def _test_copilot_oauth(self) -> str:
        """Connectivity check for the Copilot subscription (OAuth) path."""
        from openai import OpenAI
        model = (self.llm_copilot_model_id.strip() or "gpt-4o").lstrip("/")
        token = llm_oauth.get_copilot_token()
        client = OpenAI(
            api_key=token,
            base_url=llm_oauth.get_copilot_api_base(),
            default_headers=llm_oauth.copilot_request_headers(),
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with just the word OK."}],
            max_tokens=10,
        )
        preview = (response.choices[0].message.content or "").strip()[:60]
        return f"✅ GitHub Copilot (OAuth) connection successful ({model}) — model replied: {preview}"

    def _test_anthropic_oauth(self) -> str:
        """Connectivity check for the Claude subscription (OAuth) path."""
        from anthropic import Anthropic
        model = self.llm_anthropic_model_id.strip() or "claude-sonnet-4-5-20250929"
        token = llm_oauth.get_anthropic_token()
        client = Anthropic(
            auth_token=token,
            default_headers={"anthropic-beta": llm_oauth.ANTHROPIC_OAUTH_BETA},
        )
        response = client.messages.create(
            model=model,
            max_tokens=10,
            system=[
                {"type": "text", "text": llm_oauth.ANTHROPIC_OAUTH_SYSTEM_PREFIX},
                {"type": "text", "text": "You are a connectivity test."},
            ],
            messages=[{"role": "user", "content": "Reply with just the word OK."}],
        )
        preview = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()[:60]
        return f"✅ Anthropic (OAuth) connection successful ({model}) — model replied: {preview}"

    # --- Model Listing Sync Helpers ---

    def _fetch_gemini_models_sync(self, api_key: str) -> list:
        from google import genai as _genai
        client = _genai.Client(api_key=api_key)
        models = []
        for model in client.models.list():
            name = model.name
            short = name.replace("models/", "") if name.startswith("models/") else name
            if "gemini" in short.lower():
                models.append(short)
        return sorted(models)

    def _fetch_bedrock_models_sync(self, aws_key_id: str, aws_secret: str) -> list:
        import boto3 as _boto3
        region = self.llm_aws_region.strip() or "us-east-1"
        if self.llm_aws_auth_method == "SSO Profile":
            session = _boto3.Session(profile_name=self.llm_aws_profile.strip())
            client = session.client(service_name="bedrock", region_name=region)
        else:
            client = _boto3.client(
                service_name="bedrock",
                region_name=region,
                aws_access_key_id=aws_key_id,
                aws_secret_access_key=aws_secret,
            )
        models = set()
        # Inference profiles (required for newer models like Claude Opus 4)
        try:
            paginator = client.get_paginator("list_inference_profiles")
            for page in paginator.paginate():
                for profile in page.get("inferenceProfileSummaries", []):
                    profile_id = profile.get("inferenceProfileId", "")
                    profile_type = profile.get("type", "")
                    if profile_id and profile_type == "SYSTEM_DEFINED":
                        models.add(profile_id)
        except Exception:
            pass
        # Foundation models (for older models that support direct invocation)
        try:
            response = client.list_foundation_models()
            for summary in response.get("modelSummaries", []):
                model_id = summary.get("modelId", "")
                output_modalities = summary.get("outputModalities", [])
                if "TEXT" in output_modalities:
                    models.add(model_id)
        except Exception:
            pass
        return sorted(models)

    def _fetch_openai_models_sync(self, api_key: str) -> list:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        models = []
        for model in client.models.list():
            model_id = model.id
            if any(model_id.startswith(p) for p in ("gpt-", "o1-", "o3-", "o4-", "chatgpt-")):
                models.append(model_id)
        return sorted(models)

    def _fetch_copilot_models_sync(self, api_key: str) -> list:
        import requests as _requests
        try:
            resp = _requests.get(
                "https://models.github.ai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                models = []
                for m in data.get("data", []):
                    model_id = m.get("id", "")
                    if model_id:
                        models.append(model_id)
                if models:
                    return sorted(models)
        except Exception:
            pass
        # Fallback: curated list using publisher/model-name format
        return [
            "deepseek/deepseek-r1",
            "meta/llama-4-scout-17b-16e-instruct",
            "meta/meta-llama-3.1-405b-instruct",
            "microsoft/phi-4",
            "mistral-ai/mistral-small-2503",
            "openai/gpt-4.1",
            "openai/gpt-4.1-mini",
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "openai/o3-mini",
            "openai/o4-mini",
        ]

    def _fetch_anthropic_models_sync(self, api_key: str) -> list:
        """Fetch available Anthropic models from the /v1/models endpoint."""
        import requests as _requests
        try:
            resp = _requests.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                params={"limit": 100},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("id", "") for m in data.get("data", []) if m.get("id")]
                if models:
                    return sorted(models)
        except Exception:
            pass
        # Fallback: curated list of common Anthropic model IDs
        return [
            "claude-3-5-haiku-latest",
            "claude-3-5-sonnet-latest",
            "claude-3-7-sonnet-latest",
            "claude-opus-4-1",
            "claude-opus-4-5",
            "claude-sonnet-4-5",
        ]

    # --- Model Listing Async Handlers ---

    async def load_gemini_models(self):
        env_data = self._read_env_creds()
        api_key = self.llm_gemini_api_key.strip() or env_data.get("GEMINI_API_KEY", "")
        if not api_key:
            self.llm_gemini_models_error = "Enter a Gemini API key first"
            yield
            return
        self.llm_gemini_models_loading = True
        self.llm_gemini_models_error = ""
        yield
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self._fetch_gemini_models_sync(api_key)
            )
            self.llm_gemini_models = result
            if not result:
                self.llm_gemini_models_error = "No models found"
        except Exception as e:
            self.llm_gemini_models_error = f"Failed to load models: {e}"
        finally:
            self.llm_gemini_models_loading = False
        yield

    async def load_bedrock_models(self):
        env_data = self._read_env_creds()
        aws_key_id = self.llm_aws_access_key_id.strip() or env_data.get("AWS_ACCESS_KEY_ID", "")
        aws_secret = self.llm_aws_secret_access_key.strip() or env_data.get("AWS_SECRET_ACCESS_KEY", "")
        if self.llm_aws_auth_method != "SSO Profile" and (not aws_key_id or not aws_secret):
            self.llm_bedrock_models_error = "Enter AWS credentials first"
            yield
            return
        self.llm_bedrock_models_loading = True
        self.llm_bedrock_models_error = ""
        yield
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self._fetch_bedrock_models_sync(aws_key_id, aws_secret)
            )
            self.llm_bedrock_models = result
            if not result:
                self.llm_bedrock_models_error = "No models found"
        except Exception as e:
            self.llm_bedrock_models_error = f"Failed to load models: {e}"
        finally:
            self.llm_bedrock_models_loading = False
        yield

    async def load_openai_models(self):
        if self.llm_openai_auth_method == "Sign in":
            self.llm_openai_models = list(llm_oauth.OPENAI_OAUTH_MODELS)
            self.llm_openai_models_error = ""
            yield
            return
        env_data = self._read_env_creds()
        api_key = self.llm_openai_api_key.strip() or env_data.get("OPENAI_API_KEY", "")
        if not api_key:
            self.llm_openai_models_error = "Enter an OpenAI API key first"
            yield
            return
        self.llm_openai_models_loading = True
        self.llm_openai_models_error = ""
        yield
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self._fetch_openai_models_sync(api_key)
            )
            self.llm_openai_models = result
            if not result:
                self.llm_openai_models_error = "No models found"
        except Exception as e:
            self.llm_openai_models_error = f"Failed to load models: {e}"
        finally:
            self.llm_openai_models_loading = False
        yield

    async def load_copilot_models(self):
        if self.llm_copilot_auth_method == "Sign in":
            if not llm_oauth.is_signed_in("copilot"):
                self.llm_copilot_models_error = "Sign in with GitHub first"
                yield
                return
            self.llm_copilot_models_loading = True
            self.llm_copilot_models_error = ""
            yield
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, llm_oauth.copilot_list_models)
                self.llm_copilot_models = result
                if not result:
                    self.llm_copilot_models_error = "No models found"
            except Exception as e:
                self.llm_copilot_models_error = f"Failed to load models: {e}"
            finally:
                self.llm_copilot_models_loading = False
            yield
            return
        env_data = self._read_env_creds()
        api_key = self.llm_copilot_api_key.strip() or env_data.get("GITHUB_COPILOT_TOKEN", "")
        api_key = api_key.encode("ascii", errors="ignore").decode("ascii").strip()
        if not api_key:
            self.llm_copilot_models_error = "Enter a GitHub token first"
            yield
            return
        self.llm_copilot_models_loading = True
        self.llm_copilot_models_error = ""
        yield
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self._fetch_copilot_models_sync(api_key)
            )
            self.llm_copilot_models = result
            if not result:
                self.llm_copilot_models_error = "No models found"
        except Exception as e:
            self.llm_copilot_models_error = f"Failed to load models: {e}"
        finally:
            self.llm_copilot_models_loading = False
        yield

    async def load_anthropic_models(self):
        if self.llm_anthropic_auth_method == "Sign in":
            if not llm_oauth.is_signed_in("anthropic"):
                self.llm_anthropic_models_error = "Sign in with Claude first"
                yield
                return
            self.llm_anthropic_models_loading = True
            self.llm_anthropic_models_error = ""
            yield
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, llm_oauth.anthropic_list_models)
                self.llm_anthropic_models = result
                if not result:
                    self.llm_anthropic_models_error = "No models found"
            except Exception as e:
                self.llm_anthropic_models_error = f"Failed to load models: {e}"
            finally:
                self.llm_anthropic_models_loading = False
            yield
            return
        env_data = self._read_env_creds()
        api_key = self.llm_anthropic_api_key.strip() or env_data.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            self.llm_anthropic_models_error = "Enter an Anthropic API key first"
            yield
            return
        self.llm_anthropic_models_loading = True
        self.llm_anthropic_models_error = ""
        yield
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self._fetch_anthropic_models_sync(api_key)
            )
            self.llm_anthropic_models = result
            if not result:
                self.llm_anthropic_models_error = "No models found"
        except Exception as e:
            self.llm_anthropic_models_error = f"Failed to load models: {e}"
        finally:
            self.llm_anthropic_models_loading = False
        yield


# ============================================================================
# UI Components
# ============================================================================

def metric_card(metric: dict) -> rx.Component:
    """Render a metric selection card."""
    metric_id = metric["id"]
    return rx.box(
        rx.hstack(
            rx.checkbox(
                checked=State.selected_metrics.contains(metric_id),
                on_change=lambda: State.toggle_metric(metric_id),
            ),
            rx.vstack(
                rx.text(metric["name"], weight="bold", size="3"),
                rx.text(metric["description"], size="1", color_scheme="gray"),
                rx.hstack(
                    rx.text("Threshold:", size="1"),
                    rx.input(
                        default_value=str(metric["default_threshold"]),
                        on_change=lambda v: State.update_threshold(metric_id, v),
                        width="60px",
                        size="1",
                    ),
                    spacing="2",
                ),
                align_items="start",
                spacing="1",
            ),
            spacing="3",
            align="start",
        ),
        padding="3",
        border="1px solid var(--gray-6)",
        border_radius="8px",
        _hover={"border_color": "var(--accent-8)"},
    )


def dataset_card(dataset: GoldenDataset) -> rx.Component:
    """Render a dataset selection card."""
    return rx.box(
        rx.hstack(
            rx.cond(
                State.selected_dataset == dataset.filename,
                rx.icon("circle-check", color="var(--accent-9)", size=20),
                rx.icon("circle", color="var(--gray-6)", size=20),
            ),
            rx.vstack(
                rx.text(dataset.name, weight="bold", size="3"),
                rx.hstack(
                    rx.badge(rx.text(dataset.turn_count, " turns"), color_scheme="blue"),
                    rx.badge(rx.text(dataset.tools_count, " tools"), color_scheme="green"),
                    spacing="2",
                ),
                rx.text(
                    "Recorded: ", dataset.recorded_at,
                    size="1",
                    color_scheme="gray",
                ),
                align_items="start",
                spacing="1",
            ),
            rx.spacer(),
            rx.hstack(
                # Rename button
                rx.box(
                    rx.alert_dialog.root(
                        rx.alert_dialog.trigger(
                            rx.icon_button(
                                rx.icon("pencil", size=14),
                                size="1",
                                variant="ghost",
                                color_scheme="blue",
                            ),
                        ),
                        rx.alert_dialog.content(
                            rx.alert_dialog.title("Rename Golden Dataset"),
                            rx.vstack(
                                rx.text("Enter a new name:", size="2"),
                                rx.input(
                                    default_value=dataset.name,
                                    on_change=State.set_rename_dataset_new_name,
                                    width="100%",
                                ),
                                spacing="2",
                                width="100%",
                            ),
                            rx.hstack(
                                rx.alert_dialog.cancel(
                                    rx.button("Cancel", variant="soft", color_scheme="gray"),
                                ),
                                rx.alert_dialog.action(
                                    rx.button(
                                        "Save",
                                        color_scheme="blue",
                                        on_click=lambda: State.rename_dataset(
                                            dataset.filename,
                                            State.rename_dataset_new_name,
                                        ),
                                    ),
                                ),
                                justify="end",
                                spacing="2",
                                margin_top="4",
                            ),
                        ),
                    ),
                    on_click=rx.stop_propagation,
                ),
                # Replace data button
                rx.box(
                    rx.icon_button(
                        rx.icon("replace", size=14),
                        size="1",
                        variant="ghost",
                        color_scheme="orange",
                        on_click=lambda: State.start_replace_dataset(dataset.filename),
                    ),
                    on_click=rx.stop_propagation,
                ),
                # Delete button
                rx.box(
                    rx.alert_dialog.root(
                        rx.alert_dialog.trigger(
                            rx.icon_button(
                                rx.icon("trash-2", size=14),
                                size="1",
                                variant="ghost",
                                color_scheme="red",
                            ),
                        ),
                        rx.alert_dialog.content(
                            rx.alert_dialog.title("Delete Golden Dataset"),
                            rx.alert_dialog.description(
                                rx.text("Are you sure you want to delete ", rx.text(dataset.name, weight="bold", as_="span"), "? This cannot be undone."),
                                size="2",
                            ),
                            rx.hstack(
                                rx.alert_dialog.cancel(
                                    rx.button("Cancel", variant="soft", color_scheme="gray"),
                                ),
                                rx.alert_dialog.action(
                                    rx.button(
                                        rx.icon("trash-2", size=14),
                                        "Delete",
                                        color_scheme="red",
                                        on_click=lambda: State.delete_dataset(dataset.filename),
                                    ),
                                ),
                                justify="end",
                                spacing="2",
                                margin_top="4",
                            ),
                        ),
                    ),
                    on_click=rx.stop_propagation,
                ),
                spacing="1",
            ),
            spacing="3",
            align="center",
            width="100%",
        ),
        padding="3",
        border="1px solid var(--gray-6)",
        border_radius="8px",
        cursor="pointer",
        on_click=lambda: State.select_dataset(dataset.filename),
        background=rx.cond(
            State.selected_dataset == dataset.filename,
            "var(--accent-3)",
            "transparent"
        ),
        _hover={"border_color": "var(--accent-8)"},
        width="100%",
    )


def evaluation_dataset_card(dataset: GoldenDataset) -> rx.Component:
    """Minimal dataset card for evaluation selection — no edit actions."""
    return rx.box(
        rx.hstack(
            rx.cond(
                State.selected_dataset == dataset.filename,
                rx.icon("circle-check", color="var(--accent-9)", size=20),
                rx.icon("circle", color="var(--gray-6)", size=20),
            ),
            rx.vstack(
                rx.text(dataset.name, weight="bold", size="3"),
                rx.hstack(
                    rx.badge(rx.text(dataset.turn_count, " turns"), color_scheme="blue"),
                    rx.badge(rx.text(dataset.tools_count, " tools"), color_scheme="green"),
                    spacing="2",
                ),
                rx.text(
                    "Recorded: ", dataset.recorded_at,
                    size="1",
                    color_scheme="gray",
                ),
                align_items="start",
                spacing="1",
            ),
            spacing="3",
            align="center",
            width="100%",
        ),
        padding="3",
        border="1px solid var(--gray-6)",
        border_radius="8px",
        cursor="pointer",
        on_click=lambda: State.select_dataset(dataset.filename),
        background=rx.cond(
            State.selected_dataset == dataset.filename,
            "var(--accent-3)",
            "transparent"
        ),
        _hover={"border_color": "var(--accent-8)"},
        width="100%",
    )


def result_card(result: EvaluationResult) -> rx.Component:
    """Render a compact metric result tile (green = passed, red = failed)."""
    return rx.box(
        rx.hstack(
            rx.cond(
                result.passed,
                rx.icon("circle-check", color="var(--green-9)", size=22),
                rx.icon("circle-x", color="var(--red-9)", size=22),
            ),
            rx.vstack(
                rx.text(result.metric_name, weight="bold", size="2"),
                rx.hstack(
                    rx.text("Score: ", rx.text.span(result.score, weight="medium"), size="2"),
                    rx.text("Threshold: ", result.threshold, size="2", color_scheme="gray"),
                    spacing="3",
                ),
                rx.text(result.reason, size="1", color_scheme="gray"),
                align_items="start",
                spacing="1",
            ),
            spacing="3",
            align="start",
        ),
        padding="3",
        border_radius="8px",
        border_top=rx.cond(result.passed, "1px solid var(--green-6)", "1px solid var(--red-6)"),
        border_right=rx.cond(result.passed, "1px solid var(--green-6)", "1px solid var(--red-6)"),
        border_bottom=rx.cond(result.passed, "1px solid var(--green-6)", "1px solid var(--red-6)"),
        border_left=rx.cond(result.passed, "4px solid var(--green-9)", "4px solid var(--red-9)"),
        background=rx.cond(result.passed, "var(--green-2)", "var(--red-2)"),
        width="280px",
        flex_shrink="0",
    )


def evaluation_section() -> rx.Component:
    """Render the evaluation configuration section."""
    return rx.vstack(
        # Header
        rx.heading("Run DeepEval Evaluation", size="6"),
        rx.text(
            "Choose a golden dataset, select metrics, and run your evaluation.",
            color_scheme="gray",
        ),
        rx.divider(),
        
        # Project Configuration Selection
        rx.vstack(
            rx.heading("1. Load Project Configuration", size="4"),
            rx.text("Select a project configuration to use for evaluation:", size="2"),
            rx.hstack(
                rx.select(
                    State.evaluation_configs,
                    placeholder="Select a project config...",
                    value=State.evaluation_project_config,
                    on_change=State.set_evaluation_project_config,
                    width="300px",
                ),
                rx.button(
                    rx.icon("refresh-cw", size=16),
                    "Refresh",
                    on_click=State.load_evaluation_configs,
                    variant="outline",
                    size="2",
                ),
                spacing="2",
            ),
            rx.cond(
                State.evaluation_project_config != "",
                rx.badge(
                    rx.icon("circle-check", size=14),
                    rx.text("Loaded: ", State.evaluation_project_config),
                    color_scheme="green",
                    size="2",
                ),
            ),
            align_items="start",
            width="100%",
            spacing="3",
        ),
        
        # Case ID for Step Agents (conditionally shown)
        rx.cond(
            State.evaluation_agent_type == "step_agent",
            rx.vstack(
                rx.heading("Case ID (Step Agent)", size="4"),
                rx.text(
                    "This agent operates on an existing case. Provide the Pega case ID:",
                    size="2",
                ),
                rx.input(
                    placeholder="e.g., UPLUS-FS-WORK P-168004",
                    value=State.evaluation_case_id,
                    on_change=State.set_evaluation_case_id,
                    width="400px",
                ),
                rx.callout(
                    "Step agents require a live case in the correct workflow state. "
                    "The case ID is passed as contextID when creating the agent conversation.",
                    icon="info",
                    color_scheme="amber",
                ),
                align_items="start",
                width="100%",
                spacing="3",
            ),
        ),

        rx.divider(),

        # Dataset Selection
        rx.vstack(
            rx.heading("2. Select Golden Dataset", size="4"),
            rx.text("Choose a golden dataset to evaluate against:", size="2"),
            rx.button(
                rx.icon("refresh-cw", size=16),
                "Refresh List",
                on_click=State.load_available_datasets,
                size="1",
                variant="outline",
            ),
            rx.cond(
                State.evaluation_project_name != "",
                rx.cond(
                    State.filtered_datasets.length() > 0,
                    rx.vstack(
                        rx.foreach(State.filtered_datasets, evaluation_dataset_card),
                        width="100%",
                        spacing="2",
                    ),
                    rx.callout(
                        "No golden datasets found for this project. Create one in the 'Golden Datasets' tab.",
                        icon="info",
                        color_scheme="blue",
                    ),
                ),
                rx.callout(
                    "Select a project configuration above to see available golden datasets.",
                    icon="info",
                    color_scheme="blue",
                ),
            ),
            rx.cond(
                State.selected_dataset != "",
                rx.button(
                    rx.icon("eye", size=16),
                    "Preview Dataset",
                    on_click=State.preview_selected_dataset,
                    variant="outline",
                    size="2",
                ),
            ),
            align_items="start",
            width="100%",
            spacing="3",
        ),
        
        rx.divider(),
        
        # Metrics Selection
        rx.vstack(
            rx.heading("3. Select Metrics", size="4"),
            rx.text("Choose which DeepEval metrics to include in your evaluation:", size="2"),
            rx.grid(
                *[metric_card(m) for m in AVAILABLE_METRICS],
                columns="2",
                spacing="3",
                width="100%",
            ),
            align_items="start",
            width="100%",
            spacing="3",
        ),
        
        rx.divider(),
        
        # Run Evaluation
        rx.vstack(
            rx.heading("4. Run Evaluation", size="4"),
            rx.hstack(
                rx.button(
                    rx.cond(
                        State.evaluation_running,
                        rx.spinner(size="1"),
                        rx.icon("play", size=16),
                    ),
                    "Run Evaluation",
                    on_click=State.run_evaluation,
                    disabled=State.evaluation_running,
                    color_scheme="green",
                    size="3",
                ),
                rx.text(State.evaluation_status, size="2"),
                spacing="3",
                align="center",
            ),
            align_items="start",
            width="100%",
            spacing="3",
        ),
        
        # Results — hidden while running, hidden before first run, shown only on completion
        rx.cond(
            ~State.evaluation_running & (State.evaluation_results.length() > 0),
            rx.vstack(
                rx.divider(),
                rx.heading("Results", size="4"),
                rx.box(
                    rx.foreach(State.evaluation_results, result_card),
                    display="flex",
                    flex_wrap="wrap",
                    gap="12px",
                    width="100%",
                ),
                width="100%",
                spacing="3",
                align_items="start",
            ),
        ),
        
        # Log Output
        rx.cond(
            State.evaluation_log != "",
            rx.vstack(
                rx.divider(),
                rx.heading("Evaluation Log", size="4"),
                rx.box(
                    rx.code_block(
                        State.evaluation_log,
                        language="bash",
                        show_line_numbers=False,
                    ),
                    max_height="400px",
                    overflow="auto",
                    width="100%",
                ),
                width="100%",
                spacing="3",
                align_items="start",
            ),
        ),
        
        spacing="5",
        align_items="start",
        width="100%",
    )


def capture_mode_section() -> rx.Component:
    """Render the capture from Pega UI section."""
    return rx.vstack(
        rx.text(
            "Capture a golden session from an existing Pega conversation.",
            color_scheme="gray",
        ),
        # Project Configuration Selection
        rx.vstack(
            rx.text("Project Configuration:", weight="medium", size="2"),
            rx.hstack(
                rx.select(
                    State.capture_configs,
                    placeholder="Select a project config...",
                    value=State.capture_project_config,
                    on_change=State.set_capture_project_config,
                    width="100%",
                ),
                rx.button(
                    rx.icon("refresh-cw", size=16),
                    on_click=State.load_capture_configs,
                    variant="outline",
                    size="2",
                ),
                spacing="2",
                width="100%",
            ),
            rx.cond(
                State.capture_project_config != "",
                rx.badge(
                    rx.icon("circle-check", size=14),
                    f"Using: {State.capture_project_config}",
                    color_scheme="green",
                    size="2",
                ),
            ),
            align_items="start",
            width="100%",
        ),
        # Load Conversations button
        rx.hstack(
            rx.button(
                rx.cond(
                    State.capture_conversations_loading,
                    rx.spinner(size="1"),
                    rx.icon("list", size=16),
                ),
                "Load Conversations",
                on_click=State.load_conversations,
                disabled=(State.capture_project_config == "") | State.capture_conversations_loading,
                variant="outline",
                color_scheme="blue",
                size="2",
            ),
            rx.cond(
                State.capture_conversations.length() > 0,
                rx.badge(
                    f"{State.capture_conversations.length()} found",
                    color_scheme="green",
                    size="2",
                ),
            ),
            spacing="2",
            align="center",
        ),
        # Error display
        rx.cond(
            State.capture_conversations_error != "",
            rx.callout(
                State.capture_conversations_error,
                icon="triangle-alert",
                color_scheme="red",
            ),
        ),
        # Conversation dropdown (when conversations loaded)
        rx.cond(
            State.capture_conversations.length() > 0,
            rx.vstack(
                rx.text("Conversation ID:", weight="medium", size="2"),
                rx.select(
                    State.capture_conversations,
                    placeholder="Select a conversation...",
                    on_change=State.select_conversation,
                    width="100%",
                ),
                align_items="start",
                width="100%",
            ),
        ),
        # Conversation details card
        rx.cond(
            State.capture_conv_details_loading,
            rx.hstack(rx.spinner(size="2"), rx.text("Loading details...", size="2"), spacing="2"),
        ),
        rx.cond(
            State.capture_selected_conv_details.contains("turn_count"),
            rx.card(
                rx.vstack(
                    rx.hstack(
                        rx.badge(rx.text(f"Turns: {State.capture_selected_conv_details['turn_count']}"), color_scheme="blue"),
                        rx.badge(rx.text(f"Messages: {State.capture_selected_conv_details['total_messages']}"), color_scheme="purple"),
                        rx.badge(rx.text(f"Status: {State.capture_selected_conv_details['status']}"), color_scheme="green"),
                        spacing="2",
                        wrap="wrap",
                    ),
                    rx.cond(
                        State.capture_selected_conv_details['first_user_msg'] != "",
                        rx.text(
                            State.capture_selected_conv_details['first_user_msg'],
                            size="1",
                            color="gray",
                            trim="both",
                        ),
                    ),
                    spacing="2",
                ),
                width="100%",
            ),
        ),
        # Manual entry fallback
        rx.vstack(
            rx.text(
                rx.cond(
                    State.capture_conversations.length() > 0,
                    "Or enter Conversation ID manually:",
                    "Conversation ID:",
                ),
                weight="medium",
                size="2",
            ),
            rx.input(
                placeholder="PXCONV-12345",
                value=State.capture_conversation_id,
                on_change=State.set_conversation_id,
                width="100%",
            ),
            align_items="start",
            width="100%",
        ),
        rx.vstack(
            rx.text("Session Name (optional):", weight="medium", size="2"),
            rx.input(
                placeholder="e.g. Claims Flow",
                value=State.capture_session_name,
                on_change=State.set_session_name,
                width="100%",
            ),
            align_items="start",
            width="100%",
        ),
        rx.button(
            rx.cond(
                State.capture_running,
                rx.spinner(size="1"),
                rx.icon("download", size=16),
            ),
            "Capture Golden Session",
            on_click=State.capture_golden_session,
            disabled=State.capture_running,
            color_scheme="blue",
            size="3",
        ),
        rx.cond(
            State.capture_status != "",
            rx.box(
                rx.code_block(
                    State.capture_status,
                    language="bash",
                    show_line_numbers=False,
                ),
                width="100%",
                max_height="200px",
                overflow="auto",
            ),
        ),
        spacing="4",
        align_items="start",
        width="100%",
    )


def manual_mode_section() -> rx.Component:
    """Render the manual JSON entry section."""
    return rx.vstack(
        rx.text(
            "Create a golden dataset by entering JSON manually.",
            color_scheme="gray",
        ),
        rx.hstack(
            rx.button(
                rx.icon("file-text", size=16),
                "Load Template",
                on_click=State.load_template_json,
                variant="outline",
                size="2",
            ),
            spacing="2",
        ),
        rx.vstack(
            rx.text("Dataset Name:", weight="medium", size="2"),
            rx.input(
                placeholder="My Golden Dataset",
                value=State.manual_dataset_name,
                on_change=State.set_manual_name,
                width="100%",
            ),
            align_items="start",
            width="100%",
        ),
        rx.vstack(
            rx.text("JSON Content:", weight="medium", size="2"),
            rx.text_area(
                placeholder='{"turns": [...]}',
                value=State.manual_json_content,
                on_change=State.set_manual_json,
                width="100%",
                min_height="300px",
                font_family="monospace",
            ),
            rx.cond(
                State.json_validation_error != "",
                rx.callout(
                    State.json_validation_error,
                    icon="triangle-alert",
                    color_scheme="red",
                ),
            ),
            rx.cond(
                (State.manual_json_content != "") & (State.json_validation_error == ""),
                rx.callout(
                    "JSON is valid ✓",
                    icon="circle-check",
                    color_scheme="green",
                ),
            ),
            align_items="start",
            width="100%",
        ),
        rx.button(
            rx.icon("save", size=16),
            "Save Golden Dataset",
            on_click=State.save_manual_dataset,
            disabled=(State.manual_json_content == "") | (State.json_validation_error != ""),
            color_scheme="green",
            size="3",
        ),
        spacing="4",
        align_items="start",
        width="100%",
    )


def agent_output_section() -> rx.Component:
    """Render the agent output JSON capture section."""
    return rx.vstack(
        rx.text(
            "Capture a golden session from Pega agent output JSON. This extracts tools directly from tool_calls arrays for accurate detection.",
            color_scheme="gray",
        ),
        rx.callout(
            rx.vstack(
                rx.text("How to get agent output JSON:", weight="bold"),
                rx.text("1. Enable tracer or debugging in your Pega agent"),
                rx.text("2. Run a conversation and copy the full JSON response"),
                rx.text("3. The JSON should contain 'conversation_history' with 'tool_calls'"),
                spacing="1",
                align_items="start",
            ),
            icon="info",
            color_scheme="blue",
        ),
        # Project Configuration Selection
        rx.vstack(
            rx.text("Project Configuration:", weight="medium", size="2"),
            rx.hstack(
                rx.select(
                    State.capture_configs,
                    placeholder="Select a project config...",
                    value=State.capture_project_config,
                    on_change=State.set_capture_project_config,
                    width="100%",
                ),
                rx.button(
                    rx.icon("refresh-cw", size=16),
                    on_click=State.load_capture_configs,
                    variant="outline",
                    size="2",
                ),
                spacing="2",
                width="100%",
            ),
            rx.cond(
                State.capture_project_config != "",
                rx.badge(
                    rx.icon("circle-check", size=14),
                    f"Using: {State.capture_project_config}",
                    color_scheme="green",
                    size="2",
                ),
            ),
            align_items="start",
            width="100%",
        ),
        rx.vstack(
            rx.text("Session Name (optional):", weight="medium", size="2"),
            rx.input(
                placeholder="e.g., Sara Connor Complaint",
                value=State.agent_output_name,
                on_change=State.set_agent_output_name,
                width="100%",
            ),
            align_items="start",
            width="100%",
        ),
        rx.vstack(
            rx.text("Agent Output JSON:", weight="medium", size="2"),
            rx.text_area(
                placeholder='Paste the full Pega agent output JSON here...\n\n{"name": "AgentName", "conversation_history": [...], "plugins": [...]}',
                value=State.agent_output_json,
                on_change=State.set_agent_output_json,
                width="100%",
                min_height="300px",
                font_family="monospace",
            ),
            rx.cond(
                State.agent_output_error != "",
                rx.callout(
                    State.agent_output_error,
                    icon="triangle-alert",
                    color_scheme="red",
                ),
            ),
            rx.cond(
                (State.agent_output_json != "") & (State.agent_output_error == ""),
                rx.callout(
                    "Agent output JSON is valid ✓",
                    icon="circle-check",
                    color_scheme="green",
                ),
            ),
            align_items="start",
            width="100%",
        ),
        rx.button(
            rx.cond(
                State.capture_running,
                rx.spinner(size="1"),
                rx.icon("wand-sparkles", size=16),
            ),
            "Capture from Agent Output",
            on_click=State.capture_from_agent_output,
            disabled=State.capture_running | (State.agent_output_json == "") | (State.agent_output_error != ""),
            color_scheme="green",
            size="3",
        ),
        rx.cond(
            State.capture_status != "",
            rx.box(
                rx.code_block(
                    State.capture_status,
                    language="bash",
                    show_line_numbers=False,
                ),
                width="100%",
                max_height="200px",
                overflow="auto",
            ),
        ),
        spacing="4",
        align_items="start",
        width="100%",
    )


def upload_section() -> rx.Component:
    """Render the file upload section."""
    return rx.vstack(
        rx.text(
            "Upload an existing golden dataset JSON file.",
            color_scheme="gray",
        ),
        rx.upload(
            rx.vstack(
                rx.icon("upload", size=32, color="var(--gray-9)"),
                rx.text("Drag and drop or click to upload", size="2"),
                rx.text("Accepts .json files", size="1", color_scheme="gray"),
                spacing="2",
                align="center",
            ),
            id="golden_upload",
            accept={".json": ["application/json"]},
            max_files=5,
            border="2px dashed var(--gray-6)",
            border_radius="8px",
            padding="6",
            width="100%",
            _hover={"border_color": "var(--accent-8)"},
        ),
        rx.button(
            rx.icon("upload", size=16),
            "Upload Files",
            on_click=lambda: State.handle_upload(rx.upload_files(upload_id="golden_upload")),
            size="2",
        ),
        rx.cond(
            State.capture_status != "",
            rx.text(State.capture_status, size="2"),
        ),
        spacing="4",
        align_items="start",
        width="100%",
    )


def golden_dataset_section() -> rx.Component:
    """Render the golden dataset management section with full CRUD."""
    return rx.vstack(
        # Header
        rx.heading("Manage Golden Datasets", size="6"),
        rx.text(
            "Browse, create, replace, and delete golden datasets for evaluation testing.",
            color_scheme="gray",
        ),
        rx.divider(),

        # --- Section A: Existing Datasets ---
        rx.vstack(
            rx.hstack(
                rx.heading("Existing Datasets", size="4"),
                rx.spacer(),
                rx.hstack(
                    rx.select(
                        State.datasets_filter_options,
                        placeholder="Filter by project...",
                        on_change=State.set_datasets_filter_project,
                        size="1",
                        width="220px",
                    ),
                    rx.button(
                        rx.icon("refresh-cw", size=14),
                        on_click=State.load_available_datasets,
                        variant="ghost",
                        size="1",
                    ),
                    spacing="2",
                    align="center",
                ),
                width="100%",
                align="center",
            ),
            rx.cond(
                State.managed_datasets.length() > 0,
                rx.vstack(
                    rx.foreach(State.managed_datasets, dataset_card),
                    width="100%",
                    spacing="2",
                ),
                rx.callout(
                    "No datasets found. Create one below.",
                    icon="info",
                    color_scheme="gray",
                    size="1",
                ),
            ),
            align_items="start",
            width="100%",
            spacing="3",
        ),

        rx.divider(),

        # --- Section B: Create / Replace Dataset ---
        # Replace mode banner
        rx.cond(
            State.replace_dataset_filename != "",
            rx.callout(
                rx.hstack(
                    rx.text("Replacing dataset: ", rx.text.span(State.replace_dataset_filename, weight="bold")),
                    rx.spacer(),
                    rx.button(
                        "Cancel",
                        on_click=State.cancel_replace_dataset,
                        variant="soft",
                        color_scheme="gray",
                        size="1",
                    ),
                    width="100%",
                    align="center",
                ),
                icon="replace",
                color_scheme="orange",
            ),
        ),

        # Mode Selection
        rx.vstack(
            rx.heading(
                rx.cond(
                    State.replace_dataset_filename != "",
                    "Replace Using",
                    "Create New Dataset",
                ),
                size="4",
            ),
            rx.hstack(
                rx.button(
                    rx.icon("download", size=16),
                    "Capture from Pega",
                    on_click=lambda: State.set_creation_mode("capture"),
                    variant=rx.cond(State.dataset_creation_mode == "capture", "solid", "outline"),
                    color_scheme=rx.cond(State.dataset_creation_mode == "capture", "blue", "gray"),
                ),
                rx.button(
                    rx.icon("wand-sparkles", size=16),
                    "Agent Output JSON",
                    on_click=lambda: State.set_creation_mode("agent_output"),
                    variant=rx.cond(State.dataset_creation_mode == "agent_output", "solid", "outline"),
                    color_scheme=rx.cond(State.dataset_creation_mode == "agent_output", "green", "gray"),
                ),
                rx.button(
                    rx.icon("file-json", size=16),
                    "Manual JSON",
                    on_click=lambda: State.set_creation_mode("manual"),
                    variant=rx.cond(State.dataset_creation_mode == "manual", "solid", "outline"),
                    color_scheme=rx.cond(State.dataset_creation_mode == "manual", "blue", "gray"),
                ),
                rx.button(
                    rx.icon("upload", size=16),
                    "Upload File",
                    on_click=lambda: State.set_creation_mode("upload"),
                    variant=rx.cond(State.dataset_creation_mode == "upload", "solid", "outline"),
                    color_scheme=rx.cond(State.dataset_creation_mode == "upload", "blue", "gray"),
                ),
                spacing="2",
                wrap="wrap",
            ),
            align_items="start",
            width="100%",
            spacing="3",
        ),

        rx.divider(),

        # Content based on mode
        rx.cond(
            State.dataset_creation_mode == "capture",
            capture_mode_section(),
        ),
        rx.cond(
            State.dataset_creation_mode == "agent_output",
            agent_output_section(),
        ),
        rx.cond(
            State.dataset_creation_mode == "manual",
            manual_mode_section(),
        ),
        rx.cond(
            State.dataset_creation_mode == "upload",
            upload_section(),
        ),

        spacing="5",
        align_items="start",
        width="100%",
    )


def preview_modal() -> rx.Component:
    """Render the dataset preview modal."""
    return rx.dialog.root(
        rx.dialog.content(
            rx.dialog.title("Dataset Preview"),
            rx.dialog.description(
                rx.cond(
                    State.preview_dataset_str != "",
                    rx.vstack(
                        rx.code_block(
                            State.preview_dataset_str,
                            language="json",
                            show_line_numbers=True,
                        ),
                        max_height="60vh",
                        overflow="auto",
                        width="100%",
                    ),
                    rx.text("Loading..."),
                ),
            ),
            rx.flex(
                rx.dialog.close(
                    rx.button("Close", variant="soft", on_click=State.close_preview),
                ),
                justify="end",
            ),
            max_width="800px",
        ),
        open=State.show_preview,
    )


def _llm_model_selector(options, value, on_change, on_refresh, loading, error, placeholder) -> rx.Component:
    """Shared model dropdown + refresh button used by every provider block."""
    return rx.vstack(
        rx.text("Model", weight="medium", size="2"),
        rx.hstack(
            rx.select(options, placeholder=placeholder, value=value, on_change=on_change, width="100%"),
            rx.button(
                rx.cond(loading, rx.spinner(size="1"), rx.icon("refresh-cw", size=14)),
                on_click=on_refresh,
                disabled=loading,
                variant="ghost",
                size="1",
            ),
            width="100%",
            align="center",
        ),
        rx.cond(error != "", rx.text(error, size="1", color="red")),
        align_items="start",
        width="100%",
    )


def _oauth_status_row(status, signed_in, sign_out_handler) -> rx.Component:
    """Show the signed-in indicator + a Sign out button when authenticated."""
    return rx.vstack(
        rx.cond(
            signed_in,
            rx.hstack(
                rx.icon("circle-check", size=16, color="green"),
                rx.text(rx.cond(status != "", status, "Signed in"), size="2", color_scheme="green"),
                rx.button(
                    rx.icon("log-out", size=14),
                    "Sign out",
                    on_click=sign_out_handler,
                    variant="outline",
                    color_scheme="red",
                    size="1",
                ),
                spacing="2",
                align="center",
            ),
            rx.cond(
                status != "",
                rx.text(status, size="1", color_scheme="gray"),
            ),
        ),
        align_items="start",
        width="100%",
        spacing="1",
    )


def _signin_openai() -> rx.Component:
    return rx.vstack(
        rx.text(
            "Use your ChatGPT (Plus/Pro) subscription. Click sign in, authorize in the "
            "browser, then paste the resulting code or the full redirect URL below.",
            size="1",
            color_scheme="gray",
        ),
        rx.hstack(
            rx.button(
                rx.icon("log-in", size=16),
                "Sign in with ChatGPT",
                on_click=State.start_openai_login,
                color_scheme="grass",
                variant="solid",
                size="2",
            ),
            _oauth_status_row(State.llm_openai_oauth_status, State.llm_openai_signed_in, State.sign_out_openai),
            spacing="3",
            align="center",
            width="100%",
        ),
        rx.cond(
            State.llm_openai_authorize_url != "",
            rx.vstack(
                rx.link("Open ChatGPT sign-in ↗", href=State.llm_openai_authorize_url, is_external=True, size="2"),
                rx.hstack(
                    rx.input(
                        placeholder="Paste authorization code or redirect URL",
                        value=State.llm_openai_code_input,
                        on_change=State.set_llm_openai_code_input,
                        width="100%",
                    ),
                    rx.button("Complete", on_click=State.complete_openai_login, size="2"),
                    width="100%",
                    spacing="2",
                ),
                align_items="start",
                width="100%",
                spacing="2",
            ),
        ),
        align_items="start",
        width="100%",
        spacing="2",
    )


def _signin_anthropic() -> rx.Component:
    return rx.vstack(
        rx.text(
            "Use your Claude (Pro/Max) subscription. Click sign in, authorize in the "
            "browser, then paste the code shown (in the form code#state) below.",
            size="1",
            color_scheme="gray",
        ),
        rx.hstack(
            rx.button(
                rx.icon("log-in", size=16),
                "Sign in with Claude",
                on_click=State.start_anthropic_login,
                color_scheme="orange",
                variant="solid",
                size="2",
            ),
            _oauth_status_row(State.llm_anthropic_oauth_status, State.llm_anthropic_signed_in, State.sign_out_anthropic),
            spacing="3",
            align="center",
            width="100%",
        ),
        rx.cond(
            State.llm_anthropic_authorize_url != "",
            rx.vstack(
                rx.link("Open Claude sign-in ↗", href=State.llm_anthropic_authorize_url, is_external=True, size="2"),
                rx.hstack(
                    rx.input(
                        placeholder="Paste authorization code (code#state)",
                        value=State.llm_anthropic_code_input,
                        on_change=State.set_llm_anthropic_code_input,
                        width="100%",
                    ),
                    rx.button("Complete", on_click=State.complete_anthropic_login, size="2"),
                    width="100%",
                    spacing="2",
                ),
                align_items="start",
                width="100%",
                spacing="2",
            ),
        ),
        align_items="start",
        width="100%",
        spacing="2",
    )


def _signin_copilot() -> rx.Component:
    return rx.vstack(
        rx.text(
            "Use your GitHub Copilot subscription. Click sign in, then enter the code "
            "shown below at the GitHub device page — this completes automatically.",
            size="1",
            color_scheme="gray",
        ),
        rx.hstack(
            rx.button(
                rx.cond(State.llm_copilot_login_active, rx.spinner(size="1"), rx.icon("log-in", size=16)),
                "Sign in with GitHub",
                on_click=State.start_copilot_login,
                disabled=State.llm_copilot_login_active,
                color_scheme="iris",
                variant="solid",
                size="2",
            ),
            _oauth_status_row(State.llm_copilot_oauth_status, State.llm_copilot_signed_in, State.sign_out_copilot),
            spacing="3",
            align="center",
            width="100%",
        ),
        rx.cond(
            State.llm_copilot_user_code != "",
            rx.hstack(
                rx.text("Enter code", size="2", color_scheme="gray"),
                rx.code(State.llm_copilot_user_code, size="3"),
                rx.text("at", size="2", color_scheme="gray"),
                rx.link(
                    "github.com/login/device ↗",
                    href=State.llm_copilot_verification_uri,
                    is_external=True,
                    size="2",
                ),
                spacing="2",
                align="center",
            ),
        ),
        align_items="start",
        width="100%",
        spacing="2",
    )


def _auth_method_selector(value, on_change) -> rx.Component:
    return rx.vstack(
        rx.text("Authentication Method", weight="medium", size="2"),
        rx.select(
            ["API Key", "Sign in"],
            value=value,
            on_change=on_change,
            width="200px",
        ),
        align_items="start",
        width="100%",
        spacing="2",
    )


def llm_settings_section() -> rx.Component:
    """Render the LLM Judge Settings section."""
    return rx.vstack(
        rx.heading("LLM Judge Settings", size="4"),
        rx.text(
            "Select the model provider and enter credentials used for DeepEval evaluation metrics.",
            size="2",
            color_scheme="gray",
        ),
        # Saved Profiles
        rx.cond(
            State.available_llm_profiles.length() > 0,
            rx.vstack(
                rx.text("Saved Profiles", weight="medium", size="2"),
                rx.hstack(
                    rx.select(
                        State.available_llm_profiles,
                        placeholder="Select a profile...",
                        value=State.selected_llm_profile,
                        on_change=State.load_llm_profile,
                    ),
                    rx.alert_dialog.root(
                        rx.alert_dialog.trigger(
                            rx.button(
                                rx.icon("trash-2", size=16),
                                "Delete",
                                color_scheme="red",
                                variant="outline",
                                size="2",
                                disabled=State.selected_llm_profile == "",
                            ),
                        ),
                        rx.alert_dialog.content(
                            rx.alert_dialog.title("Delete LLM Profile"),
                            rx.alert_dialog.description(
                                rx.text(
                                    "Are you sure you want to delete the profile ",
                                    rx.text(State.selected_llm_profile, weight="bold", as_="span"),
                                    "? This cannot be undone.",
                                ),
                                size="2",
                            ),
                            rx.hstack(
                                rx.alert_dialog.cancel(
                                    rx.button("Cancel", variant="soft", color_scheme="gray"),
                                ),
                                rx.alert_dialog.action(
                                    rx.button(
                                        rx.icon("trash-2", size=14),
                                        "Delete",
                                        color_scheme="red",
                                        on_click=State.delete_llm_profile,
                                    ),
                                ),
                                justify="end",
                                spacing="2",
                                margin_top="4",
                            ),
                        ),
                    ),
                    spacing="2",
                    width="100%",
                ),
                align_items="start",
                width="100%",
                spacing="2",
            ),
        ),
        # Save as Profile
        rx.vstack(
            rx.text("Save as Profile", weight="medium", size="2"),
            rx.hstack(
                rx.input(
                    placeholder="Profile name (e.g., Gemini Flash)",
                    value=State.llm_profile_name_input,
                    on_change=State.set_llm_profile_name_input,
                    width="100%",
                ),
                rx.button(
                    rx.icon("bookmark-plus", size=16),
                    "Save",
                    on_click=State.save_llm_profile,
                    color_scheme="green",
                    variant="outline",
                    size="2",
                ),
                spacing="2",
                width="100%",
            ),
            align_items="start",
            width="100%",
            spacing="2",
        ),
        # Profile status
        rx.cond(
            State.llm_profile_status != "",
            rx.callout(
                State.llm_profile_status,
                icon=rx.cond(
                    State.llm_profile_status.contains("✅"),
                    "circle-check",
                    "circle-x",
                ),
                color_scheme=rx.cond(
                    State.llm_profile_status.contains("✅"),
                    "green",
                    "red",
                ),
                size="1",
            ),
        ),
        rx.divider(),
        # Provider selector
        rx.vstack(
            rx.text("Model Provider", weight="medium", size="2"),
            rx.select(
                ["Google Gemini", "AWS Bedrock", "OpenAI", "GitHub Copilot", "Anthropic"],
                placeholder="Select provider...",
                value=State.llm_provider,
                on_change=State.set_llm_provider,
                width="220px",
            ),
            align_items="start",
            width="100%",
            spacing="2",
        ),
        # Gemini fields
        rx.cond(
            State.llm_provider == "Google Gemini",
            rx.vstack(
                rx.grid(
                    rx.vstack(
                        rx.text("Gemini API Key", weight="medium", size="2"),
                        rx.input(
                            placeholder=rx.cond(
                                State.llm_gemini_key_saved,
                                "Key saved — enter new value to replace",
                                "AIzaSy...",
                            ),
                            value=State.llm_gemini_api_key,
                            on_change=State.set_llm_gemini_api_key,
                            type="password",
                            width="100%",
                        ),
                        align_items="start",
                        width="100%",
                    ),
                    rx.vstack(
                        rx.text("Model", weight="medium", size="2"),
                        rx.hstack(
                            rx.select(
                                State.gemini_model_options,
                                placeholder="gemini-2.5-flash",
                                value=State.llm_gemini_model_id,
                                on_change=State.set_llm_gemini_model_id,
                                width="100%",
                            ),
                            rx.button(
                                rx.cond(
                                    State.llm_gemini_models_loading,
                                    rx.spinner(size="1"),
                                    rx.icon("refresh-cw", size=14),
                                ),
                                on_click=State.load_gemini_models,
                                disabled=State.llm_gemini_models_loading,
                                variant="ghost",
                                size="1",
                            ),
                            width="100%",
                            align="center",
                        ),
                        align_items="start",
                        width="100%",
                    ),
                    columns="2",
                    spacing="4",
                    width="100%",
                ),
                rx.cond(
                    State.llm_gemini_models_error != "",
                    rx.text(State.llm_gemini_models_error, size="1", color="red"),
                ),
                align_items="start",
                width="100%",
                spacing="3",
            ),
        ),
        # Bedrock fields
        rx.cond(
            State.llm_provider == "AWS Bedrock",
            rx.vstack(
                # Authentication method selector
                rx.vstack(
                    rx.text("Authentication Method", weight="medium", size="2"),
                    rx.select(
                        ["Access Keys", "SSO Profile"],
                        value=State.llm_aws_auth_method,
                        on_change=State.set_llm_aws_auth_method,
                        width="200px",
                    ),
                    align_items="start",
                    width="100%",
                    spacing="2",
                ),
                # Access Keys credential fields
                rx.cond(
                    State.llm_aws_auth_method == "Access Keys",
                    rx.grid(
                        rx.vstack(
                            rx.text("AWS Access Key ID", weight="medium", size="2"),
                            rx.input(
                                placeholder=rx.cond(
                                    State.llm_aws_access_key_saved,
                                    "Key saved — enter new value to replace",
                                    "AKIAIOSFODNN7EXAMPLE",
                                ),
                                value=State.llm_aws_access_key_id,
                                on_change=State.set_llm_aws_access_key_id,
                                width="100%",
                            ),
                            align_items="start",
                            width="100%",
                        ),
                        rx.vstack(
                            rx.text("AWS Secret Access Key", weight="medium", size="2"),
                            rx.input(
                                placeholder=rx.cond(
                                    State.llm_aws_access_key_saved,
                                    "Saved — enter new value to replace",
                                    "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                                ),
                                value=State.llm_aws_secret_access_key,
                                on_change=State.set_llm_aws_secret_access_key,
                                type="password",
                                width="100%",
                            ),
                            align_items="start",
                            width="100%",
                        ),
                        columns="2",
                        spacing="4",
                        width="100%",
                    ),
                ),
                # SSO Profile credential field
                rx.cond(
                    State.llm_aws_auth_method == "SSO Profile",
                    rx.vstack(
                        rx.vstack(
                            rx.text("SSO Profile Name", weight="medium", size="2"),
                            rx.input(
                                placeholder="e.g. my-sso-profile",
                                value=State.llm_aws_profile,
                                on_change=State.set_llm_aws_profile,
                                width="100%",
                            ),
                            align_items="start",
                            width="100%",
                            spacing="2",
                        ),
                        rx.callout(
                            "The profile must be configured in ~/.aws/config. "
                            "Run `aws sso login --profile <name>` before testing.",
                            icon="info",
                            color_scheme="blue",
                            size="1",
                        ),
                        align_items="start",
                        width="100%",
                        spacing="2",
                    ),
                ),
                # Region and Model ID — always shown for Bedrock
                rx.grid(
                    rx.vstack(
                        rx.text("AWS Region", weight="medium", size="2"),
                        rx.input(
                            placeholder="us-east-1",
                            value=State.llm_aws_region,
                            on_change=State.set_llm_aws_region,
                            width="100%",
                        ),
                        align_items="start",
                        width="100%",
                    ),
                    rx.vstack(
                        rx.text("Bedrock Model ID", weight="medium", size="2"),
                        rx.hstack(
                            rx.select(
                                State.bedrock_model_options,
                                placeholder="anthropic.claude-3-5-sonnet-20241022-v2:0",
                                value=State.llm_aws_bedrock_model_id,
                                on_change=State.set_llm_aws_bedrock_model_id,
                                width="100%",
                            ),
                            rx.button(
                                rx.cond(
                                    State.llm_bedrock_models_loading,
                                    rx.spinner(size="1"),
                                    rx.icon("refresh-cw", size=14),
                                ),
                                on_click=State.load_bedrock_models,
                                disabled=State.llm_bedrock_models_loading,
                                variant="ghost",
                                size="1",
                            ),
                            width="100%",
                            align="center",
                        ),
                        rx.cond(
                            State.llm_bedrock_models_error != "",
                            rx.text(State.llm_bedrock_models_error, size="1", color="red"),
                        ),
                        align_items="start",
                        width="100%",
                    ),
                    columns="2",
                    spacing="4",
                    width="100%",
                ),
                align_items="start",
                width="100%",
                spacing="3",
            ),
        ),
        # OpenAI fields
        rx.cond(
            State.llm_provider == "OpenAI",
            rx.vstack(
                _auth_method_selector(State.llm_openai_auth_method, State.set_llm_openai_auth_method),
                rx.cond(
                    State.llm_openai_auth_method == "API Key",
                    rx.vstack(
                        rx.text("OpenAI API Key", weight="medium", size="2"),
                        rx.input(
                            placeholder=rx.cond(
                                State.llm_openai_key_saved,
                                "Key saved — enter new value to replace",
                                "sk-...",
                            ),
                            value=State.llm_openai_api_key,
                            on_change=State.set_llm_openai_api_key,
                            type="password",
                            width="100%",
                        ),
                        align_items="start",
                        width="100%",
                    ),
                    _signin_openai(),
                ),
                _llm_model_selector(
                    State.openai_model_options,
                    State.llm_openai_model_id,
                    State.set_llm_openai_model_id,
                    State.load_openai_models,
                    State.llm_openai_models_loading,
                    State.llm_openai_models_error,
                    "gpt-4o",
                ),
                align_items="start",
                width="100%",
                spacing="3",
            ),
        ),
        # GitHub Copilot fields
        rx.cond(
            State.llm_provider == "GitHub Copilot",
            rx.vstack(
                _auth_method_selector(State.llm_copilot_auth_method, State.set_llm_copilot_auth_method),
                rx.cond(
                    State.llm_copilot_auth_method == "API Key",
                    rx.vstack(
                        rx.text("GitHub Token", weight="medium", size="2"),
                        rx.input(
                            placeholder=rx.cond(
                                State.llm_copilot_key_saved,
                                "Token saved — enter new value to replace",
                                "ghp_... or github_pat_...",
                            ),
                            value=State.llm_copilot_api_key,
                            on_change=State.set_llm_copilot_api_key,
                            type="password",
                            width="100%",
                        ),
                        align_items="start",
                        width="100%",
                    ),
                    _signin_copilot(),
                ),
                _llm_model_selector(
                    State.copilot_model_options,
                    State.llm_copilot_model_id,
                    State.set_llm_copilot_model_id,
                    State.load_copilot_models,
                    State.llm_copilot_models_loading,
                    State.llm_copilot_models_error,
                    rx.cond(State.llm_copilot_auth_method == "API Key", "openai/gpt-4o", "gpt-4o"),
                ),
                align_items="start",
                width="100%",
                spacing="3",
            ),
        ),
        # Anthropic fields
        rx.cond(
            State.llm_provider == "Anthropic",
            rx.vstack(
                _auth_method_selector(State.llm_anthropic_auth_method, State.set_llm_anthropic_auth_method),
                rx.cond(
                    State.llm_anthropic_auth_method == "API Key",
                    rx.vstack(
                        rx.text("Anthropic API Key", weight="medium", size="2"),
                        rx.input(
                            placeholder=rx.cond(
                                State.llm_anthropic_key_saved,
                                "Key saved — enter new value to replace",
                                "sk-ant-...",
                            ),
                            value=State.llm_anthropic_api_key,
                            on_change=State.set_llm_anthropic_api_key,
                            type="password",
                            width="100%",
                        ),
                        align_items="start",
                        width="100%",
                    ),
                    _signin_anthropic(),
                ),
                _llm_model_selector(
                    State.anthropic_model_options,
                    State.llm_anthropic_model_id,
                    State.set_llm_anthropic_model_id,
                    State.load_anthropic_models,
                    State.llm_anthropic_models_loading,
                    State.llm_anthropic_models_error,
                    "claude-sonnet-4-5",
                ),
                align_items="start",
                width="100%",
                spacing="3",
            ),
        ),
        # Action buttons
        rx.hstack(
            rx.button(
                rx.icon("save", size=16),
                "Save LLM Settings",
                on_click=State.save_llm_settings,
                color_scheme="blue",
                size="2",
            ),
            rx.button(
                rx.cond(
                    State.llm_testing,
                    rx.spinner(size="1"),
                    rx.icon("plug", size=16),
                ),
                "Test Connection",
                on_click=State.test_llm_connection,
                disabled=State.llm_testing,
                variant="outline",
                color_scheme="gray",
                size="2",
            ),
            spacing="3",
        ),
        # Save status
        rx.cond(
            State.llm_settings_status != "",
            rx.callout(
                State.llm_settings_status,
                icon="info",
                color_scheme=rx.cond(
                    State.llm_settings_status.contains("✅"),
                    "green",
                    "red",
                ),
            ),
        ),
        # Test connection status
        rx.cond(
            State.llm_test_status != "",
            rx.callout(
                State.llm_test_status,
                icon=rx.cond(
                    State.llm_test_status.contains("✅"),
                    "circle-check",
                    rx.cond(
                        State.llm_test_status.contains("Testing"),
                        "loader",
                        "circle-x",
                    ),
                ),
                color_scheme=rx.cond(
                    State.llm_test_status.contains("✅"),
                    "green",
                    rx.cond(
                        State.llm_test_status.contains("Testing"),
                        "blue",
                        "red",
                    ),
                ),
            ),
        ),
        align_items="start",
        width="100%",
        spacing="3",
    )


def config_section() -> rx.Component:
    """Render the project configuration section."""
    return rx.vstack(
        # Header
        rx.heading("Project Configuration", size="6"),
        rx.text(
            "Create or modify project configuration for DeepEval testing.",
            color_scheme="gray",
        ),
        rx.divider(),

        # LLM Judge Settings
        llm_settings_section(),

        rx.divider(),

        # Load existing config
        rx.cond(
            State.available_configs.length() > 0,
            rx.vstack(
                rx.heading("Load Existing Configuration", size="4"),
                rx.hstack(
                    rx.select(
                        State.available_configs,
                        placeholder="Select a configuration...",
                        value=State.selected_config,
                        on_change=State.load_existing_config,
                    ),
                    rx.button(
                        rx.icon("refresh-cw", size=16),
                        "Refresh",
                        on_click=State.load_available_configs,
                        variant="outline",
                        size="2",
                    ),
                    rx.alert_dialog.root(
                        rx.alert_dialog.trigger(
                            rx.button(
                                rx.icon("trash-2", size=16),
                                "Delete",
                                color_scheme="red",
                                variant="outline",
                                size="2",
                                disabled=State.selected_config == "",
                            ),
                        ),
                        rx.alert_dialog.content(
                            rx.alert_dialog.title("Delete Configuration"),
                            rx.alert_dialog.description(
                                rx.text(
                                    "Are you sure you want to delete ",
                                    rx.text(State.selected_config, weight="bold", as_="span"),
                                    "? This will also delete any associated golden session and profile files. This cannot be undone.",
                                ),
                                size="2",
                            ),
                            rx.hstack(
                                rx.alert_dialog.cancel(
                                    rx.button("Cancel", variant="soft", color_scheme="gray"),
                                ),
                                rx.alert_dialog.action(
                                    rx.button(
                                        rx.icon("trash-2", size=14),
                                        "Delete",
                                        color_scheme="red",
                                        on_click=State.delete_project_config,
                                    ),
                                ),
                                justify="end",
                                spacing="2",
                                margin_top="4",
                            ),
                        ),
                    ),
                    spacing="2",
                    width="100%",
                ),
                align_items="start",
                width="100%",
                spacing="3",
            ),
        ),
        
        rx.divider(),
        
        # Project Info
        rx.vstack(
            rx.heading("Project Information", size="4"),
            rx.grid(
                rx.vstack(
                    rx.text("Project Name *", weight="medium", size="2"),
                    rx.input(
                        placeholder="My Agent Project",
                        value=State.config_project_name,
                        on_change=State.set_config_project_name,
                        width="100%",
                    ),
                    align_items="start",
                    width="100%",
                ),
                rx.vstack(
                    rx.text("Version", weight="medium", size="2"),
                    rx.input(
                        placeholder="1.0",
                        value=State.config_version,
                        on_change=State.set_config_version,
                        width="100%",
                    ),
                    align_items="start",
                    width="100%",
                ),
                rx.vstack(
                    rx.text("Agent Type", weight="medium", size="2"),
                    rx.select(
                        ["conversational", "step_agent"],
                        value=State.config_agent_type,
                        on_change=State.set_config_agent_type,
                        width="100%",
                    ),
                    align_items="start",
                    width="100%",
                ),
                columns="3",
                spacing="4",
                width="100%",
            ),
            align_items="start",
            width="100%",
            spacing="3",
        ),
        
        rx.divider(),
        
        # Connection Settings
        rx.vstack(
            rx.heading("Connection Settings", size="4"),
            rx.grid(
                rx.vstack(
                    rx.text("Base URL", weight="medium", size="2"),
                    rx.input(
                        placeholder="https://your-pega-instance.example.com",
                        value=State.config_base_url,
                        on_change=State.set_config_base_url,
                        width="100%",
                    ),
                    align_items="start",
                    width="100%",
                ),
                rx.vstack(
                    rx.text("Agent Name", weight="medium", size="2"),
                    rx.input(
                        placeholder="YOUR-AGENT-NAME",
                        value=State.config_agent_name,
                        on_change=State.set_config_agent_name,
                        width="100%",
                    ),
                    align_items="start",
                    width="100%",
                ),
                rx.vstack(
                    rx.text("A2A App Path", weight="medium", size="2"),
                    rx.input(
                        placeholder="your-app",
                        value=State.config_a2a_app_path,
                        on_change=State.set_config_a2a_app_path,
                        width="100%",
                    ),
                    align_items="start",
                    width="100%",
                ),
                rx.vstack(
                    rx.text("Token URL Override (optional)", weight="medium", size="2"),
                    rx.input(
                        placeholder="Leave empty to use default",
                        value=State.config_token_url_override,
                        on_change=State.set_config_token_url_override,
                        width="100%",
                    ),
                    align_items="start",
                    width="100%",
                ),
                rx.vstack(
                    rx.text("OAuth Client ID", weight="medium", size="2"),
                    rx.input(
                        placeholder="Your Pega OAuth client ID",
                        value=State.config_pega_client_id,
                        on_change=State.set_config_pega_client_id,
                        width="100%",
                    ),
                    align_items="start",
                    width="100%",
                ),
                rx.vstack(
                    rx.text("OAuth Client Secret", weight="medium", size="2"),
                    rx.input(
                        placeholder="Your Pega OAuth client secret",
                        value=State.config_pega_client_secret,
                        on_change=State.set_config_pega_client_secret,
                        type="password",
                        width="100%",
                    ),
                    align_items="start",
                    width="100%",
                ),
                columns="2",
                spacing="4",
                width="100%",
            ),
            align_items="start",
            width="100%",
            spacing="3",
        ),

        rx.divider(),

        # Agent Identity
        rx.vstack(
            rx.heading("Agent Identity", size="4"),
            rx.text(
                "These fields help the LLM judge evaluate conversation quality and role adherence.",
                size="2",
                color_scheme="gray",
            ),
            rx.vstack(
                rx.text("Role Description", weight="medium", size="2"),
                rx.text_area(
                    placeholder="Describe what this agent does...",
                    value=State.config_role,
                    on_change=State.set_config_role,
                    width="100%",
                    min_height="80px",
                ),
                align_items="start",
                width="100%",
            ),
            rx.grid(
                rx.vstack(
                    rx.text("Domain", weight="medium", size="2"),
                    rx.input(
                        placeholder="Business domain",
                        value=State.config_domain,
                        on_change=State.set_config_domain,
                        width="100%",
                    ),
                    align_items="start",
                    width="100%",
                ),
                rx.vstack(
                    rx.text("Organization", weight="medium", size="2"),
                    rx.input(
                        placeholder="Organization name",
                        value=State.config_organization,
                        on_change=State.set_config_organization,
                        width="100%",
                    ),
                    align_items="start",
                    width="100%",
                ),
                columns="2",
                spacing="4",
                width="100%",
            ),
            rx.vstack(
                rx.text("Off-Topic Guidance", weight="medium", size="2"),
                rx.text_area(
                    placeholder="What topics should the agent refuse to answer?",
                    value=State.config_off_topic_guidance,
                    on_change=State.set_config_off_topic_guidance,
                    width="100%",
                    min_height="60px",
                ),
                align_items="start",
                width="100%",
            ),
            align_items="start",
            width="100%",
            spacing="3",
        ),
        
        rx.divider(),
        
        # Workflow
        rx.vstack(
            rx.heading("Workflow Configuration", size="4"),
            rx.text(
                "Define the named workflows this agent can execute. Each entry requires an "
                "id (used with --workflow-id at capture time), a description, and a stages "
                "array (empty for FAQ/no-workflow sessions).",
                size="2",
                color_scheme="gray",
            ),
            rx.vstack(
                rx.hstack(
                    rx.text("Workflows", weight="medium", size="2"),
                    rx.badge("JSON array", color_scheme="blue", size="1"),
                    align="center",
                    spacing="2",
                ),
                rx.text_area(
                    placeholder='[\n  {\n    "id": "my_workflow",\n    "description": "A customer files a complaint which is investigated and resolved.",\n    "stages": [\n      {"name": "Stage 1", "description": "Collect details."},\n      {"name": "Stage 2", "description": "Investigate."},\n      {"name": "Stage 3", "description": "Resolve."}\n    ]\n  },\n  {\n    "id": "faq",\n    "description": "Agent answers knowledge-base questions with no case creation.",\n    "stages": []\n  }\n]',
                    value=State.config_workflows,
                    on_change=State.set_config_workflows,
                    width="100%",
                    min_height="220px",
                    font_family="monospace",
                    font_size="12px",
                ),
                rx.hstack(
                    rx.button(
                        "Validate JSON",
                        on_click=State.validate_workflows_json,
                        size="1",
                        variant="outline",
                        color_scheme="blue",
                    ),
                    rx.cond(
                        State.config_workflows_validation != "",
                        rx.text(
                            State.config_workflows_validation,
                            size="1",
                            color_scheme=rx.cond(
                                State.config_workflows_validation.startswith("✅"),
                                "green",
                                rx.cond(
                                    State.config_workflows_validation.startswith("⚠️"),
                                    "orange",
                                    "red",
                                ),
                            ),
                        ),
                    ),
                    align="center",
                    spacing="3",
                    width="100%",
                ),
                rx.text(
                    "Tip: capture with --workflow-id <id> to annotate which workflow a session exercises. "
                    "Sessions without an annotated workflow skip the ConversationCompleteness test.",
                    size="1",
                    color_scheme="gray",
                ),
                align_items="start",
                width="100%",
                spacing="2",
            ),
            align_items="start",
            width="100%",
            spacing="3",
        ),
        
        rx.divider(),
        
        # Hallucination Context
        rx.vstack(
            rx.heading("Hallucination Context", size="4"),
            rx.text(
                "Add factual statements about what the agent can and should do (one per line).",
                size="2",
                color_scheme="gray",
            ),
            rx.text_area(
                placeholder="The agent can access customer data...\nThe agent always confirms before...\nThe agent integrates with...",
                value=State.config_hallucination_context,
                on_change=State.set_config_hallucination_context,
                width="100%",
                min_height="150px",
            ),
            align_items="start",
            width="100%",
            spacing="3",
        ),
        
        rx.divider(),
        
        # Save Button
        rx.hstack(
            rx.button(
                rx.icon("save", size=16),
                "Save Configuration",
                on_click=State.save_project_config,
                color_scheme="green",
                size="3",
            ),
            rx.button(
                rx.icon("rotate-ccw", size=16),
                "Reset to Template",
                on_click=State.load_config_template,
                variant="outline",
                size="3",
            ),
            spacing="3",
        ),
        
        rx.cond(
            State.config_status != "",
            rx.callout(
                State.config_status,
                icon="info",
                color_scheme=rx.cond(
                    State.config_status.contains("✅"),
                    "green",
                    rx.cond(
                        State.config_status.contains("❌"),
                        "red",
                        "blue"
                    )
                ),
            ),
        ),

        rx.divider(),

        # API Management
        api_management_section(),

        spacing="5",
        align_items="start",
        width="100%",
    )


def api_management_section() -> rx.Component:
    """Render the API server and OAuth client management section."""
    return rx.vstack(
        rx.heading("REST API", size="4"),
        rx.text(
            "Manage the REST API server and register OAuth clients for programmatic access.",
            size="2",
            color_scheme="gray",
        ),

        # --- Server Controls ---
        rx.vstack(
            rx.text("API Server", weight="medium", size="2"),
            rx.hstack(
                rx.vstack(
                    rx.text("Port", size="2"),
                    rx.input(
                        placeholder="8100",
                        value=State.api_server_port,
                        on_change=State.set_api_server_port,
                        width="100px",
                        disabled=State.api_server_running,
                    ),
                    spacing="1",
                ),
                rx.button(
                    rx.icon("play", size=16),
                    "Start Server",
                    on_click=State.start_api_server,
                    color_scheme="green",
                    size="2",
                    disabled=State.api_server_running,
                ),
                rx.button(
                    rx.icon("square", size=16),
                    "Stop Server",
                    on_click=State.stop_api_server,
                    color_scheme="red",
                    variant="outline",
                    size="2",
                    disabled=~State.api_server_running,
                ),
                rx.cond(
                    State.api_server_running,
                    rx.link(
                        rx.button(
                            rx.icon("book-open", size=16),
                            "OpenAPI Docs",
                            variant="outline",
                            size="2",
                        ),
                        href=rx.cond(
                            State.api_server_port != "",
                            "http://localhost:" + State.api_server_port + "/docs",
                            "http://localhost:8100/docs",
                        ),
                        is_external=True,
                    ),
                ),
                align_items="end",
                spacing="3",
            ),
            rx.cond(
                State.api_server_running,
                rx.badge("Running", color_scheme="green", variant="surface", size="1"),
                rx.badge("Stopped", color_scheme="gray", variant="surface", size="1"),
            ),
            rx.cond(
                State.api_server_status != "",
                rx.callout(
                    State.api_server_status,
                    icon=rx.cond(
                        State.api_server_status.contains("✅"),
                        "circle-check",
                        "circle-x",
                    ),
                    color_scheme=rx.cond(
                        State.api_server_status.contains("✅"),
                        "green",
                        "red",
                    ),
                    size="1",
                ),
            ),
            align_items="start",
            width="100%",
            spacing="2",
        ),

        rx.divider(),

        # --- OAuth Client Registration ---
        rx.vstack(
            rx.text("OAuth Clients", weight="medium", size="2"),
            rx.text(
                "Register API clients that authenticate via OAuth 2.0 client_credentials grant.",
                size="1",
                color_scheme="gray",
            ),
            # Register new client
            rx.hstack(
                rx.input(
                    placeholder="Client description (e.g., CI Pipeline)",
                    value=State.api_new_client_description,
                    on_change=State.set_api_new_client_description,
                    width="100%",
                ),
                rx.button(
                    rx.icon("user-plus", size=16),
                    "Register Client",
                    on_click=State.register_api_client,
                    color_scheme="blue",
                    size="2",
                ),
                spacing="2",
                width="100%",
            ),
            # Show newly created credentials
            rx.cond(
                State.api_created_client_secret != "",
                rx.callout(
                    rx.vstack(
                        rx.text("Copy these credentials now — the secret will not be shown again.", weight="bold", size="2"),
                        rx.vstack(
                            rx.text("Client ID", size="1", color_scheme="gray"),
                            rx.code(State.api_created_client_id, size="2"),
                            spacing="1",
                        ),
                        rx.vstack(
                            rx.text("Client Secret", size="1", color_scheme="gray"),
                            rx.code(State.api_created_client_secret, size="2"),
                            spacing="1",
                        ),
                        rx.hstack(
                            rx.button(
                                rx.icon("download", size=14),
                                "Download",
                                on_click=State.download_client_credentials,
                                variant="soft",
                                color_scheme="blue",
                                size="1",
                            ),
                            rx.button(
                                "Dismiss",
                                on_click=State.dismiss_client_credentials,
                                variant="soft",
                                size="1",
                            ),
                            spacing="2",
                        ),
                        spacing="2",
                    ),
                    icon="key",
                    color_scheme="amber",
                ),
            ),
            # Client status
            rx.cond(
                (State.api_client_status != "") & (State.api_created_client_secret == ""),
                rx.callout(
                    State.api_client_status,
                    icon=rx.cond(
                        State.api_client_status.contains("✅"),
                        "circle-check",
                        "circle-x",
                    ),
                    color_scheme=rx.cond(
                        State.api_client_status.contains("✅"),
                        "green",
                        "red",
                    ),
                    size="1",
                ),
            ),
            # Existing clients table
            rx.cond(
                State.api_clients.length() > 0,
                rx.vstack(
                    rx.text("Registered Clients", size="2", weight="medium"),
                    rx.table.root(
                        rx.table.header(
                            rx.table.row(
                                rx.table.column_header_cell("Client ID"),
                                rx.table.column_header_cell("Description"),
                                rx.table.column_header_cell("Created"),
                                rx.table.column_header_cell(""),
                            ),
                        ),
                        rx.table.body(
                            rx.foreach(
                                State.api_clients,
                                _render_client_row,
                            ),
                        ),
                        width="100%",
                        size="1",
                    ),
                    align_items="start",
                    width="100%",
                    spacing="2",
                ),
            ),
            align_items="start",
            width="100%",
            spacing="3",
        ),

        align_items="start",
        width="100%",
        spacing="3",
    )


def _render_client_row(client: ApiClientInfo) -> rx.Component:
    """Render a single row in the OAuth clients table."""
    return rx.table.row(
        rx.table.cell(rx.code(client.client_id, size="1")),
        rx.table.cell(rx.text(client.description, size="2")),
        rx.table.cell(
            rx.text(
                client.created_at.to(str)[:10],
                size="2",
                color_scheme="gray",
            ),
        ),
        rx.table.cell(
            rx.alert_dialog.root(
                rx.alert_dialog.trigger(
                    rx.button(
                        rx.icon("trash-2", size=14),
                        color_scheme="red",
                        variant="ghost",
                        size="1",
                    ),
                ),
                rx.alert_dialog.content(
                    rx.alert_dialog.title("Delete OAuth Client"),
                    rx.alert_dialog.description(
                        rx.text(
                            "Delete client ",
                            rx.text(client.client_id, weight="bold", as_="span"),
                            "? Any tokens issued to this client will stop working.",
                        ),
                        size="2",
                    ),
                    rx.hstack(
                        rx.alert_dialog.cancel(
                            rx.button("Cancel", variant="soft", color_scheme="gray"),
                        ),
                        rx.alert_dialog.action(
                            rx.button(
                                rx.icon("trash-2", size=14),
                                "Delete",
                                color_scheme="red",
                                on_click=State.delete_api_client(client.client_id),
                            ),
                        ),
                        justify="end",
                        spacing="2",
                        margin_top="4",
                    ),
                ),
            ),
        ),
    )


def navbar() -> rx.Component:
    """Render the navigation bar."""
    return rx.hstack(
        rx.hstack(
            rx.icon("flask-conical", size=28, color="var(--accent-9)"),
            rx.heading("DeepEval Pega", size="5"),
            spacing="2",
            align="center",
        ),
        rx.spacer(),
        rx.hstack(
            rx.button(
                rx.icon("bar-chart-3", size=16),
                "Evaluation",
                on_click=lambda: State.set_active_tab("evaluation"),
                variant=rx.cond(State.active_tab == "evaluation", "solid", "ghost"),
                color_scheme=rx.cond(State.active_tab == "evaluation", "blue", "gray"),
                size="2",
            ),
            rx.button(
                rx.icon("database", size=16),
                "Golden Datasets",
                on_click=lambda: State.set_active_tab("datasets"),
                variant=rx.cond(State.active_tab == "datasets", "solid", "ghost"),
                color_scheme=rx.cond(State.active_tab == "datasets", "blue", "gray"),
                size="2",
            ),
            rx.button(
                rx.icon("settings", size=16),
                "Configuration",
                on_click=lambda: State.set_active_tab("config"),
                variant=rx.cond(State.active_tab == "config", "solid", "ghost"),
                color_scheme=rx.cond(State.active_tab == "config", "blue", "gray"),
                size="2",
            ),
            spacing="2",
            align_items="center",
        ),
        rx.color_mode.button(),
        align_items="center",
        padding="4",
        border_bottom="1px solid var(--gray-5)",
        background="var(--color-background)",
        position="sticky",
        top="0",
        z_index="100",
        width="100%",
    )


def index() -> rx.Component:
    """Main page component."""
    return rx.box(
        navbar(),
        rx.container(
            rx.cond(
                State.active_tab == "evaluation",
                evaluation_section(),
                rx.cond(
                    State.active_tab == "datasets",
                    golden_dataset_section(),
                    config_section(),
                ),
            ),
            preview_modal(),
            padding="6",
            padding_bottom="120px",
            max_width="1200px",
        ),
        min_height="100vh",
        on_mount=State.init_app,
    )


app = rx.App()
app.add_page(index, title="DeepEval Pega - Evaluation Dashboard")
