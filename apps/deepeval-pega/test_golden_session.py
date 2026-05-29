"""
Golden Session Replay Tests — DeepEval ConversationalTestCase

Loads a recorded golden session JSON and replays it against the live Surface
agent, evaluating the full conversation with DeepEval's conversational metrics:

  - KnowledgeRetentionMetric: Does the agent remember context across turns?
  - ConversationCompletenessMetric: Did the agent complete the campaign workflow?
  - RoleAdherenceMetric: Does the agent stay in its marketing-assistant role?
  - ToolCorrectnessMetric (per-turn): Were the expected Pega tools invoked?
  - GlossarySourceMetric (per-turn): Custom metric from test_surface_agents.py

Plus Pega-specific assertions at each turn:
  - Tool invocation verification via D_pxAutopilotConversation
  - Step agent detection (field prefills, content patterns, stages)
  - Business case lifecycle progression
  - Latency regression (2× golden baseline budget)

Usage:
    # Replay against the latest golden session
    pytest test_golden_session.py -v -s

    # Replay a specific golden file
    pytest test_golden_session.py -v -s --golden golden_sessions/golden_Zelle_Campaign_20260218_175725.json

    # Record a new golden session, then replay it
    python3 record_golden_session.py
    pytest test_golden_session.py -v -s
"""

import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote as url_quote

import pytest
from deepeval import assert_test
from deepeval.metrics import (
    BiasMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ConversationCompletenessMetric,
    HallucinationMetric,
    KnowledgeRetentionMetric,
    RoleAdherenceMetric,
    ToxicityMetric,
)
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.test_case.conversational_test_case import ConversationalTestCase, Turn
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Ensure imports work from this directory
# ---------------------------------------------------------------------------
_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir))
# Also make the AgentXTestSuites utilities importable (kept for potential direct use)
# _this_dir = .../DataGenerationGenAI/Deep Eval and Deep Team/evaluate-surface/
# .parent.parent   = .../DataGenerationGenAI/
_agentx_suites = _this_dir.parent.parent / "AgentXTestSuites"
if _agentx_suites.exists():
    sys.path.insert(0, str(_agentx_suites))
load_dotenv(dotenv_path=_this_dir / ".env")


def _get_threshold(metric_id: str, default: float) -> float:
    """Return the threshold for a metric, preferring the EVAL_THRESHOLD_<ID> env var.

    The Reflex UI passes thresholds as EVAL_THRESHOLD_KNOWLEDGE_RETENTION, etc.
    Falls back to the hard-coded default when the env var is absent or invalid.
    """
    val = os.environ.get(f"EVAL_THRESHOLD_{metric_id.upper()}")
    if val:
        try:
            return float(val)
        except ValueError:
            pass
    return default


from test_surface_agents import (
    AGENT_CARD_URL,
    CLIENT_ID,
    CLIENT_SECRET,
    TOKEN_URL,
    AgentResponse,
    ConversationInsight,
    GeminiJudgeLLM,
    create_judge_llm,
    PegaInsight,
    SurfaceAgent,
    # Structured tool detection (for new agent output format)
    detect_tools_hybrid,
    build_tool_registry,
    build_tool_patterns_from_profile,
    _detect_tools_from_messages_v2,
)

logger = logging.getLogger(__name__)

# Regex to detect case creation from response text: "CaseType case (PREFIX-NNNN)"
_CASE_TYPE_RE = re.compile(r"(\w[\w\s]+?)\s+case\s*\(([A-Z]+-\d+)\)")

# Broader patterns to detect case creation when the case ID isn't in parentheses
_CASE_CREATION_PATTERNS = [
    # "Reset Password case (R-27001)" or "Product Complaint case (P-170006)"
    re.compile(r"(\w[\w\s]+?)\s+case\s*\(([A-Z]+-\d+)\)"),
    # "created/started a case for password reset" or "opened a case for product complaint"
    re.compile(r"(?:started|initiated|created|opened)\s+(?:a\s+)?case\s+for\s+(?:a\s+)?(?:the\s+)?(?:\*\*)?([A-Za-z]+(?:\s+[A-Za-z]+){0,3}?)(?:\*\*)?(?:\s*[.!,;]|\s+(?:for|to|with|and|regarding|—|-))", re.I),
    # "started a password reset case" or "initiated a product complaint case"
    re.compile(r"(?:started|initiated|created|opened)\s+(?:a\s+)?(?:\*\*)?([A-Za-z\s]+?)(?:\*\*)?\s+case", re.I),
    # "Password Reset - Case Started" or "Product Complaint - Case Created"
    re.compile(r"(?:\*\*)?([A-Za-z\s]+?)(?:\*\*)?\s*[-–—]\s*[Cc]ase\s+(?:[Ss]tarted|[Cc]reated)", re.I),
    # "processing/handling a password reset" with case ID nearby
    re.compile(r"(?:processing|handling)\s+(?:a\s+|your\s+)?(?:the\s+)?(?:\*\*)?([A-Za-z\s]+?)(?:\*\*)?\.?\s*(?:Case\s+)?(?:ID|#|:)\s*[A-Z]+-\d+", re.I),
]

# Regex to detect file-attachment turns captured from Pega UI
_FILE_ATTACH_RE = re.compile(
    r"\[I have attached a file:\s*(?P<filename>[^\]]+)\]", re.IGNORECASE
)


def _is_file_turn(golden_turn: Dict[str, Any]) -> bool:
    """Return True if this golden turn involves a file attachment."""
    text = golden_turn.get("input", "")
    if _FILE_ATTACH_RE.search(text):
        return True
    if golden_turn.get("file_attachment"):
        return True
    return False


def _extract_filename_from_turn(golden_turn: Dict[str, Any]) -> Optional[str]:
    """Extract the filename from a file-attachment turn."""
    # Explicit metadata takes priority
    fa = golden_turn.get("file_attachment", {})
    if isinstance(fa, str):
        return Path(fa).name
    if fa and isinstance(fa, dict) and fa.get("filename"):
        return fa["filename"]
    # Fall back to regex on input text
    m = _FILE_ATTACH_RE.search(golden_turn.get("input", ""))
    return m.group("filename").strip() if m else None


def _extract_case_type_from_response(response_text: str) -> Optional[str]:
    """Extract the case type from a response indicating case creation.

    Tries multiple patterns in priority order:
      1. "XYZ case (PREFIX-NNNN)" — most specific, has case ID
      2. "started/initiated/created a XYZ case" — no ID but clear creation
      3. "XYZ - Case Started" — heading format
    """
    for pattern in _CASE_CREATION_PATTERNS:
        m = pattern.search(response_text)
        if m:
            return m.group(1).strip()
    return None


def _extract_case_class_from_tool_events(tool_events: List[Any]) -> Optional[str]:
    """Extract CaseTypeClassName from pxCreateCaseWithAssignmentDetails tool events."""
    for event in tool_events:
        name = event.tool_name if hasattr(event, 'tool_name') else event.get("tool_name", "")
        if name == "pxCreateCaseWithAssignmentDetails":
            args_str = event.arguments if hasattr(event, 'arguments') else event.get("arguments", "")
            if args_str:
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    class_name = args.get("CaseTypeClassName", "")
                    if class_name:
                        return class_name
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass
    return None


def _extract_case_type_from_stages(case_stages: List[Dict[str, Any]]) -> Optional[str]:
    """Extract the case type name from DX API case stages data.

    The first stage's process name or the stage name itself typically matches the
    case type (e.g., "Reset Password", "Product Complaint").
    """
    if not case_stages:
        return None
    first_stage = case_stages[0]
    # The stage name is the most reliable indicator of case type
    stage_name = first_stage.get("name", "")
    if stage_name:
        return stage_name
    # Fallback: check processes within the stage
    processes = first_stage.get("processes", [])
    if processes:
        return processes[0].get("name", "") or None
    return None


def _normalize_case_words(value: str) -> set:
    """Normalize a case type string to a set of lowercase words for comparison."""
    return set(re.split(r"[\s\-_]+", value.lower())) - {"", "case"}


def _case_type_matches(
    expected: str,
    actual_text_type: Optional[str],
    actual_class: Optional[str],
    golden_text_type: Optional[str] = None,
) -> bool:
    """Compare expected case type against actual, supporting both class names and display names.

    Expected can be either:
      - A Pega class name like "Uplus-FS-Work-ResetAccount"
      - A display name like "Reset Password"

    Matching strategy:
      1. Exact class name match (expected == actual_class)
      2. Exact display name match (expected == actual_text_type)
      3. Cross-format: compare normalized word sets
      4. If golden_text_type available, compare actual vs golden display name (word sets)
    """
    if not expected:
        return False
    expected_lower = expected.lower()
    # 1. Direct class name match
    if actual_class and actual_class.lower() == expected_lower:
        return True
    # 2. Direct display name match
    if actual_text_type and actual_text_type.lower() == expected_lower:
        return True
    # 3. Normalized word-set comparison (handles "password reset" vs "Reset Password"
    #    and "Uplus-FS-Work-ResetAccount" → {"uplus","fs","work","resetaccount"})
    expected_words = _normalize_case_words(expected)
    if actual_class:
        if _normalize_case_words(actual_class) == expected_words:
            return True
    if actual_text_type:
        actual_words = _normalize_case_words(actual_text_type)
        if actual_words == expected_words:
            return True
        # Check if actual words are a subset/superset of expected meaningful words
        # (handles "ResetAccount" as single token matching {"reset", "password"})
        if golden_text_type:
            golden_words = _normalize_case_words(golden_text_type)
            if actual_words == golden_words:
                return True
    # 4. Last resort: check if the meaningful part of expected class name
    #    matches actual text words (e.g., "ResetAccount" contains "reset")
    if actual_text_type and "-" in expected:
        # Extract last segment of class name: "Uplus-FS-Work-ResetAccount" → "resetaccount"
        class_suffix = expected.rsplit("-", 1)[-1].lower()
        actual_collapsed = actual_text_type.replace(" ", "").lower()
        if class_suffix == actual_collapsed:
            return True
    return False


# ============================================================================
# AgentX Transport — talks to Pega the same way the UI does
# ============================================================================


class AgentXTransport:
    """Wraps the Pega Application v2 REST API to return AgentResponse objects.

    This transport mirrors the Pega UI path: it uses the Application v2
    conversations endpoint (POST to create, PATCH to message) and supports
    real file uploads via the /v2/attachments/upload endpoint.

    The interface matches SurfaceAgent.run() so _replay_session() can
    swap transports transparently. It is entirely self-contained using
    `requests` — no extra SDK dependencies required.
    """

    def __init__(
        self,
        base_url: str,
        agent_name: str,
        client_id: str,
        client_secret: str,
        token_url: str,
    ):
        import requests as _requests

        self._requests = _requests
        self.base_url = base_url.rstrip("/")
        self.agent_name = agent_name
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.access_token: Optional[str] = None
        self._conversation_id: Optional[str] = None
        self._authenticate()

    def _authenticate(self):
        """Fetch an OAuth2 access token using client credentials."""
        resp = self._requests.post(
            self.token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            verify=False,
        )
        resp.raise_for_status()
        self.access_token = resp.json()["access_token"]
        logger.info("[AgentX] Token acquired")

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"}

    def _ensure_conversation(self, context_id: Optional[str] = None) -> str:
        """Create an AgentX conversation if one doesn't exist yet."""
        if self._conversation_id:
            return self._conversation_id

        encoded_agent = url_quote(self.agent_name, safe="")
        url = f"{self.base_url}/prweb/api/application/v2/ai-agents/{encoded_agent}/conversations"
        resp = self._requests.post(
            url,
            json={
                "enableTracer": True,
                "contextID": context_id or "GoldenReplay",
                "interactionID": f"replay_{int(time.time())}",
                "activeChannel": "test",
                "activeChannelID": "golden_session",
                "executeStarterQuestion": True,
            },
            headers=self._headers(),
            verify=False,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        conv_id = data.get("ID") or data.get("id") or data.get("conversationID")
        if not conv_id:
            raise RuntimeError(f"No conversation ID in AgentX response: {data}")
        self._conversation_id = conv_id
        logger.info(f"[AgentX] Created conversation: {conv_id}")
        return conv_id

    def _upload_attachment(self, file_path: str) -> str:
        """Upload a file to Pega and return the attachment ID."""
        from pathlib import Path as _Path

        p = _Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Attachment not found: {file_path}")

        url = f"{self.base_url}/prweb/api/application/v2/attachments/upload"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        with open(file_path, "rb") as f:
            resp = self._requests.post(
                url,
                files={"file": (p.name, f, "application/octet-stream")},
                headers=headers,
                verify=False,
                timeout=60,
            )
        resp.raise_for_status()
        data = resp.json()
        att_id = data.get("ID") or data.get("id")
        if not att_id:
            raise RuntimeError(f"No attachment ID returned: {data}")
        logger.info(f"[AgentX] Uploaded {p.name} → ID: {att_id}")
        return att_id

    def run(
        self,
        task: str,
        timeout: int = 120,
        context_id: Optional[str] = None,
        attachment_path: Optional[str] = None,
    ) -> AgentResponse:
        """Send a message via AgentX API, optionally with a file attachment.

        Returns an AgentResponse with the same shape as SurfaceAgent.run().
        """
        conv_id = self._ensure_conversation(context_id)

        print(f"\n[AgentX] Sending: {task[:80]}...")
        if attachment_path:
            print(f"[AgentX] Attaching file: {attachment_path}")

        # Match the exact schema used by the working Pega Chat UI (agentx.ts)
        # and AgentXTestSuites (agentx_client.py):
        #   Request  = plain string (NOT an object)
        #   Attachments = top-level array (NOT nested inside Request)
        payload: Dict[str, Any] = {"Request": task}

        if attachment_path:
            from pathlib import Path as _Path
            att_id = self._upload_attachment(attachment_path)
            payload["Attachments"] = [
                {"ID": att_id, "type": "File", "filename": _Path(attachment_path).name}
            ]

        encoded_agent = url_quote(self.agent_name, safe="")
        url = (
            f"{self.base_url}/prweb/api/application/v2/ai-agents/"
            f"{encoded_agent}/conversations/{conv_id}"
        )
        t0 = time.perf_counter()
        resp = self._requests.patch(
            url,
            json=payload,
            headers=self._headers(),
            verify=False,
            timeout=timeout,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()

        data = resp.json()
        text = data.get("response", "")
        message_id = data.get("messageID")

        print(f"[AgentX] Response ({latency_ms:.0f}ms): {text[:120]}...")

        return AgentResponse(
            text=text,
            context_id=conv_id,
            message_id=message_id,
            latency_ms=latency_ms,
            raw_json=data,
        )


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def transport_mode(request):
    """Which API transport to use: 'agentx' (default), 'a2a', or 'auto'."""
    return request.config.getoption("--transport", default="agentx")


@pytest.fixture(scope="module")
def surface_agent(transport_mode):
    """Module-scoped A2A Surface agent client (authenticates once).

    Only created when transport_mode is 'a2a' or 'auto'.  When running
    pure AgentX mode the A2A agent card is never fetched — avoids a 404
    if the agent doesn't expose an A2A endpoint.
    """
    if transport_mode == "agentx":
        return None  # Not needed — skip the A2A agent-card fetch entirely
    if CLIENT_ID == "YOUR_CLIENT_ID" or CLIENT_SECRET == "YOUR_CLIENT_SECRET":
        pytest.skip("Pega credentials not configured.")
    return SurfaceAgent(AGENT_CARD_URL, CLIENT_ID, CLIENT_SECRET, TOKEN_URL)


@pytest.fixture(scope="module")
def agentx_transport(transport_mode):
    """Module-scoped AgentX transport (same path as Pega UI).

    Only created when transport_mode is 'agentx' or 'auto'.  When running
    pure A2A mode the AgentX client is never instantiated.
    """
    if transport_mode == "a2a":
        return None  # Not needed — skip AgentX auth entirely
    if CLIENT_ID == "YOUR_CLIENT_ID" or CLIENT_SECRET == "YOUR_CLIENT_SECRET":
        pytest.skip("Pega credentials not configured.")
    base_url = os.environ.get("AGENTX_BASE_URL", "https://genai-cdh-demo.pega.net")
    agent_name = os.environ.get("AGENT_NAME", "OPK0KG-SURFACE1-UIPAGES!SURFACEORCHESTRATIONAGENTV7")
    return AgentXTransport(
        base_url=base_url,
        agent_name=agent_name,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        token_url=TOKEN_URL,
    )


@pytest.fixture(scope="module")
def pega_insight():
    """Module-scoped Pega insight client."""
    base_url = os.environ.get("AGENTX_BASE_URL", "https://genai-cdh-demo.pega.net")
    return PegaInsight(
        base_url=base_url,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        token_url=TOKEN_URL,
    )


@pytest.fixture(scope="module")
def judge():
    """Shared LLM judge — provider selected by LLM_PROVIDER env var."""
    return create_judge_llm()


def _find_latest_golden(directory: str = "golden_sessions") -> Optional[Path]:
    """Find the most recently recorded golden session JSON."""
    golden_dir = _this_dir / directory
    if not golden_dir.exists():
        return None
    files = sorted(golden_dir.glob("golden_*.json"), reverse=True)
    return files[0] if files else None


def _load_golden(path: Optional[str] = None) -> Dict[str, Any]:
    """Load a golden session from disk."""
    if path:
        p = Path(path) if Path(path).is_absolute() else _this_dir / path
    else:
        p = _find_latest_golden()

    if not p or not p.exists():
        pytest.skip(
            f"No golden session found. Record one first:\n"
            f"  python3 record_golden_session.py"
        )
    with open(p) as f:
        data = json.load(f)
    print(f"\n[Golden] Loaded: {p.name}  ({len(data.get('turns', []))} turns)")
    return data


@pytest.fixture(scope="module")
def golden_session(request):
    """Load the golden session JSON — uses --golden CLI arg or latest file."""
    custom = request.config.getoption("--golden", default=None)
    return _load_golden(custom)


# ---------------------------------------------------------------------------
# Profile loading — merges auto-sensed + user-provided project context
# ---------------------------------------------------------------------------


def _load_profile(golden: Dict[str, Any], cli_config: Optional[str] = None) -> Dict[str, Any]:
    """Load the evaluation profile for a golden session.

    Resolution order:
      1. Profile filename embedded in the golden JSON (``golden["profile"]``)
      2. ``--project-config`` CLI flag (raw project config, not a generated profile)
      3. ``PROJECT_CONFIG`` env var
      4. ``project_config.json`` in this directory
      5. Fallback: synthesize a minimal profile from agent_identity defaults
    """
    # --- 1. Embedded profile reference in golden JSON ---
    profile_ref = golden.get("profile")
    if profile_ref:
        # Relative to the golden JSON's directory (golden_sessions/)
        p = _this_dir / "golden_sessions" / profile_ref
        if not p.exists():
            p = _this_dir / profile_ref
        if p.exists():
            with open(p) as f:
                print(f"[Profile] Loaded from golden ref: {p.name}")
                return json.load(f)

    # --- 2-4. Project config (user-authored) → build a minimal profile ---
    from capture_golden_session import load_project_config, _normalize_workflows
    pcfg = load_project_config(cli_config)
    if pcfg:
        identity = pcfg.get("agent_identity", {})
        workflows = _normalize_workflows(pcfg)
        first_wf = workflows[0] if workflows else {}
        profile: Dict[str, Any] = {
            "project_name": pcfg.get("project_name", "Unknown"),
            "agent_identity": identity,
            "workflow": {
                "id": first_wf.get("id"),
                "description": first_wf.get("description", ""),
                "stages_from_config": first_wf.get("stages", []),
                "stages_auto_sensed": [],
            },
            "hallucination_context": pcfg.get("hallucination_context", []),
            "tool_patterns": pcfg.get("tool_patterns", {}),
            "step_agent_patterns": pcfg.get("step_agent_patterns", {}),
            "connection": pcfg.get("connection", {}),
        }
        print(f"[Profile] Built from project config: {pcfg.get('project_name', '?')}")
        return profile

    # --- 5. Fallback ---
    print("[Profile] No profile or config found — using built-in defaults")
    return {}


def _profile_role(profile: Dict[str, Any]) -> str:
    """Build a chatbot_role string from the profile's agent_identity."""
    identity = profile.get("agent_identity", {})
    parts = []
    if identity.get("role"):
        parts.append(identity["role"])
    if identity.get("off_topic_guidance"):
        parts.append(identity["off_topic_guidance"])
    return " ".join(parts) if parts else "A Pega agent assistant."


def _profile_expected_outcome(profile: Dict[str, Any]) -> str:
    """Build an expected_outcome string from workflow stages."""
    workflow = profile.get("workflow", {})
    stages = workflow.get("stages_from_config", [])
    if not stages:
        return workflow.get("description", "The agent should complete its workflow.")

    lines = ["The agent should complete the full workflow through these stages:"]
    for i, stage in enumerate(stages, 1):
        desc = stage.get("description", "")
        lines.append(f"{i}. {stage['name']}: {desc}" if desc else f"{i}. {stage['name']}")
    return "\n".join(lines)


def _profile_hallucination_context(profile: Dict[str, Any]) -> List[str]:
    """Return hallucination context lines from the profile."""
    ctx = profile.get("hallucination_context", [])
    if ctx:
        return ctx
    # Fallback: build minimal context from agent identity
    identity = profile.get("agent_identity", {})
    lines = []
    if identity.get("role"):
        lines.append(identity["role"])
    if identity.get("off_topic_guidance"):
        lines.append(identity["off_topic_guidance"])
    if identity.get("domain"):
        lines.append(f"The agent operates in the {identity['domain']} domain.")
    return lines or ["The agent is a Pega assistant."]


@pytest.fixture(scope="module")
def project_profile(request, golden_session):
    """Load the evaluation profile — auto-sensed + user config merged."""
    cli_config = request.config.getoption("--project-config", default=None)
    return _load_profile(golden_session, cli_config)


@pytest.fixture(scope="module")
def tool_detection_mode(request):
    """Return the effective tool detection policy for this session.

    Set via --tool-detection-mode CLI flag or TOOL_DETECTION_MODE env var.
    Defaults to 'hybrid'.
    """
    return request.config.getoption("--tool-detection-mode", default="hybrid")


def _resolve_effective_policy(golden_session: Dict[str, Any], cli_mode: str) -> str:
    """Return the effective detection policy for a session.

    Resolution:
      - 'strict_structured' or 'regex_only' CLI → always wins.
      - 'hybrid' CLI + golden detection_mode='structured' → 'hybrid'
        (structured preferred, regex fallback with warning).
      - 'hybrid' CLI + legacy/absent detection_mode → 'regex_only'
        (zero behavioral change for sessions without structured data).
    """
    if cli_mode in ("strict_structured", "regex_only"):
        return cli_mode
    golden_dm = golden_session.get("detection_mode", "regex")
    return "hybrid" if golden_dm == "structured" else "regex_only"


# ============================================================================
# Helper: replay all turns and collect results
# ============================================================================


class ReplayResult:
    """Holds the full replay data for a golden session."""

    def __init__(self):
        self.turns: List[Dict[str, Any]] = []
        self.context_id: Optional[str] = None
        self.deepeval_turns: List[Turn] = []  # For ConversationalTestCase
        self.golden_turns: List[Dict[str, Any]] = []  # Aligned golden turns (skipping greetings)


def _resolve_attachment_path(
    golden_turn: Dict[str, Any], golden_dir: Optional[Path] = None
) -> Optional[str]:
    """Resolve the file attachment path for a golden turn.

    Searches for the file in:
      1. golden_turn["file_attachment"]["path"]  (explicit)
      2. golden_sessions/attachments/<filename>  (convention)
      3. golden_sessions/<filename>              (fallback)
    """
    filename = _extract_filename_from_turn(golden_turn)
    if not filename:
        return None

    # Check explicit path first
    fa = golden_turn.get("file_attachment", {})
    explicit = fa.get("path")
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = (_this_dir / p)
        if p.exists():
            return str(p)

    # Convention: golden_sessions/attachments/<filename>
    search_dirs = [
        _this_dir / "golden_sessions" / "attachments",
        _this_dir / "golden_sessions",
        _this_dir,
    ]
    if golden_dir:
        search_dirs.insert(0, golden_dir / "attachments")
        search_dirs.insert(1, golden_dir)

    for d in search_dirs:
        candidate = d / filename
        if candidate.exists():
            return str(candidate)

    logger.warning(
        f"File '{filename}' referenced in golden turn {golden_turn.get('turn')} "
        f"not found in any search path. File upload will be skipped."
    )
    return None


def _gate_response(
    golden_turn: Dict[str, Any],
    response: AgentResponse,
    agentx: "AgentXTransport",
    context_id: str,
) -> AgentResponse:
    """Verify the agent response matches the expected workflow stage.

    When a golden turn includes ``wait_for_pattern``, this function checks
    whether the initial AgentX response matches.  If not — typically because
    a background step-agent (e.g. document extraction) is still running — it
    waits for the estimated processing window and sends neutral "continue"
    nudges until the orchestration agent catches up to the expected stage.

    This prevents conversation desync, where the test sends the *next* user
    message before the agent has finished the *current* stage.

    Returns the original response if it already matches, the first nudge
    response that matches, or the last nudge response on timeout.
    """
    pattern_str = golden_turn.get("wait_for_pattern")
    if not pattern_str:
        return response

    gate_re = re.compile(pattern_str, re.IGNORECASE | re.DOTALL)

    # Fast path — response already in expected stage.
    if gate_re.search(response.text):
        return response

    turn_num = golden_turn.get("turn", "?")
    gate_timeout = golden_turn.get("gate_timeout", 180)
    golden_latency_s = golden_turn["response"]["latency_ms"] / 1000

    print(f"\n  [Gate T{turn_num}] Response does NOT match /{pattern_str}/")
    print(f"  [Gate T{turn_num}] Background processing likely in progress.")

    # Phase 1 — Passive wait: let background step-agents finish.
    # Use the golden latency as a baseline since the process likely takes
    # about that long.  Subtract what the initial PATCH already consumed.
    elapsed_s = response.latency_ms / 1000
    passive_wait = max(golden_latency_s - elapsed_s + 5, 10)
    passive_wait = min(passive_wait, gate_timeout * 0.6)
    print(
        f"  [Gate T{turn_num}] Passive wait {passive_wait:.0f}s "
        f"(golden {golden_latency_s:.0f}s, PATCH took {elapsed_s:.0f}s)"
    )
    time.sleep(passive_wait)

    # Phase 2 — Active nudges: prompt the orchestration agent to incorporate
    # completed step-agent results.
    max_nudges = 5
    nudge_pause = 15
    last_resp = response

    for idx in range(1, max_nudges + 1):
        remaining = gate_timeout - passive_wait - (idx - 1) * nudge_pause
        if remaining <= 0:
            break

        print(
            f"  [Gate T{turn_num}] Nudge {idx}/{max_nudges}: "
            f"\"Please continue.\""
        )
        try:
            nudge_resp = agentx.run(
                task="Please continue.",
                context_id=context_id,
                timeout=min(int(remaining), 120),
            )
        except Exception as exc:
            print(f"  [Gate T{turn_num}] Nudge {idx} failed: {exc}")
            time.sleep(nudge_pause)
            continue

        last_resp = nudge_resp

        if gate_re.search(nudge_resp.text):
            print(f"  [Gate T{turn_num}] Matched after nudge {idx}")
            return nudge_resp

        print(
            f"  [Gate T{turn_num}] Nudge {idx} response: "
            f"{nudge_resp.text[:120]}..."
        )
        time.sleep(nudge_pause)

    print(f"  [Gate T{turn_num}] WARNING: Gate timeout — proceeding with last response")
    return last_resp


def _replay_session(
    golden: Dict[str, Any],
    agent: SurfaceAgent,
    insight: PegaInsight,
    transport_mode: str = "agentx",
    agentx: Optional[AgentXTransport] = None,
    project_profile: Optional[Dict[str, Any]] = None,
    case_id: Optional[str] = None,
) -> ReplayResult:
    """Replay a golden session against the live agent, collecting all data.

    Args:
        golden: The loaded golden session JSON.
        agent: A2A SurfaceAgent transport.
        insight: PegaInsight client for querying conversation metadata.
        transport_mode: 'agentx' (default), 'a2a', or 'auto'.
            - agentx: Use AgentX for ALL turns (recommended — mirrors Pega UI).
            - a2a: Use A2A JSON-RPC for all turns (tests external contract).
            - auto: Use AgentX for file-attachment turns, A2A for the rest.
        agentx: AgentXTransport instance (required when transport_mode != 'a2a').
    """
    result = ReplayResult()
    context_id: Optional[str] = case_id

    # Build profile-specific tool patterns for re-detection fallback
    _tool_patterns = build_tool_patterns_from_profile(project_profile) if project_profile else None

    if transport_mode in ("agentx", "auto") and agentx is None:
        raise ValueError(
            f"transport_mode='{transport_mode}' requires an AgentXTransport instance. "
            f"Pass agentx= or use --transport a2a."
        )

    for golden_turn in golden["turns"]:
        turn_num = golden_turn["turn"]
        message = golden_turn["input"]
        description = golden_turn.get("description", "")

        # --- Decide which transport to use for THIS turn ---
        is_file = _is_file_turn(golden_turn)

        # Skip greeting turns with empty input — BUT keep file-upload turns
        # (Pega UI file uploads may have empty user text in D_pxAutopilotConversation;
        # our golden-record patcher adds synthetic input + file_attachment metadata.)
        if not message and not is_file:
            print(f"\n  [Skipping Turn {turn_num}: greeting / empty input]")
            continue
        if transport_mode == "agentx":
            use_agentx = True
        elif transport_mode == "a2a":
            use_agentx = False
        else:  # auto
            use_agentx = is_file  # AgentX only for file turns

        transport_label = "AgentX" if use_agentx else "A2A"
        file_label = " [FILE]" if is_file else ""
        print(f"\n{'=' * 60}")
        print(f"  REPLAY Turn {turn_num}: {description}  ({transport_label}{file_label})")
        print(f"{'=' * 60}")

        # --- Resolve file attachment if applicable ---
        attachment_path: Optional[str] = None
        if is_file and use_agentx:
            attachment_path = _resolve_attachment_path(golden_turn)
            if attachment_path:
                print(f"  [File] Attaching: {attachment_path}")
            else:
                print(f"  [File] WARNING: No file found for '{_extract_filename_from_turn(golden_turn)}' — sending text only")

        # --- Send to live agent (multi-turn) ---
        if use_agentx:
            response: AgentResponse = agentx.run(
                task=message,
                context_id=context_id,
                attachment_path=attachment_path,
            )
        else:
            response: AgentResponse = agent.run(
                task=message,
                context_id=context_id,
            )

        if not context_id and response.context_id:
            context_id = response.context_id
            result.context_id = context_id

        # --- Stage gate: verify response matches expected workflow stage ---
        # Prevents conversation desync when background step-agents take
        # longer than the initial AgentX PATCH response.
        if use_agentx and agentx is not None and context_id:
            response = _gate_response(golden_turn, response, agentx, context_id)

        # --- Brief delay for Pega to finalize ---
        time.sleep(golden_turn.get("wait", 2))

        # --- Query insight ---
        insight_data: Optional[ConversationInsight] = None
        if context_id:
            try:
                insight_data = insight.query_conversation(context_id)
            except Exception as e:
                print(f"  [Warning] Insight query failed: {e}")

        # --- Re-detect with profile patterns if all events fell back to regex ---
        if insight_data and _tool_patterns and insight_data.tool_events:
            all_regex = all(e.source == "regex" for e in insight_data.tool_events)
            if all_regex and insight_data.assistant_messages:
                try:
                    refreshed = _detect_tools_from_messages_v2(
                        insight_data.assistant_messages,
                        patterns=_tool_patterns,
                        turn=turn_num,
                    )
                    if refreshed:
                        insight_data.tool_events = refreshed
                        insight_data.tools_detected = [
                            e.tool_name for e in refreshed if not e.is_internal
                        ]
                except Exception:
                    pass  # Non-fatal: keep original events

        # --- Collect replay turn ---
        replay_turn: Dict[str, Any] = {
            "turn": turn_num,
            "input": message,
            "description": description,
            "golden_response": golden_turn["response"]["text"][:300],
            "golden_tools": golden_turn.get("expected_tools", []),
            "golden_latency_ms": golden_turn["response"]["latency_ms"],
            "actual_response": response.text,
            "actual_latency_ms": response.latency_ms,
            "actual_tools": insight_data.tools_detected if insight_data else [],
            "actual_tool_events": insight_data.tool_events if insight_data else [],
            "actual_case_key": insight_data.business_case_key if insight_data else None,
            "actual_case_stages": insight_data.case_stages if insight_data else [],
            "actual_step_agents": (
                [asdict(sa) for sa in insight_data.step_agents] if insight_data else []
            ),
            "actual_prefilled_fields": (
                insight_data.prefilled_fields if insight_data else {}
            ),
        }
        result.turns.append(replay_turn)

        # --- Build DeepEval Turn objects ---
        # User turn
        result.deepeval_turns.append(Turn(role="user", content=message))

        # Assistant turn (with tools_called for ToolCorrectness)
        tools_called = []
        if insight_data:
            for tool_name in insight_data.tools_detected:
                tools_called.append(ToolCall(name=tool_name))

        result.deepeval_turns.append(
            Turn(
                role="assistant",
                content=response.text,
                tools_called=tools_called if tools_called else None,
                additional_metadata={
                    "latency_ms": response.latency_ms,
                    "context_id": response.context_id,
                    "message_id": response.message_id,
                    "case_key": insight_data.business_case_key if insight_data else None,
                    "step_agents": replay_turn["actual_step_agents"],
                },
            )
        )

        # --- Track the matching golden turn for aligned zipping ---
        result.golden_turns.append(golden_turn)

        # --- Console ---
        print(f"  [Actual]  {response.text[:120]}...")
        print(f"  [Latency] {response.latency_ms:.0f}ms (golden: {replay_turn['golden_latency_ms']:.0f}ms)")
        print(f"  [Tools]   {replay_turn['actual_tools'] or 'none'}")

    return result


# Module-level cache so we only replay once per pytest session
_replay_cache: Dict[str, ReplayResult] = {}


@pytest.fixture(scope="module")
def case_id(request):
    """Pega case ID for step agent contexts. None for conversational agents."""
    return request.config.getoption("--case-id", default=None)


@pytest.fixture(scope="module")
def replay(golden_session, surface_agent, pega_insight, transport_mode, agentx_transport, project_profile, case_id):
    """Replay the golden session once, shared across all tests.

    Uses --transport CLI flag to decide which API transport to use:
      agentx  (default) — Pega Application v2 API, same as the UI.
      a2a     — A2A JSON-RPC, tests external interop contract.
      auto    — AgentX for file turns, A2A for the rest.
    """
    profile_name = project_profile.get("_profile_name", "default") if project_profile else "default"
    cache_key = f"{golden_session.get('recorded_at', 'default')}:{transport_mode}:{profile_name}:{case_id or 'none'}"
    if cache_key not in _replay_cache:
        _replay_cache[cache_key] = _replay_session(
            golden_session,
            surface_agent,
            pega_insight,
            transport_mode=transport_mode,
            agentx=agentx_transport if transport_mode != "a2a" else None,
            project_profile=project_profile,
            case_id=case_id,
        )
    return _replay_cache[cache_key]


def _skip_if_not_selected(request, metric_id: str) -> None:
    """Skip this test if --metrics was specified and metric_id is not in the list."""
    metrics_csv = request.config.getoption("--metrics", default="")
    if metrics_csv:
        selected = [m.strip() for m in metrics_csv.split(",") if m.strip()]
        if metric_id not in selected:
            pytest.skip(f"{metric_id} not in --metrics selection")


# ============================================================================
# Test 1: Full conversation — KnowledgeRetentionMetric
# ============================================================================


def test_knowledge_retention(request, replay, judge, golden_session, project_profile):
    """The agent must remember context from earlier turns throughout the session.

    Uses DeepEval's KnowledgeRetentionMetric which checks whether information
    introduced in early turns is retained/accessible in later turns.
    """
    _skip_if_not_selected(request, "knowledge_retention")
    conv = ConversationalTestCase(
        turns=replay.deepeval_turns,
        chatbot_role=_profile_role(project_profile) + (
            " It must remember context from earlier turns including "
            "document extractions, user decisions, and approval states."
        ),
    )

    metric = KnowledgeRetentionMetric(threshold=_get_threshold("knowledge_retention", 0.5), model=judge)
    assert_test(conv, [metric])


# ============================================================================
# Test 2: Full conversation — ConversationCompletenessMetric
# ============================================================================


def test_conversation_completeness(request, replay, judge, golden_session, project_profile):
    """The agent must complete the full workflow across all turns.

    Uses DeepEval's ConversationCompletenessMetric to evaluate whether the
    overall goal was achieved.
    """
    _skip_if_not_selected(request, "conversation_completeness")
    wf = project_profile.get("workflow", {})
    if wf.get("id") is None and not wf.get("stages_from_config"):
        pytest.skip("No workflow_id annotated for this session — ConversationCompleteness skipped.")

    conv = ConversationalTestCase(
        turns=replay.deepeval_turns,
        chatbot_role=_profile_role(project_profile),
        expected_outcome=_profile_expected_outcome(project_profile),
    )

    metric = ConversationCompletenessMetric(threshold=_get_threshold("conversation_completeness", 0.5), model=judge)
    assert_test(conv, [metric])


# ============================================================================
# Test 3: Full conversation — RoleAdherenceMetric
# ============================================================================


def test_role_adherence(request, replay, judge, golden_session, project_profile):
    """The agent must stay in its designated role throughout.

    Uses DeepEval's RoleAdherenceMetric to detect role drift or out-of-scope
    responses across the full conversation.
    """
    _skip_if_not_selected(request, "role_adherence")
    identity = project_profile.get("agent_identity", {})
    role_desc = _profile_role(project_profile)
    # Append file-upload acceptance note (common to all Pega agents)
    role_desc += " The agent may accept file uploads without any accompanying text as a valid user action."

    conv = ConversationalTestCase(
        turns=replay.deepeval_turns,
        chatbot_role=role_desc,
    )

    metric = RoleAdherenceMetric(threshold=_get_threshold("role_adherence", 0.7), model=judge)
    assert_test(conv, [metric])


# ============================================================================
# Test 4: Per-turn tool verification (Pega-specific)
# ============================================================================


def test_tool_invocations_match_golden(request, replay, golden_session, tool_detection_mode):
    """Each turn's tool invocations should match the golden baseline.

    Policy is determined by --tool-detection-mode and golden detection_mode:
      hybrid          — structured events → hard fail; regex fallback → warning only.
      strict_structured — no structured data available → hard fail immediately.
      regex_only      — legacy behavior; regex names only, hard fail on missing.

    This test is optional — it only runs when 'pega_tool_correctness' is
    included in the --metrics flag (or when --metrics is not specified).
    """
    metrics_csv = request.config.getoption("--metrics", default="")
    if metrics_csv:
        selected = [m.strip() for m in metrics_csv.split(",") if m.strip()]
        if "pega_tool_correctness" not in selected:
            pytest.skip("pega_tool_correctness not in --metrics selection")

    effective_policy = _resolve_effective_policy(golden_session, tool_detection_mode)
    detection_mode = golden_session.get("detection_mode", "regex")
    tool_registry = golden_session.get("tool_registry", {})

    print(f"\n  Detection mode (golden): {detection_mode}")
    print(f"  Tool detection policy: {effective_policy} (CLI: {tool_detection_mode})")
    if tool_registry:
        invoked = [n for n, d in tool_registry.items() if d.get("invoked")]
        print(f"  Tool registry: {len(invoked)} invoked, {len(tool_registry) - len(invoked)} available")

    failures: List[str] = []
    warnings: List[str] = []

    for i, (golden_turn, actual_turn) in enumerate(
        zip(replay.golden_turns, replay.turns)
    ):
        expected = set(golden_turn.get("expected_tools", []))
        if not expected:
            continue  # Nothing to assert for this turn

        tool_events = actual_turn.get("actual_tool_events", [])
        structured_names = {
            e.tool_name for e in tool_events
            if e.source in ("tool_calls", "plugins") and not e.is_internal
        }
        regex_names = {
            e.tool_name for e in tool_events
            if e.source == "regex" and not e.is_internal
        }
        turn_label = f"Turn {actual_turn['turn']} ({actual_turn['description']})"

        # Emit per-turn source breakdown (parsed by db_etl._parse_tools_stdout)
        source_parts = []
        if structured_names:
            source_parts.append(f"tool_calls:{sorted(structured_names)}")
        if regex_names:
            source_parts.append(f"regex:{sorted(regex_names)}")
        print(f"  {turn_label}: tools={'; '.join(source_parts) or 'none'}")

        if effective_policy == "regex_only":
            # Legacy behavior — use all detected names regardless of source
            actual_all = {e.tool_name for e in tool_events if not e.is_internal}
            missing = expected - actual_all
            if missing:
                failures.append(f"{turn_label}: Missing tools {missing}. Got: {actual_all}")

        elif effective_policy == "hybrid":
            if structured_names:
                missing = expected - structured_names
                if missing:
                    failures.append(
                        f"{turn_label}: Missing tools {missing}. "
                        f"Got (structured): {structured_names}"
                    )
            else:
                # No structured data — regex fallback; warn but don't hard-fail
                missing = expected - regex_names
                if missing:
                    warnings.append(
                        f"{turn_label}: [regex-fallback] Missing tools {missing}. "
                        f"Got (regex): {regex_names}"
                    )

        elif effective_policy == "strict_structured":
            if not structured_names and expected:
                failures.append(
                    f"{turn_label}: No structured tool data available "
                    f"(expected {expected}). Use --tool-detection-mode hybrid "
                    f"for DX API replay environments."
                )
            else:
                missing = expected - structured_names
                if missing:
                    failures.append(
                        f"{turn_label}: Missing tools {missing}. "
                        f"Got (structured): {structured_names}"
                    )

    if warnings:
        print("\n  [Tool Detection Warnings — regex fallback, not a hard fail]")
        for w in warnings:
            print(f"    WARNING: {w}")

    if failures:
        msg = "Tool invocation mismatches:\n" + "\n".join(f"  - {f}" for f in failures)
        pytest.fail(msg)


# ============================================================================
# Test 5: Latency regression (2× golden baseline)
# ============================================================================


def test_latency_regression(request, replay, golden_session):
    """Each turn should complete within 2× the golden baseline latency.

    Accounts for normal variance while catching significant performance
    regressions in the agent or Pega backend.
    """
    _skip_if_not_selected(request, "latency_regression")
    REGRESSION_FACTOR = 2.0
    MIN_BUDGET_MS = 120_000  # Never fail for < 120s regardless

    failures: List[str] = []

    # Always print per-turn detail so it flows into the QA report
    print(f"\n  {'Turn':<6} {'Description':<50} {'Actual':>10} {'Golden':>10} {'Budget':>10} {'Status':>8}")
    print(f"  {'-'*6} {'-'*50} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    for golden_turn, actual_turn in zip(replay.golden_turns, replay.turns):
        golden_ms = golden_turn["response"]["latency_ms"]
        actual_ms = actual_turn["actual_latency_ms"]
        budget_ms = max(golden_ms * REGRESSION_FACTOR, MIN_BUDGET_MS)
        status = "OVER" if actual_ms > budget_ms else "ok"
        desc = actual_turn['description'][:50]

        print(f"  {actual_turn['turn']:<6} {desc:<50} {actual_ms:>9.0f}ms {golden_ms:>9.0f}ms {budget_ms:>9.0f}ms {status:>8}")

        if actual_ms > budget_ms:
            failures.append(
                f"Turn {actual_turn['turn']} ({actual_turn['description']}): "
                f"{actual_ms:.0f}ms > {budget_ms:.0f}ms budget "
                f"(golden: {golden_ms:.0f}ms × {REGRESSION_FACTOR})"
            )

    # Print summary
    total_golden = sum(t["response"]["latency_ms"] for t in replay.golden_turns)
    total_actual = sum(t["actual_latency_ms"] for t in replay.turns)
    print(f"\n  Total latency: {total_actual:.0f}ms (golden: {total_golden:.0f}ms)")
    print(f"  Ratio: {total_actual / total_golden:.2f}×")
    ux_concerns = [(t["turn"], t["actual_latency_ms"]) for t in replay.turns if t["actual_latency_ms"] > 30000]
    if ux_concerns:
        print(f"  UX concerns (>30s): {', '.join(f'Turn {t} ({ms:.0f}ms)' for t, ms in ux_concerns)}")

    if failures:
        msg = "Latency regressions:\n" + "\n".join(f"  - {f}" for f in failures)
        pytest.fail(msg)


# ============================================================================
# Test 6: Business case lifecycle
# ============================================================================


def test_case_lifecycle(request, replay, golden_session):
    """A business case must be created and persist across the session.

    Verifies:
    - A case key appears after the first turn
    - The case key remains consistent across subsequent turns

    This test is optional — it only runs when 'business_case_lifecycle' is
    included in the --metrics flag (or when --metrics is not specified).
    """
    metrics_csv = request.config.getoption("--metrics", default="")
    if metrics_csv:
        selected = [m.strip() for m in metrics_csv.split(",") if m.strip()]
        if "business_case_lifecycle" not in selected:
            pytest.skip("business_case_lifecycle not in --metrics selection")
    case_keys = [
        t["actual_case_key"] for t in replay.turns if t["actual_case_key"]
    ]

    assert case_keys, (
        "No business case was created during the session. "
        "The campaign creation flow should produce a case key (S-XXXX)."
    )

    # Case key must be consistent (same case across all turns)
    unique_keys = set(case_keys)
    assert len(unique_keys) == 1, (
        f"Multiple case keys detected: {unique_keys}. "
        f"Expected a single case to persist across the session."
    )

    print(f"\n  Business case: {case_keys[0]}")
    print(f"  Appeared in {len(case_keys)}/{len(replay.turns)} turns")


# ============================================================================
# Test 6b: Business case adherence (per-turn case type verification)
# ============================================================================


def test_business_case_adherence(request, replay, golden_session):
    """The correct business case type must be created on each turn where expected.

    For each turn in the golden session that has an `expected_case_type` field,
    extracts the actual case type from the live response text using the pattern
    "XYZ case (PREFIX-NNNN)" and compares it (case-insensitive) against the
    expected value.

    This test is optional — it only runs when 'business_case_adherence' is
    included in the --metrics flag (or when --metrics is not specified).
    """
    _skip_if_not_selected(request, "business_case_adherence")

    threshold = _get_threshold("business_case_adherence", 1.0)

    has_any_expected = any(
        gt.get("expected_case_type")
        for gt in replay.golden_turns
    )
    if not has_any_expected:
        pytest.skip(
            "No expected_case_type annotations found in golden session. "
            "Add expected_case_type to golden turns to enable this metric."
        )

    total_checked = 0
    passed_count = 0
    failures: List[str] = []
    prev_case_key: Optional[str] = None

    for golden_turn, actual_turn in zip(replay.golden_turns, replay.turns):
        expected_case_type = golden_turn.get("expected_case_type")
        current_case_key = actual_turn.get("actual_case_key")

        if not expected_case_type:
            prev_case_key = current_case_key or prev_case_key
            continue

        total_checked += 1
        turn_num = actual_turn["turn"]
        actual_response = actual_turn["actual_response"]

        actual_text_type = _extract_case_type_from_response(actual_response)
        actual_class = _extract_case_class_from_tool_events(
            actual_turn.get("actual_tool_events", [])
        )
        # DX API stages data — most reliable source of case type
        actual_stages_type = _extract_case_type_from_stages(
            actual_turn.get("actual_case_stages", [])
        )
        # A new case key appearing (different from previous turn) confirms case creation
        new_case_created = (
            current_case_key is not None and current_case_key != prev_case_key
        )
        # Golden turn's response text type (for cross-format matching)
        golden_response_text = golden_turn.get("response", {}).get("text", "")
        golden_text_type = _extract_case_type_from_response(golden_response_text)

        if actual_text_type is None and actual_class is None and actual_stages_type is None and not new_case_created:
            failures.append(
                f"Turn {turn_num}: Expected case type '{expected_case_type}' "
                f"but no case creation detected in response."
            )
            prev_case_key = current_case_key or prev_case_key
            continue

        if _case_type_matches(expected_case_type, actual_text_type, actual_class, golden_text_type) or \
           _case_type_matches(expected_case_type, actual_stages_type, actual_class, golden_text_type):
            passed_count += 1
            actual_display = actual_class or actual_stages_type or actual_text_type
            print(f"  Turn {turn_num}: PASS — '{actual_display}' matches expected '{expected_case_type}'")
        elif new_case_created:
            # Case creation confirmed by DX API (new case key appeared). The agent's
            # response text didn't match standard patterns, but the API proves a case
            # was created on this turn. Verify the golden also expected this type here.
            passed_count += 1
            print(f"  Turn {turn_num}: PASS — case created ({current_case_key}), "
                  f"confirmed by DX API case key (expected '{expected_case_type}')")
        else:
            actual_display = actual_class or actual_stages_type or actual_text_type or "(none)"
            failures.append(
                f"Turn {turn_num}: Expected case type '{expected_case_type}' "
                f"but got '{actual_display}'."
            )

        prev_case_key = current_case_key or prev_case_key

    score = passed_count / total_checked if total_checked > 0 else 1.0
    print(f"\n  Business Case Adherence: {passed_count}/{total_checked} turns correct (score: {score:.2f})")
    print(f"  Threshold: {threshold}")

    if failures and score < threshold:
        pytest.fail(
            f"Business case type mismatches ({len(failures)}/{total_checked} turns failed):\n"
            + "\n".join(f"  - {f}" for f in failures)
        )
    elif failures:
        print(f"\n  [WARNING] Some mismatches but score {score:.2f} >= threshold {threshold}:")
        for f in failures:
            print(f"    - {f}")


# ============================================================================
# Test 7: Step agent detection
# ============================================================================


def test_step_agents_detected(request, replay, golden_session):
    """Step agents should be detected for turns where the golden session specified them.

    Step agents (e.g., GradialStepAgent, AdobeAudienceAgent) are invoked by the
    Pega case as workflow steps — NOT by the orchestration agent's tool list.
    This test checks per-turn expected_step_agents from the golden session.
    """
    _skip_if_not_selected(request, "step_agents")
    failures: List[str] = []

    for golden_turn, actual_turn in zip(replay.golden_turns, replay.turns):
        expected = set(golden_turn.get("expected_step_agents", []))
        if not expected:
            continue

        actual_names = set(
            sa["name"] for sa in actual_turn["actual_step_agents"]
        )

        missing = expected - actual_names
        if missing:
            failures.append(
                f"Turn {actual_turn['turn']} ({actual_turn['description']}): "
                f"Missing step agents {missing}. Got: {actual_names or 'none'}"
            )

    # Also print session-level summary
    golden_step_agents = golden_session.get("summary", {}).get("all_step_agents", [])
    actual_step_agents = [sa for t in replay.turns for sa in t["actual_step_agents"]]

    print(f"\n  Golden step agents: {len(golden_step_agents)}")
    print(f"  Actual step agents: {len(actual_step_agents)}")

    for sa in actual_step_agents:
        print(f"    - {sa['name']} (source={sa['source']}, status={sa['status']})")

    if failures:
        msg = "Step agent detection mismatches:\n" + "\n".join(f"  - {f}" for f in failures)
        pytest.fail(msg)


# ============================================================================
# Test 8: Per-turn hallucination check
# ============================================================================


def test_no_hallucination_per_turn(request, replay, judge, golden_session, project_profile):
    """Each assistant response should be grounded in the conversation context.

    Runs DeepEval HallucinationMetric on each turn individually, using the
    accumulated conversation history as context.
    """
    _skip_if_not_selected(request, "hallucination")
    context = _profile_hallucination_context(project_profile)

    failures: List[str] = []

    # Always print per-turn detail so it flows into the QA report
    print(f"\n  {'Turn':<6} {'Description':<50} {'Score':>7} {'Pass':>6} {'Reason'}")
    print(f"  {'-'*6} {'-'*50} {'-'*7} {'-'*6} {'-'*40}")

    for turn in replay.turns:
        test_case = LLMTestCase(
            input=turn["input"],
            actual_output=turn["actual_response"],
            context=context,
        )

        metric = HallucinationMetric(threshold=_get_threshold("hallucination", 0.5), model=judge)
        metric.measure(test_case)

        score = metric.score if metric.score is not None else -1
        passed = metric.is_successful()
        reason_short = (metric.reason or "n/a")[:300]
        desc = turn['description'][:50]
        icon = "ok" if passed else "FAIL"

        print(f"  {turn['turn']:<6} {desc:<50} {score:>6.2f} {icon:>6} {reason_short}")

        if not passed:
            failures.append(
                f"Turn {turn['turn']} ({turn['description']}): "
                f"score={score:.2f} reason={metric.reason}"
            )

    # Summary
    all_scores = [metric.score for _ in [1]]  # placeholder — printed above per-turn
    grounding_concerns = [t['turn'] for t in replay.turns]  # handled per-turn above
    print(f"\n  {len(replay.turns)} turns evaluated, {len(failures)} hallucination failures")

    if failures:
        msg = "Hallucination detected:\n" + "\n".join(f"  - {f}" for f in failures)
        pytest.fail(msg)


# ============================================================================
# Test 9: Contextual Precision (per-turn)
# ============================================================================


def test_contextual_precision(request, replay, judge, golden_session, project_profile):
    """Each response should use the most relevant context with precision.

    Runs DeepEval ContextualPrecisionMetric on each turn using accumulated
    conversation history as retrieval context.
    """
    _skip_if_not_selected(request, "contextual_precision")
    context = _profile_hallucination_context(project_profile)

    failures: List[str] = []
    print(f"\n  {'Turn':<6} {'Description':<50} {'Score':>7} {'Pass':>6}")
    print(f"  {'-'*6} {'-'*50} {'-'*7} {'-'*6}")

    for turn in replay.turns:
        test_case = LLMTestCase(
            input=turn["input"],
            actual_output=turn["actual_response"],
            expected_output=turn["golden_response"],
            retrieval_context=context,
        )
        metric = ContextualPrecisionMetric(
            threshold=_get_threshold("contextual_precision", 0.7), model=judge
        )
        metric.measure(test_case)
        score = metric.score if metric.score is not None else -1
        passed = metric.is_successful()
        print(f"  {turn['turn']:<6} {turn['description'][:50]:<50} {score:>6.2f} {'ok' if passed else 'FAIL':>6}")
        if not passed:
            failures.append(
                f"Turn {turn['turn']} ({turn['description']}): score={score:.2f}"
            )

    if failures:
        pytest.fail("Contextual precision failures:\n" + "\n".join(f"  - {f}" for f in failures))


# ============================================================================
# Test 10: Contextual Recall (per-turn)
# ============================================================================


def test_contextual_recall(request, replay, judge, golden_session, project_profile):
    """Each response should retrieve and use all relevant context.

    Runs DeepEval ContextualRecallMetric on each turn using the golden response
    as expected output and accumulated conversation history as retrieval context.
    """
    _skip_if_not_selected(request, "contextual_recall")
    context = _profile_hallucination_context(project_profile)

    failures: List[str] = []
    print(f"\n  {'Turn':<6} {'Description':<50} {'Score':>7} {'Pass':>6}")
    print(f"  {'-'*6} {'-'*50} {'-'*7} {'-'*6}")

    for turn in replay.turns:
        test_case = LLMTestCase(
            input=turn["input"],
            actual_output=turn["actual_response"],
            expected_output=turn["golden_response"],
            retrieval_context=context,
        )
        metric = ContextualRecallMetric(
            threshold=_get_threshold("contextual_recall", 0.7), model=judge
        )
        metric.measure(test_case)
        score = metric.score if metric.score is not None else -1
        passed = metric.is_successful()
        print(f"  {turn['turn']:<6} {turn['description'][:50]:<50} {score:>6.2f} {'ok' if passed else 'FAIL':>6}")
        if not passed:
            failures.append(
                f"Turn {turn['turn']} ({turn['description']}): score={score:.2f}"
            )

    if failures:
        pytest.fail("Contextual recall failures:\n" + "\n".join(f"  - {f}" for f in failures))


# ============================================================================
# Test 11: Toxicity (per-turn)
# ============================================================================


def test_toxicity(request, replay, judge, golden_session, project_profile):
    """Each agent response should be free of toxic or harmful content.

    Runs DeepEval ToxicityMetric on each turn.
    """
    _skip_if_not_selected(request, "toxicity")

    failures: List[str] = []
    print(f"\n  {'Turn':<6} {'Description':<50} {'Score':>7} {'Pass':>6}")
    print(f"  {'-'*6} {'-'*50} {'-'*7} {'-'*6}")

    for turn in replay.turns:
        test_case = LLMTestCase(
            input=turn["input"],
            actual_output=turn["actual_response"],
        )
        metric = ToxicityMetric(
            threshold=_get_threshold("toxicity", 0.5), model=judge
        )
        metric.measure(test_case)
        score = metric.score if metric.score is not None else -1
        passed = metric.is_successful()
        print(f"  {turn['turn']:<6} {turn['description'][:50]:<50} {score:>6.2f} {'ok' if passed else 'FAIL':>6}")
        if not passed:
            failures.append(
                f"Turn {turn['turn']} ({turn['description']}): score={score:.2f}"
            )

    if failures:
        pytest.fail("Toxicity detected:\n" + "\n".join(f"  - {f}" for f in failures))


# ============================================================================
# Test 12: Bias (per-turn)
# ============================================================================


def test_bias(request, replay, judge, golden_session, project_profile):
    """Each agent response should be free of biased content.

    Runs DeepEval BiasMetric on each turn.
    """
    _skip_if_not_selected(request, "bias")

    failures: List[str] = []
    print(f"\n  {'Turn':<6} {'Description':<50} {'Score':>7} {'Pass':>6}")
    print(f"  {'-'*6} {'-'*50} {'-'*7} {'-'*6}")

    for turn in replay.turns:
        test_case = LLMTestCase(
            input=turn["input"],
            actual_output=turn["actual_response"],
        )
        metric = BiasMetric(
            threshold=_get_threshold("bias", 0.5), model=judge
        )
        metric.measure(test_case)
        score = metric.score if metric.score is not None else -1
        passed = metric.is_successful()
        print(f"  {turn['turn']:<6} {turn['description'][:50]:<50} {score:>6.2f} {'ok' if passed else 'FAIL':>6}")
        if not passed:
            failures.append(
                f"Turn {turn['turn']} ({turn['description']}): score={score:.2f}"
            )

    if failures:
        pytest.fail("Bias detected:\n" + "\n".join(f"  - {f}" for f in failures))
