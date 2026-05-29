import pytest
import requests
import os
import json
import re
import uuid
import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric, BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval import assert_test
from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import quote

# Load environment variables from .env file (same directory as this script)
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path)

# --- LLM Judge base and provider implementations ---
from google import genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()


class _JudgeLLMBase(DeepEvalBaseLLM):
    """Shared JSON-processing helpers for all DeepEval judge LLMs."""

    _SYSTEM_INSTRUCTION = """You are an evaluation judge for conversational AI systems.
When asked to extract data, knowledge, or information and return JSON:
- The "data" field must be a string or a list of strings, NEVER a dictionary/object
- Good: {"data": "Customer wants to file a product complaint"}
- Good: {"data": ["Customer wants to file a product complaint", "Customer name is Connor"]}
- Bad: {"data": {"Customer Inquiry": "Product Complaint"}}
Always flatten structured information into descriptive strings."""

    def _fix_json_escapes(self, text: str) -> str:
        r"""Fix invalid JSON escape sequences that LLMs sometimes produce.

        Valid JSON escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
        Invalid escapes like \U, \x, \' etc. need to be double-escaped.
        """
        result = []
        i = 0
        while i < len(text):
            if text[i] == '\\' and i + 1 < len(text):
                next_char = text[i + 1]
                if next_char in r'"\\/bfnrt':
                    result.append(text[i:i+2])
                    i += 2
                elif next_char == 'u' and i + 5 < len(text):
                    hex_part = text[i+2:i+6]
                    if all(c in '0123456789abcdefABCDEF' for c in hex_part):
                        result.append(text[i:i+6])
                        i += 6
                    else:
                        result.append('\\\\')
                        i += 1
                else:
                    result.append('\\\\')
                    i += 1
            else:
                result.append(text[i])
                i += 1
        return ''.join(result)

    def _flatten_data_in_response(self, response_text: str) -> str:
        """Post-process response to flatten dict values in 'data' fields to strings.

        DeepEval's Knowledge model expects data: str | list[str], but LLMs often
        return data: dict. This method converts dicts to descriptive strings.
        Also strips markdown code blocks that LLMs sometimes wrap JSON in.
        """
        text = response_text.strip()

        # Strip markdown code blocks (```json ... ``` or ``` ... ```)
        if text.startswith('```'):
            lines = text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            text = '\n'.join(lines).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                text = self._fix_json_escapes(text)
                data = json.loads(text)
            except json.JSONDecodeError:
                return response_text
        except Exception:
            return response_text

        if isinstance(data, dict) and 'data' in data:
            if isinstance(data['data'], dict):
                items = [f"{k}: {v}" for k, v in data['data'].items()]
                data['data'] = ', '.join(items)
            elif isinstance(data['data'], list):
                flattened = []
                for item in data['data']:
                    if isinstance(item, dict):
                        items = [f"{k}: {v}" for k, v in item.items()]
                        flattened.append(', '.join(items))
                    else:
                        flattened.append(str(item))
                data['data'] = flattened
        return json.dumps(data)

    def supports_structured_outputs(self) -> bool:
        return False

    def supports_json_mode(self) -> bool:
        return False


class GeminiJudgeLLM(_JudgeLLMBase):
    """DeepEval judge that calls Google AI Gemini via API key."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self._model_name = model_name
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        self._client = genai.Client(api_key=api_key)

    def load_model(self):
        return self._client

    def generate(self, prompt: str, schema=None, *args, **kwargs) -> str:
        from google.genai import types
        config = types.GenerateContentConfig(
            systemInstruction=self._SYSTEM_INSTRUCTION,
        )
        resp = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=config,
        )
        return self._flatten_data_in_response(resp.text)

    async def a_generate(self, prompt: str, schema=None, *args, **kwargs) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.generate(prompt, schema=schema)
        )

    def get_model_name(self) -> str:
        return self._model_name


class BedrockJudgeLLM(_JudgeLLMBase):
    """DeepEval judge that calls AWS Bedrock (Claude, Titan, or Llama models).

    Supports two authentication methods, selected by AWS_AUTH_METHOD env var:
      - "access_keys"  (default): uses AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY
      - "sso_profile":            uses a named profile from ~/.aws/config
                                  (requires prior `aws sso login --profile <name>`)
    """

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        region: str = "us-east-1",
    ):
        self._model_id = model_id
        self._region = region

        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for AWS Bedrock support: pip install boto3"
            )

        auth_method = os.environ.get("AWS_AUTH_METHOD", "access_keys").lower()

        if auth_method == "sso_profile":
            profile = os.environ.get("AWS_PROFILE", "")
            if not profile:
                raise ValueError(
                    "AWS_PROFILE environment variable is required when "
                    "AWS_AUTH_METHOD=sso_profile"
                )
            session = boto3.Session(profile_name=profile)
            self._client = session.client(
                service_name="bedrock-runtime",
                region_name=region,
            )
        else:
            aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
            aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
            if not aws_access_key_id or not aws_secret_access_key:
                raise ValueError(
                    "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment "
                    "variables are required when AWS_AUTH_METHOD=access_keys"
                )
            self._client = boto3.client(
                service_name="bedrock-runtime",
                region_name=region,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )

    def load_model(self):
        return self._client

    def _call_anthropic(self, prompt: str) -> str:
        import json as _json
        body = _json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "system": self._SYSTEM_INSTRUCTION,
            "messages": [{"role": "user", "content": prompt}],
        })
        resp = self._client.invoke_model(
            modelId=self._model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        body = json.loads(resp["body"].read())
        return body["content"][0]["text"]

    def _call_amazon_titan(self, prompt: str) -> str:
        import json as _json
        body = _json.dumps({
            "inputText": f"{self._SYSTEM_INSTRUCTION}\n\n{prompt}",
            "textGenerationConfig": {"maxTokenCount": 4096, "temperature": 0.1},
        })
        resp = self._client.invoke_model(
            modelId=self._model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        body = json.loads(resp["body"].read())
        return body["results"][0]["outputText"]

    def _call_meta_llama(self, prompt: str) -> str:
        import json as _json
        full_prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{self._SYSTEM_INSTRUCTION}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )
        body = _json.dumps({
            "prompt": full_prompt,
            "max_gen_len": 4096,
            "temperature": 0.1,
        })
        resp = self._client.invoke_model(
            modelId=self._model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        body = json.loads(resp["body"].read())
        return body["generation"]

    def generate(self, prompt: str, schema=None, *args, **kwargs) -> str:
        if "anthropic" in self._model_id:
            raw = self._call_anthropic(prompt)
        elif "amazon" in self._model_id:
            raw = self._call_amazon_titan(prompt)
        elif "meta" in self._model_id:
            raw = self._call_meta_llama(prompt)
        else:
            # Default to Anthropic-style for unknown model prefixes
            raw = self._call_anthropic(prompt)
        return self._flatten_data_in_response(raw)

    async def a_generate(self, prompt: str, schema=None, *args, **kwargs) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.generate(prompt, schema=schema)
        )

    def get_model_name(self) -> str:
        return self._model_id


class OpenAIJudgeLLM(_JudgeLLMBase):
    """DeepEval judge that calls OpenAI via API key."""

    def __init__(self, model_name: str = "gpt-4o"):
        self._model_name = model_name
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)

    def load_model(self):
        return self._client

    def generate(self, prompt: str, schema=None, *args, **kwargs) -> str:
        response = self._client.chat.completions.create(
            model=self._model_name,
            temperature=0,
            messages=[
                {"role": "system", "content": self._SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
        )
        return self._flatten_data_in_response(response.choices[0].message.content or "")

    async def a_generate(self, prompt: str, schema=None, *args, **kwargs) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.generate(prompt, schema=schema)
        )

    def get_model_name(self) -> str:
        return self._model_name


class GitHubCopilotJudgeLLM(_JudgeLLMBase):
    """DeepEval judge that calls GitHub Copilot models via the OpenAI-compatible API."""

    def __init__(self, model_name: str = "gpt-4o"):
        self._model_name = model_name.lstrip("/")
        raw_token = os.environ.get("GITHUB_COPILOT_TOKEN", "")
        token = raw_token.encode("ascii", errors="ignore").decode("ascii").strip()
        if not token:
            raise ValueError("GITHUB_COPILOT_TOKEN environment variable is required")
        from openai import OpenAI
        self._client = OpenAI(
            api_key=token,
            base_url="https://models.github.ai/inference",
        )

    def load_model(self):
        return self._client

    def generate(self, prompt: str, schema=None, *args, **kwargs) -> str:
        response = self._client.chat.completions.create(
            model=self._model_name,
            temperature=0,
            messages=[
                {"role": "system", "content": self._SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
        )
        return self._flatten_data_in_response(response.choices[0].message.content or "")

    async def a_generate(self, prompt: str, schema=None, *args, **kwargs) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.generate(prompt, schema=schema)
        )

    def get_model_name(self) -> str:
        return self._model_name


def create_judge_llm() -> _JudgeLLMBase:
    """Factory: return the correct LLM judge based on the LLM_PROVIDER env var.

    Reads LLM_PROVIDER from environment (set in .env):
      - "gemini"  → GeminiJudgeLLM  (requires GEMINI_API_KEY)
      - "bedrock" → BedrockJudgeLLM
            AWS_AUTH_METHOD=access_keys (default):
              requires AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
            AWS_AUTH_METHOD=sso_profile:
              requires AWS_PROFILE (named profile in ~/.aws/config;
              run `aws sso login --profile <name>` before use)
            Both methods also use AWS_REGION and AWS_BEDROCK_MODEL_ID.
      - "openai"  → OpenAIJudgeLLM  (requires OPENAI_API_KEY; OPENAI_MODEL_ID optional)
      - "copilot" → GitHubCopilotJudgeLLM (requires GITHUB_COPILOT_TOKEN;
            GITHUB_COPILOT_MODEL_ID optional, defaults to gpt-4o)
    """
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    if provider == "bedrock":
        model_id = os.environ.get(
            "AWS_BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"
        )
        region = os.environ.get("AWS_REGION", "us-east-1")
        return BedrockJudgeLLM(model_id=model_id, region=region)
    if provider == "openai":
        model_name = os.environ.get("OPENAI_MODEL_ID", "gpt-4o")
        return OpenAIJudgeLLM(model_name=model_name)
    if provider == "copilot":
        model_name = os.environ.get("GITHUB_COPILOT_MODEL_ID", "openai/gpt-4o")
        return GitHubCopilotJudgeLLM(model_name=model_name)
    model_name = os.environ.get("GEMINI_MODEL_ID", "gemini-2.5-flash")
    return GeminiJudgeLLM(model_name=model_name)

# --- Agent Configuration (reads from .env / environment) ---
_AGENTX_BASE_URL = os.environ.get("AGENTX_BASE_URL", "https://genai-cdh-demo.pega.net").rstrip("/")
_AGENT_NAME = os.environ.get("AGENT_NAME", "OPK0KG-SURFACE1-UIPAGES!SURFACEORCHESTRATIONAGENTV7")

AGENT_CARD_URL = (
    f"{_AGENTX_BASE_URL}/prweb/app/surface1dev/api/agent2agent/v1/"
    f"ai-agents/{_AGENT_NAME}/.well-known/agent.json"
)

# OAuth Client ID and Secret
CLIENT_ID = os.environ.get("PEGA_CLIENT_ID", "YOUR_CLIENT_ID")
CLIENT_SECRET = os.environ.get("PEGA_CLIENT_SECRET", "YOUR_CLIENT_SECRET")

# Token URL — derived from the same base URL
TOKEN_URL = f"{_AGENTX_BASE_URL}/prweb/PRRestService/oauth2/v1/token"


@dataclass
class AgentResponse:
    """Rich response object capturing both output text and trace metadata."""
    text: str                                   # The agent's reply text
    context_id: Optional[str] = None            # PXCONV-XXXX — links to AI Tracer
    message_id: Optional[str] = None            # MSG-XXXX
    latency_ms: float = 0.0                     # Round-trip time in milliseconds
    raw_json: Dict[str, Any] = field(default_factory=dict)  # Full JSON-RPC response


@dataclass
class StepAgentExecution:
    """Represents a Step Agent execution detected in the case workflow.

    Step Agents are controlled agents executed by the Pega case (not by user chat).
    They run as workflow steps and pre-fill property values using Knowledge Base queries.
    Evidence comes from:
    - Pre-populated defaultValue entries in pzAssignmentFieldsMetaData
    - 🤖 emoji / "specialist agent" references in assistant content
    - Completed steps in the stages API response
    """
    name: str                                   # Step agent or step name
    source: str                                 # How it was detected: "field_prefill" | "content_pattern" | "stage_step"
    status: str = "detected"                    # "completed" | "active" | "detected"
    details: Dict[str, Any] = field(default_factory=dict)  # Extra info (fields filled, step ID, etc.)


@dataclass
class CanonicalToolEvent:
    """Normalized tool detection event — internal currency for all tool assertions.

    One instance per unique (turn, tool_name) combination.  Callers that only
    need names can use ``[e.tool_name for e in events if not e.is_internal]``.
    """
    turn: int                        # 1-indexed conversation turn; 0 = session-level
    tool_name: str                   # Canonical tool identifier
    source: str                      # "tool_calls" | "plugins" | "regex"
    call_id: Optional[str] = None    # Opaque id from tc["id"]; None for regex events
    arguments: Optional[str] = None  # JSON string of arguments; None for regex events
    confidence: str = "high"         # "high" (structured) | "low" (regex)
    is_internal: bool = False        # True if name is in _INTERNAL_TOOL_NAMES


@dataclass
class ConversationInsight:
    """Parsed conversation data from D_pxAutopilotConversation data view.

    Queried after each agent turn to verify which tools were actually invoked
    (same approach used by the SurfacePOC/NewUI chat interface).
    """
    conversation_id: str
    status: str
    messages: List[Dict[str, Any]]              # All pyMessages (user + assistant)
    tools_detected: List[str]                    # Pega tool names detected from message content
    assistant_messages: List[Dict[str, Any]]     # Only assistant messages
    business_case_key: Optional[str] = None      # pzCaseKey if a case was created
    assignment_key: Optional[str] = None         # pzAssignmentKey if an assignment is open
    step_agents: List[StepAgentExecution] = field(default_factory=list)  # Step agents detected
    prefilled_fields: Dict[str, str] = field(default_factory=dict)       # Fields pre-filled by step agents
    case_stages: List[Dict[str, Any]] = field(default_factory=list)      # Stages/steps from stages API
    raw_json: Dict[str, Any] = field(default_factory=dict)
    tool_events: List[CanonicalToolEvent] = field(default_factory=list)  # Rich tool events with source metadata


# ============================================================================
# Tool detection patterns — ported from
# SurfacePOC/NewUI/pega-chat/server/insight-transformer.ts
# These are the DEFAULT patterns for the Surface project.
# For other projects, provide patterns via project_config / profile.
# ============================================================================

_DEFAULT_TOOL_DETECTION_PATTERNS: List[tuple] = [
    (re.compile(r"(?:creat|launch|start|initiat)\w*\s+(?:new\s+)?(?:campaign\s+)?case[:\s]*\[?([A-Z]-\d+)", re.I),
     "pxCreateCaseWithAssignmentDetails"),
    (re.compile(r"\[([A-Z]-\d+)\]\(https?://", re.I),
     "pxCreateCaseWithAssignmentDetails"),
    (re.compile(r"(?:Surface\s+New\s+Campaign|SurfaceNewCampaign|campaign\s+automation)\s+(?:case|flow)", re.I),
     "SurfaceNewCampaignAutomation"),
    (re.compile(r"creating\s+(?:a\s+)?new.*campaign.*case", re.I),
     "SurfaceNewCampaignAutomation"),
    (re.compile(r"extract\w*\s+(?:information|data|fields|details)\s+from\s+(?:your|the)\s+document", re.I),
     "pxPerformAssignment"),
    (re.compile(r"(?:source|citation)s?\s+of\s+(?:the\s+)?(?:prefilled|pre-filled)", re.I),
     "CaseWorkHistory"),
    (re.compile(r"case\s+history|work\s+history|field.*modif", re.I),
     "CaseWorkHistory"),
    (re.compile(r"(?:assignment|task)s?\s+(?:for|on)\s+(?:this\s+)?case", re.I),
     "GetMyAssignmentsforCase"),
    (re.compile(r"(?:your|all)\s+(?:current\s+)?assignments", re.I),
     "GetMyAssignmentsAll"),
    (re.compile(r"(?:case\s+stages|(?:current|active)\s+stage|stages?\s+(?:API|endpoint|response)|GetCaseStages|lifecycle\s+(?:of|for)\s+(?:the\s+)?case|where.*(?:am|are).*(?:in\s+the\s+)?case)", re.I),
     "GetCaseStages"),
    (re.compile(r"(?:glossary|terminolog|acronym|defin)", re.I),
     "glossary_agent"),
    (re.compile(r"(?:taxonomy|product.*valid|journey.*type)", re.I),
     "taxonomy_agent"),
    # NOTE: Gradial_Agent is NOT an orchestration tool — it is a Step Agent
    # invoked by the Pega case (GradialStepAgent), not by the orchestration agent.
    # Detection moved to _STEP_AGENT_CONTENT_PATTERNS below.
    (re.compile(r"(?:pulse.*comment|post.*(?:to|in)\s+(?:the\s+)?(?:case\s+)?(?:conversation|thread))", re.I),
     "pyPostToPulse"),
    (re.compile(r"\U0001f4c4|extracted\s+from.*document", re.I),
     "pxPerformAssignment"),
    (re.compile(r"\U0001f916|specialist.*(?:sub-)?agent|knowledge\s+base", re.I),
     "glossary_agent"),
]

_DEFAULT_TOOL_LABELS = {
    "pxCreateCaseWithAssignmentDetails": "Created Case",
    "pxPerformAssignment": "Processed Assignment",
    "GetMyAssignmentsforCase": "Checked Assignments",
    "GetMyAssignmentsAll": "Retrieved All Assignments",
    "GetOrchestrationCaseDetail": "Retrieved Case Detail",
    "GetCaseStages": "Checked Case Stages",
    "GetAssignmentDetailsFromAgent": "Loaded Assignment Details",
    "PerformNonBackToBackAssignments": "Completed Assignment",
    "CaseWorkHistory": "Reviewed Case History",
    "SurfaceNewCampaignAutomation": "Started Campaign Automation",
    "CampaignMemoryAdvisor": "Consulted Campaign Memory",
    "glossary_agent": "Looked Up Terminology",
    "taxonomy_agent": "Validated Product Taxonomy",
    "person_agent": "Consulted People Directory",
    "pyPostToPulse": "Posted Pulse Comment",
    "pega_context": "Pega Context",
    "CreateComplaintCasePlugin": "Create Complaint Case",
}

# Backward-compatible aliases — existing code that references the old names
# continues to work until it is updated to use the profile-aware functions.
_TOOL_DETECTION_PATTERNS = _DEFAULT_TOOL_DETECTION_PATTERNS
_TOOL_LABELS = _DEFAULT_TOOL_LABELS


# ============================================================================
# Step Agent detection patterns — identify case-driven agent executions
# ============================================================================

# Content patterns in assistant messages that indicate step agent activity.
# Step agents pre-fill properties using KB queries and the orchestration agent
# references them with specific markers (🤖, "specialist agent", "knowledge base").
_DEFAULT_STEP_AGENT_CONTENT_PATTERNS: List[tuple] = [
    (re.compile(r"\U0001f916.*(?:specialist|sub-?agent|step\s*agent)", re.I),
     "specialist_sub_agent"),
    (re.compile(r"(?:specialist|sub-?agent)\s*(?:filled|provided|set|populated)", re.I),
     "specialist_sub_agent"),
    (re.compile(r"(?:pre-?filled|pre-?populated)\s+(?:by|from|via)\s+(?:a\s+)?(?:specialist|agent|knowledge)", re.I),
     "knowledge_base_prefill"),
    (re.compile(r"(?:knowledge\s+base|KB)\s+(?:quer|lookup|search|retriev)", re.I),
     "knowledge_base_query"),
    (re.compile(r"(?:Accessed|Queried)\s+KB\s+for\s+(\w+)\s+property", re.I),
     "kb_property_access"),
    (re.compile(r"(?:upstream|automated|controlled)\s+(?:step\s+)?agent", re.I),
     "upstream_step_agent"),
    (re.compile(r"(?:properties?\s+extracted|values?\s+extracted)\s+from\s+(?:document|doc)", re.I),
     "document_extraction"),
    (re.compile(r"(?:gradial|content\s+(?:assembly|generation|creation))\s*(?:agent|integration|workspace)", re.I),
     "gradial_content_agent"),
    (re.compile(r"app\.gradial\.com", re.I),
     "gradial_content_agent"),
    (re.compile(r"(?:hero.*banner|marketing.*content.*generat)", re.I),
     "gradial_content_agent"),
    (re.compile(r"(?:Adobe|audience)\s+(?:waterfall|reach|segment)", re.I),
     "adobe_audience_agent"),
]

# Backward-compatible alias
_STEP_AGENT_CONTENT_PATTERNS = _DEFAULT_STEP_AGENT_CONTENT_PATTERNS


# ============================================================================
# Profile-aware pattern builders
# ============================================================================


def build_tool_patterns_from_profile(
    profile: Optional[Dict[str, Any]] = None,
) -> List[tuple]:
    """Build compiled tool detection patterns from a profile.

    If the profile contains ``tool_patterns.patterns``, those are compiled and
    **merged** with the defaults (profile patterns checked first).  If no
    profile is provided, the built-in defaults are returned.
    """
    if not profile:
        return _DEFAULT_TOOL_DETECTION_PATTERNS

    tp = profile.get("tool_patterns", {})
    raw_patterns = tp.get("patterns", [])
    if not raw_patterns:
        return _DEFAULT_TOOL_DETECTION_PATTERNS

    extra = []
    for entry in raw_patterns:
        regex_str = entry.get("regex")
        tool = entry.get("tool")
        if regex_str and tool:
            try:
                extra.append((re.compile(regex_str, re.I), tool))
            except re.error as exc:
                print(f"[Patterns] Bad tool regex in profile: {regex_str!r} — {exc}")

    # Profile patterns first (higher priority), then defaults
    return extra + list(_DEFAULT_TOOL_DETECTION_PATTERNS)


def build_tool_labels_from_profile(
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Build tool label dict from profile, merged with defaults."""
    labels = dict(_DEFAULT_TOOL_LABELS)
    if profile:
        tp = profile.get("tool_patterns", {})
        extra_labels = tp.get("labels", {})
        labels.update(extra_labels)
    return labels


def build_step_agent_patterns_from_profile(
    profile: Optional[Dict[str, Any]] = None,
) -> List[tuple]:
    """Build compiled step-agent content patterns from a profile.

    Profile patterns are prepended to the defaults.
    """
    if not profile:
        return _DEFAULT_STEP_AGENT_CONTENT_PATTERNS

    sap = profile.get("step_agent_patterns", {})
    raw_patterns = sap.get("patterns", [])
    if not raw_patterns:
        return _DEFAULT_STEP_AGENT_CONTENT_PATTERNS

    extra = []
    for entry in raw_patterns:
        regex_str = entry.get("regex")
        agent = entry.get("agent")
        if regex_str and agent:
            try:
                extra.append((re.compile(regex_str, re.I), agent))
            except re.error as exc:
                print(f"[Patterns] Bad step-agent regex in profile: {regex_str!r} — {exc}")

    return extra + list(_DEFAULT_STEP_AGENT_CONTENT_PATTERNS)


def _detect_step_agents_from_content(
    messages: List[Dict[str, Any]],
    patterns: Optional[List[tuple]] = None,
) -> List[StepAgentExecution]:
    """Detect step agent activity from assistant message content patterns.

    Args:
        messages: List of assistant message dicts with ``content`` keys.
        patterns: Optional compiled (regex, agent_name) pairs.  Defaults to
                  ``_DEFAULT_STEP_AGENT_CONTENT_PATTERNS``.
    """
    if patterns is None:
        patterns = _DEFAULT_STEP_AGENT_CONTENT_PATTERNS
    seen: set = set()
    agents: list = []
    for msg in messages:
        content = msg.get("content", "")
        if not content:
            continue
        for pattern, agent_name in patterns:
            match = pattern.search(content)
            if match and agent_name not in seen:
                seen.add(agent_name)
                agents.append(StepAgentExecution(
                    name=agent_name,
                    source="content_pattern",
                    status="detected",
                    details={
                        "matched_text": match.group()[:120],
                        "message_id": msg.get("message_id", ""),
                    },
                ))
    return agents


def _detect_step_agents_from_fields(fields_metadata_str: Optional[str]) -> tuple:
    """Detect step agent pre-fills from pzAssignmentFieldsMetaData.

    Returns (step_agents, prefilled_fields) tuple.
    Fields with a defaultValue indicate an upstream step agent filled them.
    """
    step_agents: List[StepAgentExecution] = []
    prefilled_fields: Dict[str, str] = {}

    if not fields_metadata_str:
        return step_agents, prefilled_fields

    try:
        fields = json.loads(fields_metadata_str)
    except (json.JSONDecodeError, TypeError):
        return step_agents, prefilled_fields

    for f in fields:
        default_val = f.get("defaultValue")
        field_name = f.get("name", "unknown")
        if default_val:
            prefilled_fields[field_name] = str(default_val)

    if prefilled_fields:
        step_agents.append(StepAgentExecution(
            name="field_prefill_agent",
            source="field_prefill",
            status="completed",
            details={
                "fields_count": len(prefilled_fields),
                "field_names": list(prefilled_fields.keys()),
            },
        ))

    return step_agents, prefilled_fields


def _parse_stages_for_step_agents(stages_data: dict) -> tuple:
    """Parse stages API response for completed automated steps (step agents).

    Returns (step_agents, case_stages) tuple.
    Steps that completed automatically (not user assignments) are step agent executions.
    """
    step_agents: List[StepAgentExecution] = []
    case_stages: List[Dict[str, Any]] = []

    for stage in stages_data.get("stages", []):
        stage_info: Dict[str, Any] = {
            "id": stage.get("ID", ""),
            "name": stage.get("name", ""),
            "status": stage.get("visited_status", ""),
            "steps": [],
            "processes": [],
        }

        for seq in stage.get("processSequences", []):
            for proc in seq.get("processes", []):
                proc_info = {
                    "id": proc.get("ID", ""),
                    "name": proc.get("name", ""),
                    "status": proc.get("visited_status", ""),
                    "started_by": proc.get("startedBy", ""),
                }
                stage_info["processes"].append(proc_info)

                for step in proc.get("steps", []):
                    step_status = step.get("visited_status", "")
                    step_info = {
                        "id": step.get("ID", ""),
                        "name": step.get("name", ""),
                        "status": step_status,
                        "process_name": proc.get("name", ""),
                    }
                    stage_info["steps"].append(step_info)

                    # Completed steps that are NOT user assignments are likely step agents.
                    # Step agents complete automatically; user assignments wait for user input.
                    if step_status == "completed":
                        step_name = step.get("name", "")
                        # Heuristic: steps with "agent" in name or that completed automatically
                        if any(kw in step_name.lower() for kw in ["agent", "auto", "genai", "kb", "knowledge", "extract"]):
                            step_agents.append(StepAgentExecution(
                                name=step_name,
                                source="stage_step",
                                status="completed",
                                details={
                                    "step_id": step.get("ID", ""),
                                    "process_name": proc.get("name", ""),
                                    "stage_name": stage.get("name", ""),
                                },
                            ))

        case_stages.append(stage_info)

    return step_agents, case_stages


def _detect_tools_from_messages_v2(
    messages: List[Dict[str, Any]],
    patterns: Optional[List[tuple]] = None,
    turn: int = 0,
) -> List["CanonicalToolEvent"]:
    """Detect tools from messages and return rich CanonicalToolEvent objects.

    Detection strategy (in order of reliability):
      1. **Structural**: ``tool_calls`` arrays on messages → source="tool_calls",
         confidence="high".  Returns immediately if any are found.
      2. **Regex fallback**: compiled patterns on content → source="regex",
         confidence="low".

    Names in ``_INTERNAL_TOOL_NAMES`` are included with ``is_internal=True``
    so callers can decide whether to surface them.

    Args:
        messages: List of message dicts.
        patterns: Optional compiled (regex, tool_name) pairs. Defaults to
                  ``_DEFAULT_TOOL_DETECTION_PATTERNS``.
        turn: 1-indexed turn number to embed in events (0 = session-level).
    """
    # --- Strategy 1: Structural detection from tool_calls arrays ---
    seen: set = set()
    events: List[CanonicalToolEvent] = []
    found_structural = False

    for msg in messages:
        for tc in msg.get("tool_calls", []):
            tool_name = tc.get("function", {}).get("name")
            if tool_name and tool_name not in seen:
                seen.add(tool_name)
                events.append(CanonicalToolEvent(
                    turn=turn,
                    tool_name=tool_name,
                    source="tool_calls",
                    call_id=tc.get("id"),
                    arguments=tc.get("function", {}).get("arguments"),
                    confidence="high",
                    is_internal=tool_name in _INTERNAL_TOOL_NAMES,
                ))
                found_structural = True

    if found_structural:
        return events

    # --- Strategy 2: Regex pattern matching on message content ---
    if patterns is None:
        patterns = _DEFAULT_TOOL_DETECTION_PATTERNS
    seen = set()
    for msg in messages:
        content = msg.get("content", "")
        if not content:
            continue
        for pattern, tool_name in patterns:
            if pattern.search(content) and tool_name not in seen:
                seen.add(tool_name)
                events.append(CanonicalToolEvent(
                    turn=turn,
                    tool_name=tool_name,
                    source="regex",
                    confidence="low",
                    is_internal=tool_name in _INTERNAL_TOOL_NAMES,
                ))
    return events


def _detect_tools_from_messages(
    messages: List[Dict[str, Any]],
    patterns: Optional[List[tuple]] = None,
) -> List[str]:
    """Detect which tools were used by analyzing assistant message content.

    Detection strategy (in order of reliability):
      1. **Structural**: Check for explicit ``tool_calls`` arrays on messages
         (present when messages come from the Pega agent execution output).
      2. **Regex fallback**: Match content text against compiled patterns
         (used when messages come from D_pxAutopilotConversation flat view).

    Returns tool names only.  For rich metadata use
    ``_detect_tools_from_messages_v2()``.
    """
    events = _detect_tools_from_messages_v2(messages, patterns, turn=0)
    return [e.tool_name for e in events if not e.is_internal]


# ============================================================================
# Structured Tool Detection — Extract from Pega Agent output JSON
# ============================================================================
# These functions parse tool calls from the structured agent output format,
# which contains explicit tool_calls arrays and plugins metadata.
# This is more reliable than regex when the structured data is available.

# Tools to exclude from reporting (data infrastructure, not business tools)
# Note: pega_context IS included as it's a meaningful tool invocation indicating context lookup
_INTERNAL_TOOL_NAMES: set = {"data_from_all_sources", "complete_data_source"}


def extract_tools_from_conversation_history(
    agent_output: Dict[str, Any],
    include_internal: bool = False,
) -> List[str]:
    """Extract tool names from conversation_history -> tool_calls.

    Args:
        agent_output: The full Pega agent output JSON containing conversation_history.
        include_internal: If False, filters out internal tools like pega_context.

    Returns:
        List of unique tool names in order of first appearance.
    """
    tools: List[str] = []
    seen: set = set()

    for conv in agent_output.get("conversation_history", []):
        for msg in conv.get("history", []):
            # Extract from tool_calls array on assistant messages
            for tc in msg.get("tool_calls", []):
                tool_name = tc.get("function", {}).get("name")
                if tool_name and tool_name not in seen:
                    if include_internal or tool_name not in _INTERNAL_TOOL_NAMES:
                        seen.add(tool_name)
                        tools.append(tool_name)

            # Also extract from tool role messages (tool responses)
            if msg.get("role") == "tool" and msg.get("name"):
                tool_name = msg["name"]
                if tool_name not in seen:
                    if include_internal or tool_name not in _INTERNAL_TOOL_NAMES:
                        seen.add(tool_name)
                        tools.append(tool_name)

    return tools


def extract_tools_from_plugins(
    agent_output: Dict[str, Any],
) -> List[str]:
    """Extract tool names from the plugins array (execution metadata).

    The plugins array contains detailed execution data for each tool invocation,
    including timing, prerequisites, and sub-agent calls.

    Args:
        agent_output: The full Pega agent output JSON containing plugins.

    Returns:
        List of unique plugin/tool names in order of first execution.
    """
    tools: List[str] = []
    seen: set = set()

    for plugin in agent_output.get("plugins", []):
        name = plugin.get("name")
        if name and name not in seen:
            seen.add(name)
            tools.append(name)

        # Also check prerequisites for sub-tools
        for prereq_name, prereq_data in plugin.get("prerequisites", {}).items():
            if isinstance(prereq_data, dict):
                prereq_tool = prereq_data.get("name")
                if prereq_tool and prereq_tool not in seen and prereq_tool not in _INTERNAL_TOOL_NAMES:
                    seen.add(prereq_tool)
                    tools.append(prereq_tool)

    return tools


def extract_available_tools_from_metrics(
    agent_output: Dict[str, Any],
) -> List[str]:
    """Extract the full list of available tools from metrics -> vector_store entries.

    The metrics array contains vector_store entries for each tool that was
    considered during processing. This gives us the complete tool inventory.

    Args:
        agent_output: The full Pega agent output JSON containing metrics.

    Returns:
        List of unique tool names that are available to the agent.
    """
    tools: List[str] = []
    seen: set = set()

    for metric in agent_output.get("metrics", []):
        for artifact in metric.get("artifact_metrics", []):
            if artifact.get("_type") == "vector_store":
                name = artifact.get("name")
                if name and name not in seen:
                    seen.add(name)
                    tools.append(name)

    return tools


def build_tool_registry(
    agent_output: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Build a comprehensive registry of all tools discovered from agent output.

    Extracts tools from multiple sources and combines them into a registry
    that can be saved in the golden session for later comparison.

    Args:
        agent_output: The full Pega agent output JSON.

    Returns:
        Dict mapping tool name to metadata about the tool.
    """
    registry: Dict[str, Dict[str, Any]] = {}

    # Source 1: conversation_history tool_calls (tools actually invoked)
    for conv in agent_output.get("conversation_history", []):
        for msg in conv.get("history", []):
            for tc in msg.get("tool_calls", []):
                name = tc.get("function", {}).get("name")
                if name and name not in registry:
                    registry[name] = {
                        "name": name,
                        "discovered_from": "tool_calls",
                        "sample_args": tc.get("function", {}).get("arguments"),
                        "invoked": True,
                    }
                elif name:
                    registry[name]["invoked"] = True

    # Source 2: plugins array (execution details)
    for plugin in agent_output.get("plugins", []):
        name = plugin.get("name")
        if name:
            if name not in registry:
                registry[name] = {
                    "name": name,
                    "discovered_from": "plugins",
                    "invoked": True,
                }
            registry[name]["execution_time_ms"] = (
                (plugin.get("end_time", 0) - plugin.get("start_time", 0)) * 1000
                if plugin.get("start_time") and plugin.get("end_time")
                else None
            )

    # Source 3: metrics vector_store (available tools)
    for metric in agent_output.get("metrics", []):
        for artifact in metric.get("artifact_metrics", []):
            if artifact.get("_type") == "vector_store":
                name = artifact.get("name")
                if name and name not in registry:
                    registry[name] = {
                        "name": name,
                        "discovered_from": "vector_store",
                        "invoked": False,
                    }

    return registry


def detect_tools_hybrid(
    agent_output: Optional[Dict[str, Any]] = None,
    assistant_messages: Optional[List[Dict[str, Any]]] = None,
    fallback_patterns: Optional[List[tuple]] = None,
    include_internal: bool = False,
) -> List[str]:
    """Detect tools using structured data first, regex patterns as fallback.

    This is the recommended entry point for tool detection. It tries:
    1. Structured extraction from conversation_history (most reliable)
    2. Structured extraction from plugins array
    3. Regex pattern matching on assistant message content (legacy fallback)

    Args:
        agent_output: The full Pega agent output JSON (preferred source).
        assistant_messages: List of assistant message dicts for regex fallback.
        fallback_patterns: Regex patterns for fallback detection.
        include_internal: If False, filters out internal tools like pega_context.

    Returns:
        List of unique tool names.
    """
    # Try structured extraction first
    if agent_output:
        # Primary: conversation_history tool_calls
        tools = extract_tools_from_conversation_history(agent_output, include_internal)
        if tools:
            return tools

        # Secondary: plugins array
        tools = extract_tools_from_plugins(agent_output)
        if tools:
            return tools

    # Fallback: regex pattern matching
    if assistant_messages:
        return _detect_tools_from_messages(assistant_messages, fallback_patterns)

    return []


def extract_per_turn_tools(
    agent_output: Dict[str, Any],
    include_internal: bool = False,
) -> List[Dict[str, Any]]:
    """Extract tools invoked at each turn of the conversation.

    Returns a list aligned with user turns, each containing the tools
    invoked during that turn's processing.

    Args:
        agent_output: The full Pega agent output JSON.
        include_internal: If False, filters out internal tools like pega_context.

    Returns:
        List of dicts, one per user turn, each with:
        - turn: Turn number (1-indexed)
        - user_input: The user's message
        - tools_invoked: List of tool names invoked for this turn
        - tool_details: List of dicts with tool call details
    """
    turns: List[Dict[str, Any]] = []
    turn_num = 0

    for conv in agent_output.get("conversation_history", []):
        history = conv.get("history", [])
        i = 0

        while i < len(history):
            msg = history[i]

            # Find user messages
            if msg.get("role") == "user":
                turn_num += 1
                user_input = msg.get("content", "")
                tools_invoked: List[str] = []
                tool_details: List[Dict[str, Any]] = []

                # Collect tools from subsequent assistant messages until next user
                j = i + 1
                while j < len(history) and history[j].get("role") != "user":
                    asst_msg = history[j]

                    # Get tool_calls from assistant message
                    for tc in asst_msg.get("tool_calls", []):
                        tool_name = tc.get("function", {}).get("name")
                        if tool_name:
                            if include_internal or tool_name not in _INTERNAL_TOOL_NAMES:
                                if tool_name not in tools_invoked:
                                    tools_invoked.append(tool_name)
                                tool_details.append({
                                    "name": tool_name,
                                    "call_id": tc.get("id"),
                                    "arguments": tc.get("function", {}).get("arguments"),
                                })

                    j += 1

                turns.append({
                    "turn": turn_num,
                    "user_input": user_input,
                    "tools_invoked": tools_invoked,
                    "tool_details": tool_details,
                })

                i = j
            else:
                i += 1

    return turns


# ============================================================================
# PegaInsight — DX API client for conversation insight
# ============================================================================

class PegaInsight:
    """
    Queries Pega DX API v2 to retrieve conversation insight from the
    D_pxAutopilotConversation data view.

    After each agent turn, call query_conversation(context_id) to get:
    - Full message history (user + assistant + intermediate tool messages)
    - Detected tool invocations
    - Business case / assignment metadata

    This is the same approach used by the SurfacePOC/NewUI chat interface
    (see server/index.ts GET /api/insight/conversation/:conversationId).
    """

    def __init__(self, base_url: str, client_id: str, client_secret: str, token_url: str):
        self.base_url = base_url.rstrip("/")
        self.dx_api_base = f"{self.base_url}/prweb/api/application/v2"
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self._access_token: Optional[str] = None

    def _authenticate(self):
        """Fetch OAuth2 token."""
        resp = requests.post(self.token_url, data={
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        })
        resp.raise_for_status()
        self._access_token = resp.json()["access_token"]
        print("[Insight Auth] Token acquired")

    def _headers(self):
        if not self._access_token:
            self._authenticate()
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

    def query_conversation(
        self,
        conversation_id: str,
        retry_count: int = 3,
        retry_delay: float = 2.0,
    ) -> ConversationInsight:
        """
        Query D_pxAutopilotConversation data view for full conversation insight.

        Args:
            conversation_id: The contextId from the A2A response (e.g. PXCONV-12345)
            retry_count: Number of retries if data isn't available yet
            retry_delay: Seconds between retries
        """
        params_json = json.dumps({"InteractionID": conversation_id})
        url = f"{self.dx_api_base}/data_views/D_pxAutopilotConversation?dataViewParameters={quote(params_json)}"

        print(f"\n[Insight] Querying conversation: {conversation_id}")
        print(f"[Insight] URL: {url}")

        last_error = None
        for attempt in range(retry_count):
            try:
                resp = requests.get(url, headers=self._headers(), timeout=30)

                # If 401, re-authenticate and retry
                if resp.status_code == 401 and attempt < retry_count - 1:
                    print(f"[Insight] 401 — re-authenticating (attempt {attempt + 1}/{retry_count})")
                    self._authenticate()
                    continue

                resp.raise_for_status()
                raw = resp.json()

                messages_raw = raw.get("pyMessages", [])
                if not messages_raw and attempt < retry_count - 1:
                    print(f"[Insight] No messages yet (attempt {attempt + 1}/{retry_count}), retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue

                return self._parse_response(raw, conversation_id)

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < retry_count - 1:
                    print(f"[Insight] Request failed (attempt {attempt + 1}/{retry_count}): {e}")
                    time.sleep(retry_delay)
                    self._authenticate()
                    continue

        raise RuntimeError(
            f"Failed to query conversation insight after {retry_count} attempts: {last_error}"
        )

    def _parse_response(self, raw: dict, conversation_id: str) -> ConversationInsight:
        """Parse raw D_pxAutopilotConversation response into ConversationInsight."""
        messages_raw = raw.get("pyMessages", [])
        cpd = raw.get("pzCaseProcessingData", {})

        parsed_messages: List[Dict[str, Any]] = []
        assistant_messages: List[Dict[str, Any]] = []

        for m in messages_raw:
            role_raw = m.get("pyRole", "")
            msg: Dict[str, Any] = {
                "role": "user" if role_raw == "USER" else "assistant",
                "content": _decode_html_entities(m.get("pyContent", "")),
                "message_id": m.get("pyMessageID", ""),
                "timestamp": m.get("pxCreateDateTime", ""),
                "raw_role": role_raw,
            }
            parsed_messages.append(msg)
            if role_raw == "ASSISTANT":
                assistant_messages.append(msg)

        # Detect tools from ALL assistant messages (not just the final response)
        tool_events = _detect_tools_from_messages_v2(assistant_messages, turn=0)
        tools_detected = [e.tool_name for e in tool_events if not e.is_internal]

        # --- Step Agent Detection ---
        all_step_agents: List[StepAgentExecution] = []
        prefilled_fields: Dict[str, str] = {}

        # Source 1: Content patterns in assistant messages
        content_step_agents = _detect_step_agents_from_content(assistant_messages)
        all_step_agents.extend(content_step_agents)

        # Source 2: Pre-filled field values from pzAssignmentFieldsMetaData
        field_step_agents, prefilled_fields = _detect_step_agents_from_fields(
            cpd.get("pzAssignmentFieldsMetaData")
        )
        all_step_agents.extend(field_step_agents)

        # Source 3: Previous assignment metadata (step agents may have filled fields in prior step)
        prev_field_agents, prev_prefilled = _detect_step_agents_from_fields(
            cpd.get("pzPrevAssignmentFieldsMetaData")
        )
        for agent in prev_field_agents:
            agent.name = "prev_step_" + agent.name
            all_step_agents.append(agent)
        if prev_prefilled:
            for k, v in prev_prefilled.items():
                prefilled_fields.setdefault(f"prev:{k}", v)

        # Source 4: Query stages API if a business case exists
        case_stages: List[Dict[str, Any]] = []
        if cpd.get("pzCaseKey"):
            try:
                stages_data = self._query_case_stages(cpd["pzCaseKey"])
                if stages_data:
                    stage_step_agents, case_stages = _parse_stages_for_step_agents(stages_data)
                    all_step_agents.extend(stage_step_agents)
            except Exception as e:
                print(f"[Insight] Stages query failed (non-fatal): {e}")

        # --- Logging ---
        print(f"[Insight] Messages: {len(parsed_messages)} total, {len(assistant_messages)} assistant")
        if tool_events:
            for evt in tool_events:
                if not evt.is_internal:
                    print(f"[Insight] Tool detected: {evt.tool_name} "
                          f"(source={evt.source}, confidence={evt.confidence})")
        else:
            print("[Insight] Tools detected: none")
        print(f"[Insight] Step agents: {len(all_step_agents)} detected")
        for sa in all_step_agents:
            print(f"[Insight]   Step Agent: {sa.name} (source={sa.source}, status={sa.status})")
        if prefilled_fields:
            print(f"[Insight] Pre-filled fields: {list(prefilled_fields.keys())}")
        for i, msg in enumerate(assistant_messages):
            preview = msg["content"][:120].replace("\n", " ")
            print(f"[Insight]   Assistant[{i}]: {preview}...")
        if cpd.get("pzCaseKey"):
            print(f"[Insight] Business case: {cpd['pzCaseKey']}")
        if cpd.get("pzAssignmentKey"):
            print(f"[Insight] Assignment: {cpd['pzAssignmentKey']}")
        if case_stages:
            for stage in case_stages:
                completed = [s for s in stage.get("steps", []) if s.get("status") == "completed"]
                active = [s for s in stage.get("steps", []) if s.get("status") == "active"]
                print(f"[Insight]   Stage '{stage['name']}': {len(completed)} completed, {len(active)} active steps")

        return ConversationInsight(
            conversation_id=raw.get("pyID", conversation_id),
            status=raw.get("pyStatusWork", "Unknown"),
            messages=parsed_messages,
            tools_detected=tools_detected,
            assistant_messages=assistant_messages,
            business_case_key=cpd.get("pzCaseKey"),
            assignment_key=cpd.get("pzAssignmentKey"),
            step_agents=all_step_agents,
            prefilled_fields=prefilled_fields,
            case_stages=case_stages,
            raw_json=raw,
            tool_events=tool_events,
        )

    def _query_case_stages(self, case_key: str) -> Optional[dict]:
        """Query the stages API for a business case to get step-level detail.

        Uses: GET /api/application/v2/cases/{caseKey}/stages
        Returns the raw stages response or None if unavailable.
        """
        encoded_key = quote(case_key, safe="")
        url = f"{self.dx_api_base}/cases/{encoded_key}/stages"
        print(f"[Insight] Fetching stages for case: {case_key}")

        try:
            resp = requests.get(url, headers=self._headers(), timeout=15)
            if resp.status_code == 401:
                self._authenticate()
                resp = requests.get(url, headers=self._headers(), timeout=15)
            if resp.ok:
                data = resp.json()
                stage_count = len(data.get("stages", []))
                print(f"[Insight] Stages API: {stage_count} stages found")
                return data
            else:
                print(f"[Insight] Stages API returned {resp.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"[Insight] Stages API request failed: {e}")
            return None


def _decode_html_entities(text: str) -> str:
    """Decode common HTML entities that Pega stores in pyContent."""
    if not text:
        return text
    return (
        text
        .replace("&#x27;", "'")
        .replace("&#39;", "'")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
    )


class SurfaceAgent:
    """
    A client for interacting with a Pega Surface Agent via A2A.
    """
    def __init__(self, agent_card_url, client_id, client_secret, token_url):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.access_token = None
        self.agent_card_url = agent_card_url
        
        # 1. Authenticate first (agent card requires auth)
        self._authenticate()
        
        try:
            # 2. Discover agent endpoint from the agent card URL (authenticated)
            headers = self._auth_headers()
            resp = requests.get(agent_card_url, headers=headers)
            resp.raise_for_status()
            agent_config = resp.json()
            
            # Print the agent card for debugging
            print(f"\n--- Agent Card Response ---\n{json.dumps(agent_config, indent=2)}\n---")
            
            # A2A spec uses "url" at the top level for the agent's endpoint
            self.api_endpoint = agent_config.get("url")
            if not self.api_endpoint:
                raise ValueError(f"Could not find 'url' in agent configuration. Keys found: {list(agent_config.keys())}")
                
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to fetch agent configuration: {e}") from e
        except (ValueError, KeyError) as e:
            raise RuntimeError(f"Failed to parse agent configuration: {e}") from e

    def _authenticate(self):
        """Fetches an OAuth2 access token using client credentials."""
        resp = requests.post(self.token_url, data={
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        })
        resp.raise_for_status()
        token_data = resp.json()
        self.access_token = token_data["access_token"]
        print(f"[Auth] Token acquired (expires in {token_data.get('expires_in', '?')}s)")

    def _auth_headers(self):
        """Returns headers with the Bearer token."""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    def run(self, task, timeout=120, context_id=None):
        """Sends a task to the Surface agent via A2A JSON-RPC and returns an AgentResponse.

        Args:
            task: The user message text to send.
            timeout: HTTP request timeout in seconds.
            context_id: Optional contextId for multi-turn conversations.
                        If provided, continues an existing conversation instead
                        of starting a new one.  Pass the contextId from a
                        previous AgentResponse to chain turns together.
        """
        if not self.api_endpoint:
            raise RuntimeError("API endpoint is not configured.")
        
        # A2A standard JSON-RPC payload
        message: Dict[str, Any] = {
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": task
                }
            ],
        }
        # Attach contextId for multi-turn continuation
        if context_id:
            message["contextId"] = context_id

        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/send",
            "params": {
                "message": message
            }
        }
        
        print(f"\n[Agent] Sending task: {task}")
        t0 = time.perf_counter()
        response = requests.post(
            self.api_endpoint,
            json=payload,
            headers=self._auth_headers(),
            timeout=timeout
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        response.raise_for_status()
        
        # Print raw response for debugging
        resp_json = response.json()
        print(f"\n--- Agent Response ({latency_ms:.0f}ms) ---\n{json.dumps(resp_json, indent=2)}\n---")
        
        # Extract metadata from A2A JSON-RPC response
        result = resp_json.get("result", resp_json)
        context_id = result.get("contextId")
        message_id = result.get("messageId")
        
        # Extract text from parts
        text = ""
        parts = result.get("parts", [])
        if parts:
            texts = [p.get("text", "") for p in parts if p.get("text")]
            text = " ".join(texts)
        
        # Try artifacts fallback
        if not text:
            artifacts = result.get("artifacts", [])
            if artifacts:
                artifact_parts = artifacts[0].get("parts", [])
                texts = [p.get("text", "") for p in artifact_parts if p.get("text")]
                text = " ".join(texts)
        
        # Fallback: try common response keys
        if not text:
            for key in ["response", "output", "text", "message"]:
                if key in result and isinstance(result[key], str):
                    text = result[key]
                    break
        
        # Last resort: return serialized response
        if not text:
            text = json.dumps(resp_json)
        
        return AgentResponse(
            text=text,
            context_id=context_id,
            message_id=message_id,
            latency_ms=latency_ms,
            raw_json=resp_json,
        )

@pytest.fixture(scope="module")
def surface_agent():
    """
    Pytest fixture to set up the SurfaceAgent.
    Scope is 'module' to avoid re-authenticating for every test.
    """
    if CLIENT_ID == "YOUR_CLIENT_ID" or CLIENT_SECRET == "YOUR_CLIENT_SECRET":
        pytest.skip("Pega client credentials are not configured. Skipping integration tests.")
    
    try:
        return SurfaceAgent(AGENT_CARD_URL, CLIENT_ID, CLIENT_SECRET, TOKEN_URL)
    except RuntimeError as e:
        pytest.fail(f"Failed to initialize SurfaceAgent: {e}")


@pytest.fixture(scope="module")
def pega_insight():
    """
    Pytest fixture for querying Pega conversation insight via DX API.
    Uses the same OAuth credentials as SurfaceAgent.
    """
    base_url = os.environ.get("AGENTX_BASE_URL", "https://genai-cdh-demo.pega.net")
    return PegaInsight(
        base_url=base_url,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        token_url=TOKEN_URL,
    )

# ============================================================================
# Ground-truth test cases based on real Surface agent behavior
# ============================================================================

# Shared judge instance (Vertex AI Gemini) — reused across tests
_judge = None

def _get_judge():
    global _judge
    if _judge is None:
        _judge = GeminiJudgeLLM()
    return _judge


# ---------- Test 1: Campaign creation flow ----------

def test_create_new_campaign(surface_agent, pega_insight):
    """
    When a marketer says 'Create a new campaign', the agent should:
    - Acknowledge and start a new campaign case
    - Ask whether the user wants to provide a document or proceed without one
    AI Tracer expected tools: Pega context → Data from all sources → Pxcreatecasewithassignme
    """
    task = "Create a new campaign"
    response = surface_agent.run(task)

    # ---- Trace metadata assertions ----
    assert response.context_id is not None, "No contextId returned — agent session not established"
    assert response.message_id is not None, "No messageId returned"
    assert response.latency_ms < 30_000, (
        f"Response too slow: {response.latency_ms:.0f}ms (budget: 30 000ms)"
    )
    print(f"  ↳ contextId={response.context_id}  messageId={response.message_id}  latency={response.latency_ms:.0f}ms")

    # ---- Conversation Insight: verify tool invocations ----
    time.sleep(2)  # Brief delay for Pega to finalize conversation state
    insight = pega_insight.query_conversation(response.context_id)

    # Campaign creation should invoke SurfaceNewCampaignAutomation or pxCreateCaseWithAssignmentDetails
    # AND create a business case
    campaign_tools = {"pxCreateCaseWithAssignmentDetails", "SurfaceNewCampaignAutomation"}
    found_campaign_tool = campaign_tools & set(insight.tools_detected)

    # Primary: business case must exist (definitive proof a case was created)
    assert insight.business_case_key is not None, (
        f"No business case created for '{task}'. "
        f"Tools detected: {insight.tools_detected}. "
        f"Expected a campaign case to be created."
    )
    print(f"  ↳ Business case created: {insight.business_case_key}")

    # Secondary: at least one campaign-related tool should be detected
    if found_campaign_tool:
        print(f"  ↳ Campaign tools detected: {found_campaign_tool}")
    else:
        print(f"  ⚠ No campaign tool pattern matched, but case was created. Tools: {insight.tools_detected}")

    print(f"  ↳ Conversation tools: {insight.tools_detected}")
    if insight.assignment_key:
        print(f"  ↳ Assignment: {insight.assignment_key}")

    # ---- Step Agent analysis ----
    if insight.step_agents:
        print(f"  ↳ Step Agents detected: {len(insight.step_agents)}")
        for sa in insight.step_agents:
            print(f"    • {sa.name} (source={sa.source}, status={sa.status})")
            if sa.details:
                print(f"      details: {sa.details}")
    else:
        print(f"  ↳ Step Agents: none detected (case may still be initializing)")

    if insight.prefilled_fields:
        print(f"  ↳ Pre-filled fields ({len(insight.prefilled_fields)}): {list(insight.prefilled_fields.keys())}")

    if insight.case_stages:
        print(f"  ↳ Case stages: {len(insight.case_stages)}")
        for stage in insight.case_stages:
            steps = stage.get("steps", [])
            completed = [s for s in steps if s.get("status") == "completed"]
            print(f"    • Stage '{stage['name']}': {len(steps)} steps ({len(completed)} completed)")

    # ---- DeepEval quality assertions ----
    test_case = LLMTestCase(
        input=task,
        actual_output=response.text,
        expected_output=(
            "The agent starts a new campaign case and asks whether the user "
            "wants to provide a document or proceed without one. "
            "Options presented: 'Will provide document' and 'Proceed without one'."
        ),
        context=[
            "The Surface agent is a marketing automation assistant.",
            "When asked to create a campaign, it starts a new campaign case.",
            "It then asks the user whether they want to upload a document or proceed without one.",
            "Valid options are: 'Will provide document (you can upload now)' and 'Proceed without one'.",
        ],
    )

    judge = _get_judge()
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=judge)
    hallucination_metric = HallucinationMetric(threshold=0.5, model=judge)

    assert_test(test_case, [relevancy_metric, hallucination_metric])


# ---------- Test 2: Glossary / knowledge lookup ----------

def test_glossary_lookup_p2p(surface_agent, pega_insight):
    """
    When a marketer asks 'what does P2P mean?', the agent should:
    - Look it up in the marketing glossary
    - Return: P2P = Person-to-Person Payment
    - Mention digital platforms, Zelle, Wells Fargo context
    AI Tracer expected tools: Pega context → Glossary agent
    """
    task = "what does P2P mean?"
    response = surface_agent.run(task)

    # ---- Trace metadata assertions ----
    assert response.context_id is not None, "No contextId returned — agent session not established"
    assert response.message_id is not None, "No messageId returned"
    assert response.latency_ms < 30_000, (
        f"Response too slow: {response.latency_ms:.0f}ms (budget: 30 000ms)"
    )
    print(f"  ↳ contextId={response.context_id}  messageId={response.message_id}  latency={response.latency_ms:.0f}ms")

    # ---- Conversation Insight: verify glossary_agent was invoked ----
    time.sleep(2)
    insight = pega_insight.query_conversation(response.context_id)

    assert "glossary_agent" in insight.tools_detected, (
        f"glossary_agent was NOT invoked for '{task}'. "
        f"Tools detected: {insight.tools_detected}. "
        f"The agent may have used general LLM knowledge instead of the glossary tool. "
        f"Assistant messages: {[m['content'][:100] for m in insight.assistant_messages]}"
    )
    print(f"  ↳ Conversation tools: {insight.tools_detected}")

    # ---- Step Agent analysis ----
    if insight.step_agents:
        print(f"  ↳ Step Agents: {[(sa.name, sa.source) for sa in insight.step_agents]}")
    else:
        print(f"  ↳ Step Agents: none (expected for glossary-only queries)")

    # ---- DeepEval quality assertions ----
    test_case = LLMTestCase(
        input=task,
        actual_output=response.text,
        expected_output=(
            "P2P stands for Person-to-Person Payment. "
            "It refers to electronic money transfers made from one person to another "
            "through digital platforms. In banking, P2P payments allow customers to send "
            "money directly to friends, family, or others without physical cash or checks. "
            "Zelle is an example of a P2P payment service offered by Wells Fargo."
        ),
        context=[
            "P2P stands for Person-to-Person Payment.",
            "P2P refers to electronic money transfers between individuals via digital platforms.",
            "In banking, P2P lets customers send money to friends, family, or others without cash or checks.",
            "Zelle is a P2P payment service offered by Wells Fargo.",
            "The agent should look this up in the marketing glossary.",
        ],
    )

    judge = _get_judge()
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=judge)
    hallucination_metric = HallucinationMetric(threshold=0.5, model=judge)

    assert_test(test_case, [relevancy_metric, hallucination_metric])


# ---------- Test 3: Off-topic / guardrail test ----------

def test_off_topic_rejection(surface_agent, pega_insight):
    """
    The agent should stay on topic (marketing automation).
    An unrelated question should NOT produce a weather answer.
    Instead it should redirect the user to marketing tasks.
    
    We use HallucinationMetric here: the context says "this is a marketing agent",
    so any weather-related content would be a hallucination (not grounded in context).
    We also do a simple assertion that the response mentions marketing.
    """
    task = "What is the weather in New York today?"
    response = surface_agent.run(task)

    # ---- Trace metadata assertions ----
    assert response.context_id is not None, "No contextId returned — agent session not established"
    assert response.latency_ms < 30_000, (
        f"Response too slow: {response.latency_ms:.0f}ms (budget: 30 000ms)"
    )
    print(f"  ↳ contextId={response.context_id}  latency={response.latency_ms:.0f}ms")

    # ---- Conversation Insight: verify no knowledge tools were invoked ----
    time.sleep(2)
    insight = pega_insight.query_conversation(response.context_id)
    # Off-topic should NOT trigger any knowledge tools
    knowledge_tools = {"glossary_agent", "taxonomy_agent", "person_agent", "Gradial_Agent"}
    unexpected = set(insight.tools_detected) & knowledge_tools
    if unexpected:
        print(f"  ⚠ Unexpected knowledge tools invoked for off-topic: {unexpected}")
    print(f"  ↳ Conversation tools: {insight.tools_detected or 'none (expected)'}")

    # ---- Step Agent analysis ----
    if insight.step_agents:
        print(f"  ⚠ Step Agents detected for off-topic query: {[(sa.name, sa.source) for sa in insight.step_agents]}")
    else:
        print(f"  ↳ Step Agents: none (expected for off-topic)")

    # Simple guardrail assertion: agent should mention marketing and NOT give weather info
    output_lower = response.text.lower()
    assert "marketing" in output_lower or "campaign" in output_lower, (
        f"Agent did not redirect to marketing topics. Response: {response.text[:200]}"
    )
    assert "degrees" not in output_lower and "forecast" not in output_lower, (
        f"Agent answered the weather question instead of redirecting. Response: {response.text[:200]}"
    )

    # Also evaluate hallucination: agent shouldn't make up weather facts
    test_case = LLMTestCase(
        input=task,
        actual_output=response.text,
        context=[
            "The Surface agent is strictly a marketing automation assistant for U+ Bank.",
            "It can create campaigns, process marketing briefs, and look up marketing glossary terms.",
            "It does not have access to weather data or any non-marketing information.",
        ],
    )

    judge = _get_judge()
    hallucination_metric = HallucinationMetric(threshold=0.5, model=judge)

    assert_test(test_case, [hallucination_metric])


# ============================================================================
# Custom metric: Glossary Source Verification
# ============================================================================

# Phrases that indicate the agent did NOT use the Glossary tool and instead
# fell back to general LLM knowledge or hedged with multiple guesses.
_GENERAL_KNOWLEDGE_INDICATORS = [
    "could refer to",
    "could mean",
    "might refer to",
    "might mean",
    "depending on the context",
    "depending on context",
    "various concepts",
    "various meanings",
    "multiple meanings",
    "i don't have specific information",
    "i don't have access to a specific definition",
    "i'm not sure",
    "i am not sure",
    "commonly stands for",
    "can stand for",
    "here are some possible",
    "several possible",
]


class GlossarySourceMetric(BaseMetric):
    """
    Evaluates whether the agent's response was sourced from the Glossary tool
    rather than general LLM knowledge.

    Scoring rubric (0-1):
    - 1.0: Single authoritative definition, domain-specific, no hedging
    - 0.5: Definition present but some ambiguity or extra sources mixed in
    - 0.0: Multiple possible meanings listed, hedging, or "I don't know"

    Uses Gemini judge for nuanced evaluation + hard keyword checks.
    """

    def __init__(self, threshold: float = 0.8, model: DeepEvalBaseLLM = None):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False

    @property
    def __name__(self):
        return "GlossarySourceMetric"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        output = test_case.actual_output
        output_lower = output.lower()

        # ---- Hard-fail: general-knowledge hedging phrases ----
        found_indicators = [
            phrase for phrase in _GENERAL_KNOWLEDGE_INDICATORS
            if phrase in output_lower
        ]
        if found_indicators:
            self.score = 0.0
            self.reason = (
                f"Response contains general-knowledge hedging language, "
                f"indicating the Glossary agent was NOT used. "
                f"Detected phrases: {found_indicators}"
            )
            self.success = False
            return self.score

        # ---- LLM judge: is this a glossary-sourced definition? ----
        judge_prompt = f"""You are evaluating whether an AI agent's response to a definition question
was sourced from a structured internal glossary tool versus general LLM knowledge.

USER QUESTION: {test_case.input}

AGENT RESPONSE:
{output}

EXPECTED GLOSSARY BEHAVIOR:
- Provides ONE authoritative definition (not a list of possibilities)
- Uses domain-specific context (banking, marketing, financial services)
- Confident, definitive tone — no hedging or "could mean" language
- May reference specific products, services, or industry context

GENERAL KNOWLEDGE FALLBACK BEHAVIOR (bad):
- Lists multiple possible meanings ("could refer to X, Y, or Z")
- Hedges with "depending on context" or "I'm not sure"
- Gives a generic Wikipedia-style definition without domain specificity
- Says "I don't have specific information"

Score the response on a scale of 0 to 10:
- 10: Clearly from a structured glossary — single authoritative domain-specific definition
- 5: Definition present but mixed with general knowledge or slight ambiguity
- 0: General knowledge fallback, multiple meanings, or no definition at all

Respond with ONLY a JSON object: {{"score": <0-10>, "reason": "<brief explanation>"}}"""

        try:
            judge = _get_judge()  # Use module-level singleton
            raw = judge.generate(judge_prompt)
            # Parse JSON from judge response
            match = re.search(r'\{[^}]+\}', raw)
            if match:
                parsed = json.loads(match.group())
                self.score = parsed.get("score", 0) / 10.0  # normalize to 0-1
                self.reason = parsed.get("reason", "No reason provided by judge")
            else:
                self.score = 0.0
                self.reason = f"Could not parse judge response: {raw[:200]}"
        except Exception as e:
            self.score = 0.0
            self.reason = f"Judge evaluation failed: {e}"

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.measure, test_case)

    def is_successful(self) -> bool:
        return self.success


# ---------- Test 4: Definition questions MUST use Glossary agent ----------

# Parametrize with different phrasings — all should route to the Glossary tool
_DEFINITION_CASES = [
    {
        "id": "p2p_define",
        "task": "Define P2P",
        "term": "P2P",
        # Keywords that MUST ALL appear when the Glossary agent is used
        "required_keywords": ["payment"],
        # At least ONE of these variants must appear (glossary canonical form)
        "definition_variants": ["person to person", "peer to peer"],
        # Any ONE of these is sufficient (domain-specific context from glossary)
        "domain_keywords": ["zelle", "wells fargo", "digital payment", "bank account"],
        "expected_snippet": "Person-to-Person Payment",
    },
    {
        "id": "p2p_what_does",
        "task": "What does P2P mean?",
        "term": "P2P",
        "required_keywords": ["payment"],
        "definition_variants": ["person to person", "peer to peer"],
        "domain_keywords": ["zelle", "wells fargo", "digital payment", "bank account"],
        "expected_snippet": "Person-to-Person Payment",
    },
    {
        "id": "p2p_definition_of",
        "task": "What is the definition of P2P?",
        "term": "P2P",
        "required_keywords": ["payment"],
        "definition_variants": ["person to person", "peer to peer"],
        "domain_keywords": ["zelle", "wells fargo", "digital payment", "bank account"],
        "expected_snippet": "Person-to-Person Payment",
    },
]


@pytest.mark.parametrize(
    "case",
    _DEFINITION_CASES,
    ids=[c["id"] for c in _DEFINITION_CASES],
)
def test_definition_uses_glossary_only(surface_agent, pega_insight, case):
    """
    Definition questions (Define X / What does X mean? / What is the definition of X?)
    MUST be answered by the Glossary agent — not general LLM knowledge.

    Verification layers:
    1. Conversation Insight: Query D_pxAutopilotConversation to confirm glossary_agent
       tool was actually invoked (definitive proof, not heuristic)
    2. Output quality: Hard keyword checks + GlossarySourceMetric LLM judge
    3. Hallucination: Ensure response is grounded in glossary context
    """
    task = case["task"]
    term = case["term"]
    expected_snippet = case["expected_snippet"]
    required_keywords = case.get("required_keywords", [])
    domain_keywords = case.get("domain_keywords", [])

    response = surface_agent.run(task)

    # ---- Trace metadata ----
    assert response.context_id is not None, "No contextId — agent session not established"
    assert response.latency_ms < 30_000, (
        f"Response too slow: {response.latency_ms:.0f}ms (budget: 30 000ms)"
    )
    print(f"  ↳ contextId={response.context_id}  latency={response.latency_ms:.0f}ms")
    print(f"  ↳ Response preview: {response.text[:200]}")

    # ==== LAYER 1: Conversation Insight — definitive tool invocation proof ====
    time.sleep(2)
    insight = pega_insight.query_conversation(response.context_id)

    print(f"  ↳ Conversation tools: {insight.tools_detected}")
    print(f"  ↳ Messages: {len(insight.messages)} total, {len(insight.assistant_messages)} assistant")

    # ---- Step Agent analysis ----
    if insight.step_agents:
        print(f"  ↳ Step Agents: {[(sa.name, sa.source) for sa in insight.step_agents]}")

    assert "glossary_agent" in insight.tools_detected, (
        f"DEFINITIVE FAILURE: glossary_agent was NOT invoked for '{task}'. "
        f"Tools detected from conversation insight: {insight.tools_detected}. "
        f"The agent answered from general LLM knowledge, not the glossary tool. "
        f"Assistant messages: {[m['content'][:100] for m in insight.assistant_messages]}"
    )

    # ==== LAYER 2: Output quality checks ====
    output_lower = response.text.lower()
    # Normalize hyphens to spaces for flexible keyword matching
    # (agent may say "Person-to-Person" or "Person to Person")
    output_normalized = output_lower.replace("-", " ")

    # ---- Hard assertion: must NOT contain general-knowledge hedging ----
    # Check this FIRST — if the agent hedges, the glossary was not used
    for phrase in _GENERAL_KNOWLEDGE_INDICATORS:
        assert phrase not in output_lower, (
            f"Glossary agent NOT used — general knowledge detected. "
            f"Found hedging phrase: '{phrase}'. Response: {response.text[:300]}"
        )

    # ---- Hard assertion: ALL required keywords must be present ----
    for kw in required_keywords:
        kw_normalized = kw.replace("-", " ")
        assert kw_normalized in output_normalized, (
            f"Glossary definition missing required keyword '{kw}' for term '{term}'. "
            f"Got: {response.text[:300]}"
        )

    # ---- Hard assertion: at least ONE definition variant must match ----
    definition_variants = case.get("definition_variants", [])
    if definition_variants:
        found_variant = [v for v in definition_variants if v in output_normalized]
        assert found_variant, (
            f"No recognized definition variant for '{term}'. "
            f"Expected one of {definition_variants}. "
            f"Got: {response.text[:300]}"
        )
        print(f"  \u21b3 Definition variant matched: {found_variant}")

    # ---- Hard assertion: at least ONE domain keyword must be present ----
    if domain_keywords:
        found_domain = [kw for kw in domain_keywords if kw in output_normalized]
        assert found_domain, (
            f"No domain-specific context found for '{term}'. "
            f"Expected at least one of {domain_keywords}. "
            f"Got: {response.text[:300]}"
        )
        print(f"  ↳ Domain keywords found: {found_domain}")

    # ---- LLM judge: GlossarySourceMetric (threshold 0.8) ----
    test_case = LLMTestCase(
        input=task,
        actual_output=response.text,
        expected_output=f"A single authoritative glossary definition of {term}: {expected_snippet}",
        context=[
            f"The term '{term}' is defined in the marketing glossary.",
            f"The glossary definition includes: {expected_snippet}.",
            "The agent MUST use the Glossary agent tool for definition questions.",
            "The agent must NOT fall back to general LLM knowledge.",
            "The response should provide one authoritative definition, not multiple possibilities.",
        ],
    )

    judge = _get_judge()
    glossary_metric = GlossarySourceMetric(threshold=0.8, model=judge)
    hallucination_metric = HallucinationMetric(threshold=0.5, model=judge)

    assert_test(test_case, [glossary_metric, hallucination_metric])

