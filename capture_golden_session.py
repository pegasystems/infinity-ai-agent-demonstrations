"""
Passive Golden Session Recorder — Capture from Pega UI History

Instead of driving the conversation from the command line, this script lets you:

  1. Run the full Surface campaign flow through the Pega UI (chat interface)
  2. Copy the conversation ID (PXCONV-XXXXX) from the browser
  3. Run this script — it pulls the ENTIRE conversation history from
     D_pxAutopilotConversation and reconstructs a golden session JSON

The output is identical in format to record_golden_session.py so it works
directly with test_golden_session.py for replay & regression testing.

Usage:
    # Capture from a single conversation
    python3 capture_golden_session.py PXCONV-12345

    # Capture and provide a friendly name
    python3 capture_golden_session.py PXCONV-12345 --name "Full Campaign with Zelle PDF"

    # Capture from multiple conversations (e.g., a multi-session flow)
    python3 capture_golden_session.py PXCONV-12345 PXCONV-12346

    # Custom output directory
    python3 capture_golden_session.py PXCONV-12345 -o golden_sessions/v8

    # List recent conversations (last 24h) to find the right ID
    python3 capture_golden_session.py --list-recent

How to find your conversation ID:
    - In the Surface chat UI, open browser DevTools → Network tab
    - Look for requests to /api/insight/conversation/<ID>
    - Or check the URL bar / conversation panel for PXCONV-XXXXX
    - Or look at the agent card response contextId field
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Environment variable that points to the active project configuration file.
# This allows tests and the capture script to share the same config.
_PROJECT_CONFIG_ENV = "PROJECT_CONFIG"

# ---------------------------------------------------------------------------
# Imports from the test suite
# ---------------------------------------------------------------------------
_this_dir = Path(__file__).resolve().parent
_templates_dir = _this_dir / "project_templates"
sys.path.insert(0, str(_this_dir))
load_dotenv(dotenv_path=_this_dir / ".env")

from test_surface_agents import (
    AGENT_CARD_URL,
    CLIENT_ID,
    CLIENT_SECRET,
    TOKEN_URL,
    ConversationInsight,
    PegaInsight,
    StepAgentExecution,
    _detect_tools_from_messages,
    _detect_step_agents_from_content,
    _INTERNAL_TOOL_NAMES,
    # Profile-aware pattern builders
    build_tool_patterns_from_profile,
    # Structured tool extraction (for new agent output format)
    extract_tools_from_conversation_history,
    extract_tools_from_plugins,
    extract_available_tools_from_metrics,
    build_tool_registry,
    detect_tools_hybrid,
    extract_per_turn_tools,
)

# Regex to detect case creation from response text: "CaseType case (PREFIX-NNNN)"
_CASE_TYPE_RE = re.compile(r"(\w[\w\s]+?)\s+case\s*\(([A-Z]+-\d+)\)")


def _extract_case_type_from_tool_args(tool_details: List[Dict[str, Any]]) -> Optional[str]:
    """Extract case type class name from pxCreateCaseWithAssignmentDetails tool arguments."""
    for td in tool_details:
        if td.get("name") == "pxCreateCaseWithAssignmentDetails":
            args_str = td.get("arguments", "")
            if args_str:
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    class_name = args.get("CaseTypeClassName", "")
                    if class_name:
                        return class_name
                except (json.JSONDecodeError, TypeError):
                    pass
    return None


def _auto_extract_case_type(
    response_text: str,
    tool_details: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Auto-detect case type for golden annotation.

    Checks tool arguments first (CaseTypeClassName from pxCreateCaseWithAssignmentDetails),
    then falls back to response text regex pattern "XYZ case (PREFIX-NNNN)".
    """
    if tool_details:
        class_name = _extract_case_type_from_tool_args(tool_details)
        if class_name:
            return class_name
    m = _CASE_TYPE_RE.search(response_text)
    return m.group(1).strip() if m else None


# ============================================================================
# Project config helpers
# ============================================================================


def load_project_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load a project configuration JSON file.

    Resolution order:
      1. Explicit *path* argument
      2. ``PROJECT_CONFIG`` environment variable
      3. ``project_config.json`` in this directory
      4. First ``project_config.*.json`` glob match in project_templates directory
      5. Empty dict (all defaults)
    """
    candidates: List[Path] = []
    if path:
        candidates.append(Path(path))
    env_path = os.environ.get(_PROJECT_CONFIG_ENV)
    if env_path:
        candidates.append(Path(env_path))
        # Also check in project_templates if env_path is just a filename
        if not Path(env_path).is_absolute():
            candidates.append(_templates_dir / env_path)
    candidates.append(_templates_dir / "project_config.json")

    for p in candidates:
        if p.exists():
            with open(p) as f:
                print(f"[Config] Loaded project config: {p}")
                return json.load(f)

    # Glob fallback: pick first project_config.*.json (skips template)
    glob_hits = sorted(_templates_dir.glob("project_config.*.json"))
    for p in glob_hits:
        if "template" not in p.name:
            with open(p) as f:
                print(f"[Config] Loaded project config (glob): {p.name}")
                return json.load(f)

    print("[Config] No project config found — using defaults")
    return {}


def _normalize_workflows(project_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the normalized workflows list from a project config.

    Accepts both the new ``workflows`` array and the legacy ``workflow`` object,
    coercing the latter to a single-element list with id="default".
    Returns an empty list when neither key is present.
    """
    if "workflows" in project_config:
        return project_config["workflows"]
    legacy = project_config.get("workflow")
    if legacy:
        return [{
            "id": "default",
            "description": legacy.get("description", ""),
            "stages": legacy.get("stages", []),
        }]
    return []


def _auto_wait_for_pattern(response_text: str, turn_num: int) -> Optional[str]:
    """Derive a ``wait_for_pattern`` regex for a turn from the assistant response.

    Strategy: extract 2-3 distinctive keyword phrases from the response that
    together uniquely identify the workflow stage.  We prefer:
      - Section headers (## headings)
      - Binary-choice prompts ("Yes / No", "Move forward")
      - Stage-transition language ("approved", "assembled", "creation")
    """
    alternatives: List[str] = []

    # 1. Markdown headings
    headings = re.findall(r"^##\s+(.{10,80})$", response_text, re.MULTILINE)
    for h in headings[:2]:
        # simplify to a few key words
        words = re.findall(r"[a-zA-Z]{4,}", h)
        if len(words) >= 2:
            alternatives.append(r".*".join(re.escape(w) for w in words[:3]))

    # 2. Choice prompts
    choices = re.findall(r"^-\s+(.{3,60})$", response_text, re.MULTILINE)
    for c in choices[:2]:
        words = re.findall(r"[a-zA-Z]{4,}", c)
        if words:
            alternatives.append(re.escape(words[0]))

    # 3. Stage-transition keywords
    transition_keywords = [
        r"approved", r"submitted", r"created", r"assembled",
        r"content.*variation", r"audience.*reach", r"3rd party",
        r"campaign.*case", r"document", r"upload", r"gradial",
        r"content.*assembly", r"final.*review", r"waterfall",
    ]
    text_lower = response_text.lower()
    for kw in transition_keywords:
        if re.search(kw, text_lower):
            alternatives.append(kw)
            if len(alternatives) >= 4:
                break

    if not alternatives:
        return None

    # De-duplicate while preserving order
    seen: set = set()
    unique: List[str] = []
    for a in alternatives:
        key = a.lower()
        if key not in seen:
            seen.add(key)
            unique.append(a)
        if len(unique) >= 3:
            break

    return "|".join(unique) if unique else None


def _auto_gate_timeout(latency_ms: float) -> Optional[int]:
    """Return a gate_timeout override for slow turns, else None (use default)."""
    if latency_ms > 30_000:
        return max(int(latency_ms / 1000 * 3), 120)
    return None


def _build_profile(
    project_config: Dict[str, Any],
    session: Dict[str, Any],
    conversations: List[ConversationInsight],
    workflow_id: Optional[str] = None,
    case_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a complete evaluation profile by merging user config with auto-sensed data.

    The profile is saved alongside the golden JSON and loaded by the test suite.
    It contains everything the 8 tests need to evaluate any Pega agent project.
    """
    # ---- Connection (from config, fall back to env) ----
    conn = project_config.get("connection", {})
    profile: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "project_name": project_config.get("project_name", session.get("name", "Unknown")),
        "version": project_config.get("version", "1.0"),
        "connection": {
            "base_url": conn.get("base_url", os.environ.get("AGENTX_BASE_URL", "")),
            "agent_name": conn.get("agent_name", os.environ.get("AGENT_NAME", "")),
            "a2a_app_path": conn.get("a2a_app_path", ""),
            "token_url_override": conn.get("token_url_override"),
        },
    }

    # ---- Agent identity (from config — cannot be auto-sensed) ----
    identity = project_config.get("agent_identity", {})
    profile["agent_identity"] = {
        "role": identity.get("role", "A Pega agent assistant."),
        "domain": identity.get("domain", "General"),
        "organization": identity.get("organization", ""),
        "off_topic_guidance": identity.get("off_topic_guidance", ""),
    }

    # ---- Workflow stages (auto-sensed from stages API, merged with config) ----
    auto_stages: List[Dict[str, str]] = []
    for conv in conversations:
        for stage in conv.case_stages:
            auto_stages.append({
                "name": stage.get("name", ""),
                "status": stage.get("status", "unknown"),
                "id": stage.get("id", ""),
            })
    workflows = _normalize_workflows(project_config)
    active_wf: Dict[str, Any] = {}
    if workflow_id:
        active_wf = next((w for w in workflows if w.get("id") == workflow_id), {})
        if not active_wf:
            print(f"[Profile] WARNING: workflow_id '{workflow_id}' not found in config; workflow will be empty.")
    elif len(workflows) == 1:
        active_wf = workflows[0]
        workflow_id = active_wf.get("id")
    profile["workflow"] = {
        "id": workflow_id,
        "description": active_wf.get("description", ""),
        "stages_from_config": active_wf.get("stages", []),
        "stages_auto_sensed": auto_stages,
    }

    # ---- Hallucination context (from config — cannot be auto-sensed) ----
    profile["hallucination_context"] = project_config.get("hallucination_context", [])

    # ---- Tool patterns (merge config supplementals with auto-sensed) ----
    auto_tool_patterns: List[Dict[str, str]] = []
    auto_tool_names: set = set()
    for conv in conversations:
        for tool_name in conv.tools_detected:
            if tool_name not in auto_tool_names:
                auto_tool_names.add(tool_name)
                auto_tool_patterns.append({
                    "tool": tool_name,
                    "source": "auto_sensed",
                })
    config_tool_patterns = project_config.get("tool_patterns", {}).get("patterns", [])
    config_tool_labels = project_config.get("tool_patterns", {}).get("labels", {})
    profile["tool_patterns"] = {
        "from_config": config_tool_patterns,
        "auto_sensed": auto_tool_patterns,
        "labels": config_tool_labels,
    }

    # ---- Step agent patterns (merge config with auto-sensed) ----
    auto_step_agents: List[Dict[str, str]] = []
    auto_sa_names: set = set()
    for conv in conversations:
        for sa in conv.step_agents:
            if sa.name not in auto_sa_names:
                auto_sa_names.add(sa.name)
                auto_step_agents.append({
                    "agent": sa.name,
                    "source": sa.source,
                    "status": sa.status,
                })
    config_sa_patterns = project_config.get("step_agent_patterns", {}).get("patterns", [])
    profile["step_agent_patterns"] = {
        "from_config": config_sa_patterns,
        "auto_sensed": auto_step_agents,
    }

    # ---- Silent upload patterns (from config) ----
    profile["silent_upload_patterns"] = project_config.get("silent_upload_patterns", {}).get("patterns", [])

    # ---- Step agent context (case ID metadata) ----
    profile["step_agent_context"] = {
        "captured_case_id": case_id,
        "agent_type": project_config.get("agent_type", "conversational"),
    }

    return profile


# ============================================================================
# Turn-pair reconstruction
# ============================================================================


def _overlay_supplement_tools(
    turns: List[Dict[str, Any]],
    supplement_per_turn: List[Dict[str, Any]],
) -> None:
    """Overlay tool detections from structured agent output onto captured turns.

    The *supplement_per_turn* list comes from ``extract_per_turn_tools`` run on
    the Pega agent execution output JSON.  It contains the **correct** tool
    invocations per user turn extracted from explicit ``tool_calls`` arrays.

    This function matches supplementary turns to captured turns by index and
    replaces the regex-detected tools with the structurally detected ones.
    """
    # The supplement may contain a different number of turns if the structured
    # output includes internal prompts (e.g., repeated initial prompts).
    # We match by user input text first, falling back to index alignment.
    matched = 0
    for turn in turns:
        turn_input = turn.get("input", "").strip()
        if not turn_input:
            continue

        # Try exact match on user_input text
        best_match: Optional[Dict[str, Any]] = None
        for spt in supplement_per_turn:
            if spt.get("user_input", "").strip() == turn_input:
                best_match = spt
                break

        # Fallback: match by turn index (1-indexed turn vs per_turn_tools 0-indexed)
        if not best_match:
            idx = turn["turn"] - 1
            if 0 <= idx < len(supplement_per_turn):
                best_match = supplement_per_turn[idx]

        if best_match and best_match.get("tools_invoked"):
            old_tools = turn.get("expected_tools", [])
            new_tools = best_match["tools_invoked"]
            if old_tools != new_tools:
                print(f"  [Supplement] Turn {turn['turn']}: "
                      f"{old_tools or '(none)'} → {new_tools}")
            turn["expected_tools"] = new_tools
            turn["insight"]["tools_detected"] = new_tools
            if "tool_details" in best_match:
                turn["insight"]["tool_details"] = best_match["tool_details"]
            matched += 1

    print(f"  [Supplement] Overlaid tools on {matched}/{len(turns)} turns")


def _pair_user_assistant_turns(
    messages: List[Dict[str, Any]],
    tool_patterns: Optional[List[tuple]] = None,
) -> List[Dict[str, Any]]:
    """Reconstruct turn pairs from a flat or structured message list.

    Handles two formats:
      1. **Flat (D_pxAutopilotConversation)**: roles are "user" / "assistant"
         only.  Tool detection relies on regex patterns applied to content.
      2. **Structured (agent execution output)**: roles include "system",
         "assistant" (with ``tool_calls``), and "tool".  Tool detection uses
         the explicit ``tool_calls`` arrays first.

    We pair each USER message with all non-user messages that follow it to
    form logical "turns".  If multiple ASSISTANT messages follow a single
    USER message (e.g., tool thinking + final answer), the *content-bearing*
    responses are concatenated into one turn.
    """
    turns: List[Dict[str, Any]] = []
    i = 0

    while i < len(messages):
        msg = messages[i]

        # Skip leading system messages
        if msg.get("role") == "system":
            i += 1
            continue

        if msg["role"] == "user":
            user_msg = msg
            # Collect all non-user messages that follow before the next USER
            assistant_parts: List[Dict[str, Any]] = []
            j = i + 1
            while j < len(messages) and messages[j].get("role") != "user":
                assistant_parts.append(messages[j])
                j += 1

            # Combine assistant responses (only content-bearing messages)
            if assistant_parts:
                combined_text = "\n\n".join(
                    p["content"] for p in assistant_parts
                    if p.get("content") and p.get("role") in ("assistant", None)
                )
                # Detect tools — _detect_tools_from_messages now checks for
                # tool_calls arrays first, then falls back to regex patterns.
                turn_tools = _detect_tools_from_messages(assistant_parts, tool_patterns)
                # Detect step agents from content
                turn_step_agents = _detect_step_agents_from_content(assistant_parts)

                # Detect file attachment from user message text
                file_attachment = _detect_file_attachment(user_msg["content"])

                turns.append({
                    "turn": len(turns) + 1,
                    "input": user_msg["content"],
                    "description": _auto_describe_turn(
                        user_msg["content"], combined_text, turn_tools, len(turns) + 1
                    ),
                    "expected_tools": turn_tools,
                    "expected_case_type": _auto_extract_case_type(combined_text),
                    "file_attachment": file_attachment,
                    "response": {
                        "text": combined_text,
                        "context_id": None,  # Filled in later
                        "message_id": assistant_parts[-1].get("message_id", ""),
                        "latency_ms": _estimate_latency(user_msg, assistant_parts[-1]),
                    },
                    "insight": {
                        "tools_detected": turn_tools,
                        "step_agents": [asdict(sa) for sa in turn_step_agents],
                    },
                    "user_timestamp": user_msg.get("timestamp", ""),
                    "assistant_timestamp": assistant_parts[-1].get("timestamp", ""),
                    "assistant_message_count": len(assistant_parts),
                })
            else:
                # USER message with no response (might be the last message)
                turns.append({
                    "turn": len(turns) + 1,
                    "input": user_msg["content"],
                    "description": f"Turn {len(turns) + 1} (no response yet)",
                    "expected_tools": [],
                    "expected_case_type": None,
                    "response": {
                        "text": "",
                        "context_id": None,
                        "message_id": "",
                        "latency_ms": 0,
                    },
                    "insight": {"tools_detected": [], "step_agents": []},
                    "user_timestamp": user_msg.get("timestamp", ""),
                    "assistant_timestamp": "",
                    "assistant_message_count": 0,
                })

            i = j  # Skip past the assistant messages we consumed
        else:
            # Leading assistant message with no preceding user turn.
            # This can be:
            #   a) A welcome/greeting before any user input
            #   b) A response to a silent file upload (user uploaded via UI without text)
            #
            # We detect case (b) by looking for file-upload cues in the assistant text,
            # e.g. "uploaded the document", "received your", "approval document".
            if not turns or turns[-1]["input"]:
                asst_text = msg.get("content", "")
                silent_file = _detect_silent_file_upload(asst_text, len(turns))

                turns.append({
                    "turn": len(turns) + 1,
                    "input": silent_file["synthetic_input"] if silent_file else "",
                    "description": (
                        silent_file["description"] if silent_file
                        else "Agent greeting / welcome message"
                    ),
                    "expected_tools": [],
                    "expected_case_type": None,
                    "file_attachment": silent_file["file_attachment"] if silent_file else None,
                    "response": {
                        "text": asst_text,
                        "context_id": None,
                        "message_id": msg.get("message_id", ""),
                        "latency_ms": 0,
                    },
                    "insight": {"tools_detected": [], "step_agents": []},
                    "user_timestamp": "",
                    "assistant_timestamp": msg.get("timestamp", ""),
                    "assistant_message_count": 1,
                })
            i += 1

    return turns


# ============================================================================
# Structured Agent Output Parser
# ============================================================================
# Parse the new Pega agent output format that contains explicit tool_calls,
# plugins array, and rich metrics. This format is more reliable than the
# D_pxAutopilotConversation data view for tool detection.


def parse_structured_agent_output(
    agent_output: Dict[str, Any],
) -> Dict[str, Any]:
    """Parse structured Pega agent output into golden session format.

    This function handles the new agent output format that includes:
    - conversation_history with explicit tool_calls
    - plugins array with execution metadata
    - metrics with vector_store entries (available tools)

    Args:
        agent_output: The full Pega agent output JSON.

    Returns:
        Dict with:
        - turns: List of turn dicts compatible with golden session format
        - tool_registry: Dict of all discovered tools
        - agent_metadata: Extracted agent info (name, inputs, etc.)
        - detection_mode: "structured" to indicate source
    """
    result: Dict[str, Any] = {
        "turns": [],
        "tool_registry": {},
        "agent_metadata": {},
        "detection_mode": "structured",
    }

    # Extract agent metadata
    result["agent_metadata"] = {
        "name": agent_output.get("name", "unknown"),
        "last_event": agent_output.get("last_event", ""),
        "inputs": agent_output.get("inputs", {}),
    }

    # Build tool registry from all sources
    result["tool_registry"] = build_tool_registry(agent_output)

    # Extract per-turn tools
    per_turn_tools = extract_per_turn_tools(agent_output, include_internal=False)

    # Build turns from conversation_history
    for conv in agent_output.get("conversation_history", []):
        history = conv.get("history", [])
        i = 0
        turn_idx = 0

        while i < len(history):
            msg = history[i]

            # Skip system messages — they contain the agent prompt, not a turn
            if msg.get("role") == "system":
                i += 1
                continue

            if msg.get("role") == "user":
                user_input = msg.get("content", "")
                turn_idx += 1

                # Collect assistant response(s)
                assistant_texts: List[str] = []
                assistant_tools: List[str] = []
                tool_details: List[Dict[str, Any]] = []
                latency_ms = 0.0
                message_id = ""

                j = i + 1
                while j < len(history) and history[j].get("role") != "user":
                    asst_msg = history[j]

                    # Get final assistant content
                    if asst_msg.get("role") == "assistant":
                        content = asst_msg.get("content", "")
                        if content:
                            assistant_texts.append(content)
                            message_id = asst_msg.get("id", "")

                        # Extract latency from metrics
                        metrics = asst_msg.get("metrics", {})
                        if metrics.get("response_time"):
                            latency_ms += metrics["response_time"] * 1000

                        # Get tool_calls - include all tools including pega_context
                        for tc in asst_msg.get("tool_calls", []):
                            tool_name = tc.get("function", {}).get("name")
                            # Only exclude data_from_all_sources and complete_data_source as internal
                            if tool_name and tool_name not in ["data_from_all_sources", "complete_data_source"]:
                                if tool_name not in assistant_tools:
                                    assistant_tools.append(tool_name)
                                tool_details.append({
                                    "name": tool_name,
                                    "call_id": tc.get("id"),
                                    "arguments": tc.get("function", {}).get("arguments"),
                                })

                    # Tool responses (may have timing info)
                    if asst_msg.get("role") == "tool":
                        tool_metrics = asst_msg.get("metrics", {})
                        if tool_metrics.get("elapsed_time"):
                            latency_ms += tool_metrics["elapsed_time"] * 1000

                    j += 1

                # Build turn
                combined_text = "\n\n".join(assistant_texts) if assistant_texts else ""

                # Get tools from per_turn_tools if available
                if turn_idx <= len(per_turn_tools):
                    turn_tools_data = per_turn_tools[turn_idx - 1]
                    if turn_tools_data["tools_invoked"]:
                        assistant_tools = turn_tools_data["tools_invoked"]
                    if turn_tools_data.get("tool_details"):
                        tool_details = turn_tools_data["tool_details"]

                turn: Dict[str, Any] = {
                    "turn": turn_idx,
                    "input": user_input,
                    "description": _auto_describe_turn_from_structured(
                        user_input, combined_text, assistant_tools, turn_idx
                    ),
                    "expected_tools": assistant_tools,
                    "expected_case_type": _auto_extract_case_type(combined_text, tool_details),
                    "response": {
                        "text": combined_text,
                        "context_id": None,
                        "message_id": message_id,
                        "latency_ms": round(latency_ms, 1),
                    },
                    "insight": {
                        "tools_detected": assistant_tools,
                        "tool_details": tool_details,
                        "step_agents": [],
                    },
                }

                result["turns"].append(turn)
                i = j
            else:
                # Handle leading assistant message (greeting) — skip tool responses
                if msg.get("role") == "assistant" and not result["turns"]:
                    content = msg.get("content", "")
                    if content and not msg.get("tool_calls"):
                        result["turns"].append({
                            "turn": 0,
                            "input": "",
                            "description": "Agent greeting / welcome message",
                            "expected_tools": [],
                            "expected_case_type": None,
                            "response": {
                                "text": content,
                                "context_id": None,
                                "message_id": "",
                                "latency_ms": 0,
                            },
                            "insight": {"tools_detected": [], "step_agents": []},
                        })
                i += 1

    # Enrich with plugin execution data
    _enrich_turns_with_plugins(result["turns"], agent_output.get("plugins", []))

    return result


def _auto_describe_turn_from_structured(
    user_input: str,
    assistant_text: str,
    tools: List[str],
    turn_num: int,
) -> str:
    """Generate an auto-description for a turn based on content and tools."""
    # Use tool name if available
    if tools:
        primary_tool = tools[0]
        # Clean up tool name for display
        display_name = primary_tool.replace("Plugin", "").replace("Tool", "")
        display_name = re.sub(r"([a-z])([A-Z])", r"\1 \2", display_name)  # CamelCase -> spaces
        return f"Turn {turn_num}: {display_name}"

    # Use first few words of user input
    if user_input:
        words = user_input.split()[:6]
        summary = " ".join(words)
        if len(user_input.split()) > 6:
            summary += "..."
        return f"Turn {turn_num}: {summary}"

    return f"Turn {turn_num}"


def _enrich_turns_with_plugins(
    turns: List[Dict[str, Any]],
    plugins: List[Dict[str, Any]],
) -> None:
    """Enrich turn data with plugin execution details.

    Matches plugins to turns based on timing and updates insight data.
    """
    for plugin in plugins:
        plugin_name = plugin.get("name")
        start_time = plugin.get("start_time", 0)
        end_time = plugin.get("end_time", 0)

        # Find the turn this plugin was invoked in
        # (This is approximate - based on the plugin existing in expected_tools)
        for turn in turns:
            if plugin_name in turn.get("expected_tools", []):
                # Add execution timing
                if "plugin_executions" not in turn["insight"]:
                    turn["insight"]["plugin_executions"] = []

                turn["insight"]["plugin_executions"].append({
                    "name": plugin_name,
                    "id": plugin.get("id"),
                    "execution_time_ms": (end_time - start_time) * 1000 if start_time and end_time else None,
                    "prerequisites": list(plugin.get("prerequisites", {}).keys()),
                })
                break


def _extract_system_prompt(agent_output: Dict[str, Any]) -> str:
    """Return the first system-role message content from the agent output, or ''."""
    for conv in agent_output.get("conversation_history", []):
        for msg in conv.get("history", []):
            if msg.get("role") == "system":
                return msg.get("content", "")
    return ""


def capture_from_structured_output(
    agent_output: Dict[str, Any],
    session_name: Optional[str] = None,
    output_dir: str = "golden_sessions",
    project_config: Optional[Dict[str, Any]] = None,
    workflow_id: Optional[str] = None,
    case_id: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Capture a golden session from structured Pega agent output.

    This is the main entry point for capturing from the new agent output format.
    Use this instead of capture_from_conversation_ids when you have the full
    agent output JSON (e.g., from tracer or direct API response).

    Args:
        agent_output: The full Pega agent output JSON.
        session_name: Optional friendly name for the session file.
        output_dir: Directory to write output files.
        project_config: Optional project configuration dict.  When *None*,
            the function attempts to load a config via ``load_project_config()``.

    Returns:
        Tuple of (golden_session_path, profile_path).
    """
    parsed = parse_structured_agent_output(agent_output)

    # Generate filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_name = parsed["agent_metadata"].get("name", "unknown")
    safe_name = re.sub(r"[^\w\-]", "_", session_name or agent_name)
    base_name = f"golden_{safe_name}_{timestamp}"

    # Build golden session
    session: Dict[str, Any] = {
        "recorded_at": datetime.now().isoformat(),
        "capture_mode": "structured_output",
        "agent_name": agent_name,
        "detection_mode": "structured",
        "step_agent_case_id": case_id,
        "turns": parsed["turns"],
        "tool_registry": parsed["tool_registry"],
        "summary": {
            "turn_count": len(parsed["turns"]),
            "all_tools": list(parsed["tool_registry"].keys()),
            "invoked_tools": [
                name for name, data in parsed["tool_registry"].items()
                if data.get("invoked")
            ],
            "available_tools": [
                name for name, data in parsed["tool_registry"].items()
                if not data.get("invoked")
            ],
        },
    }

    # ---- Load project config (if not supplied) ----
    pcfg: Dict[str, Any] = project_config if project_config is not None else load_project_config()

    # ---- Connection (from config, fall back to env) ----
    conn = pcfg.get("connection", {})

    # ---- Agent identity (config → system prompt → defaults) ----
    identity = pcfg.get("agent_identity", {})
    system_prompt = _extract_system_prompt(agent_output)
    # If the config doesn't specify a role, try to derive one from the
    # system prompt (first 200 chars) or fall back to the agent name.
    default_role = f"The {agent_name} assistant."
    if system_prompt and not identity.get("role"):
        # Use the first sentence/paragraph of the system prompt as the role
        snippet = system_prompt[:300].split("\n\n")[0].strip()
        if snippet:
            default_role = snippet

    # ---- Workflow stages (config → auto-sensed empty for structured) ----
    _wf_id = workflow_id
    _workflows = _normalize_workflows(pcfg)
    _active_wf: Dict[str, Any] = {}
    if _wf_id:
        _active_wf = next((w for w in _workflows if w.get("id") == _wf_id), {})
        if not _active_wf:
            print(f"[Profile] WARNING: workflow_id '{_wf_id}' not found in config; workflow will be empty.")
    elif len(_workflows) == 1:
        _active_wf = _workflows[0]
        _wf_id = _active_wf.get("id")

    # Build profile
    profile: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "project_name": pcfg.get("project_name", session_name or agent_name),
        "version": pcfg.get("version", "1.0"),
        "connection": {
            "base_url": conn.get("base_url", os.environ.get("AGENTX_BASE_URL", "")),
            "agent_name": conn.get("agent_name", os.environ.get("AGENT_NAME", agent_name)),
            "a2a_app_path": conn.get("a2a_app_path", ""),
            "token_url_override": conn.get("token_url_override"),
        },
        "agent_identity": {
            "role": identity.get("role", default_role),
            "domain": identity.get("domain", "General"),
            "organization": identity.get("organization", ""),
            "off_topic_guidance": identity.get("off_topic_guidance", ""),
        },
        "workflow": {
            "id": _wf_id,
            "description": _active_wf.get("description", ""),
            "stages_from_config": _active_wf.get("stages", []),
            "stages_auto_sensed": [],
        },
        "hallucination_context": pcfg.get("hallucination_context", []),
        "tool_patterns": {
            "from_config": pcfg.get("tool_patterns", {}).get("patterns", []),
            "auto_sensed": [
                {"tool": name, "source": "structured_output", "invoked": data.get("invoked", False)}
                for name, data in parsed["tool_registry"].items()
            ],
            "labels": pcfg.get("tool_patterns", {}).get("labels", {}),
            "detection_mode": "structured",
        },
        "step_agent_patterns": {
            "from_config": pcfg.get("step_agent_patterns", {}).get("patterns", []),
            "auto_sensed": [],
        },
        "silent_upload_patterns": pcfg.get("silent_upload_patterns", {}).get("patterns", []),
        "step_agent_context": {
            "captured_case_id": case_id,
            "agent_type": pcfg.get("agent_type", "orchestration"),
        },
    }

    # Write files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    golden_filename = f"{base_name}.json"
    profile_filename = f"profile_{safe_name}_{timestamp}.json"

    golden_path = output_path / golden_filename
    profile_path = output_path / profile_filename

    # Cross-reference golden session and profile
    session["profile"] = profile_filename
    profile["golden_session"] = golden_filename

    with open(golden_path, "w") as f:
        json.dump(session, f, indent=2)
    print(f"[Capture] Wrote golden session: {golden_path}")

    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"[Capture] Wrote profile: {profile_path}")

    return golden_path, profile_path


def _estimate_latency(
    user_msg: Dict[str, Any], assistant_msg: Dict[str, Any]
) -> float:
    """Estimate latency from timestamps (if available).

    Returns milliseconds, or 0 if timestamps can't be parsed.
    """
    try:
        user_ts = user_msg.get("timestamp", "")
        asst_ts = assistant_msg.get("timestamp", "")
        if user_ts and asst_ts:
            # Pega timestamps: "2026-02-18T15:30:00.000Z" or similar
            from datetime import datetime as dt

            # Try ISO format
            fmt_options = [
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y%m%dT%H%M%S.%f GMT",
            ]
            for fmt in fmt_options:
                try:
                    t_user = dt.strptime(user_ts, fmt)
                    t_asst = dt.strptime(asst_ts, fmt)
                    delta_ms = (t_asst - t_user).total_seconds() * 1000
                    return max(delta_ms, 0)
                except ValueError:
                    continue
    except Exception:
        pass
    return 0.0


# Regex to detect file-attachment text from Pega UI conversation logs
_FILE_ATTACH_RE = re.compile(
    r"\[I have attached a file:\s*(?P<filename>[^\]]+)\]", re.IGNORECASE
)


def _detect_file_attachment(user_input: str) -> Optional[Dict[str, Any]]:
    """Detect file attachment metadata from user message text.

    When the Pega UI records a file upload in D_pxAutopilotConversation,
    the user message contains text like:
        (attached file)\n\n[I have attached a file: Zelle.pdf]

    This function extracts the filename and returns metadata that the
    replay framework can use to locate and upload the actual file.

    Returns:
        dict with filename, or None if this isn't a file turn.
    """
    m = _FILE_ATTACH_RE.search(user_input)
    if not m:
        return None

    filename = m.group("filename").strip()
    return {
        "filename": filename,
        "path": f"golden_sessions/attachments/{filename}",
        "note": (
            "Place the actual file at the 'path' above (relative to evaluate-surface/) "
            "so the golden session replay can upload it via AgentX."
        ),
    }


# Patterns in assistant text that indicate the preceding user action was a
# silent file upload (no user text in D_pxAutopilotConversation).
_SILENT_UPLOAD_PATTERNS = [
    # Early-flow: main campaign document upload
    (re.compile(r"uploaded the document|received your.*document|extracted.*from your", re.I),
     "campaign_doc", "Zelle.pdf"),
    # Late-flow: 3rd party approval document
    (re.compile(r"approval.*document|approval has been submitted|3rd party approval", re.I),
     "approval_doc", "Marketing_Approval_Document.docx"),
]


def _detect_silent_file_upload(
    assistant_text: str, current_turn_count: int
) -> Optional[Dict[str, Any]]:
    """Detect when an assistant message is responding to a silent file upload.

    In the Pega UI, users can upload files without typing any accompanying
    text.  D_pxAutopilotConversation records these as assistant-only messages
    (no preceding user row).  This function infers the upload from the
    assistant's response text and returns synthetic metadata so the golden
    record has a properly tagged file-upload turn.

    Returns:
        dict with synthetic_input, description, file_attachment — or None.
    """
    for pattern, upload_type, default_filename in _SILENT_UPLOAD_PATTERNS:
        if pattern.search(assistant_text):
            return {
                "synthetic_input": f"(attached file)\n\n[I have attached a file: {default_filename}]",
                "description": f"Turn {current_turn_count + 1} \u2014 Silent file upload ({default_filename})",
                "file_attachment": {
                    "filename": default_filename,
                    "path": f"golden_sessions/attachments/{default_filename}",
                    "note": (
                        f"Auto-detected silent file upload ({upload_type}). "
                        f"Place the actual file at the path above so replay can upload it."
                    ),
                },
            }
    return None


def _auto_describe_turn(
    user_input: str, assistant_text: str, tools: List[str], turn_num: int
) -> str:
    """Generate a human-readable description for a turn."""
    desc_parts: List[str] = [f"Turn {turn_num}"]

    input_lower = user_input.lower()

    # Campaign creation
    if "create" in input_lower and "campaign" in input_lower:
        desc_parts.append("Case creation")
    elif "proceed without" in input_lower:
        desc_parts.append("Skip document upload")
    elif "will provide" in input_lower or "upload" in input_lower:
        desc_parts.append("Document upload")
    elif input_lower in ("yes", "looks good", "approved", "correct", "confirm"):
        desc_parts.append("Confirm section")
    elif "forward" in input_lower or "move forward" in input_lower or "everything" in input_lower:
        desc_parts.append("Final decision / submit")
    elif "p2p" in input_lower or "define" in input_lower or "glossary" in input_lower:
        desc_parts.append("Glossary lookup")
    elif "create" in input_lower and "complaint" in input_lower and "case" in input_lower:
        desc_parts.append("Create Complaint Case")
    else:
        # Use first 40 chars of input
        desc_parts.append(user_input[:40].replace("\n", " "))

    # Add tool info
    if tools:
        tool_labels = {
            "pxCreateCaseWithAssignmentDetails": "Created Case",
            "SurfaceNewCampaignAutomation": "Campaign Automation",
            "pxPerformAssignment": "Processed Assignment",
            "glossary_agent": "Glossary",
            "pega_context": "Pega Context",
            "taxonomy_agent": "Taxonomy",
            "GetCaseStages": "Stages",
            "CreateComplaintCasePlugin": "Create Complaint Case",
        }
        labels = [tool_labels.get(t, t) for t in tools[:3]]
        desc_parts.append(f"[{', '.join(labels)}]")

    return " — ".join(desc_parts)


# ============================================================================
# Main capture logic
# ============================================================================


def capture_golden_session(
    conversation_ids: List[str],
    insight: PegaInsight,
    name: Optional[str] = None,
    output_dir: str = "golden_sessions",
    project_config: Optional[Dict[str, Any]] = None,
    supplement_agent_output: Optional[Dict[str, Any]] = None,
    workflow_id: Optional[str] = None,
    case_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Capture a golden session from one or more completed Pega conversations.

    Args:
        conversation_ids: List of PXCONV-XXXXX IDs to capture.
        insight: Authenticated PegaInsight client.
        name: Optional friendly name for the session.
        output_dir: Where to save the JSON file.
        project_config: Project-specific configuration.
        supplement_agent_output: Optional structured Pega agent output JSON.
            When provided, tool detection is taken from the explicit
            ``tool_calls`` arrays and ``plugins`` in this data instead of
            relying on regex patterns.  This dramatically improves accuracy
            for agents whose assistant text doesn't match the default
            Surface-campaign regex patterns.

    Returns:
        The golden session dict (also saved to disk).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Pre-compute per-turn tools from the supplementary agent output ---
    supplement_per_turn: Optional[List[Dict[str, Any]]] = None
    supplement_registry: Optional[Dict[str, Dict[str, Any]]] = None
    if supplement_agent_output:
        supplement_per_turn = extract_per_turn_tools(supplement_agent_output, include_internal=False)
        supplement_registry = build_tool_registry(supplement_agent_output)
        print(f"[Supplement] Loaded structured tool data: "
              f"{len(supplement_per_turn)} turns, "
              f"{len(supplement_registry)} tools in registry")

    all_turns: List[Dict[str, Any]] = []
    all_tools: set = set()
    all_step_agents: List[Dict] = []
    conversation_data: List[Dict[str, Any]] = []
    all_conversation_insights: List[ConversationInsight] = []
    pcfg = project_config or {}

    # Build profile-aware tool detection patterns (project config + defaults)
    tool_patterns = build_tool_patterns_from_profile(pcfg) if pcfg else None

    for conv_id in conversation_ids:
        print(f"\n{'=' * 70}")
        print(f"  Capturing conversation: {conv_id}")
        print(f"{'=' * 70}")

        # --- Query the full conversation from Pega ---
        conv: ConversationInsight = insight.query_conversation(conv_id)

        print(f"  Messages: {len(conv.messages)} total")
        print(f"  Assistant messages: {len(conv.assistant_messages)}")
        print(f"  Tools detected: {conv.tools_detected}")
        print(f"  Case key: {conv.business_case_key}")
        print(f"  Step agents: {len(conv.step_agents)}")

        # --- Reconstruct turn pairs ---
        turns = _pair_user_assistant_turns(conv.messages, tool_patterns=tool_patterns)

        # Fill in context_id on each turn
        for turn in turns:
            turn["response"]["context_id"] = conv_id

        # --- Overlay tools from supplementary structured data ---
        # If the user provided the Pega agent execution output JSON, use its
        # explicit tool_calls arrays instead of the regex-detected tools.
        if supplement_per_turn:
            _overlay_supplement_tools(turns, supplement_per_turn)

        # Merge insight data from the full conversation query
        # (the per-turn insight from _pair_user_assistant_turns has content-based
        #  tool detection; the full conversation insight adds stages + field data)
        if turns:
            # Compute conversation-level tools — prefer supplement data
            conv_tools = conv.tools_detected
            if supplement_registry:
                conv_tools = [
                    name for name, data in supplement_registry.items()
                    if data.get("invoked") and name not in _INTERNAL_TOOL_NAMES
                ]

            # Enrich the last turn with the full insight data
            turns[-1]["insight"] = {
                "conversation_id": conv.conversation_id,
                "status": conv.status,
                "message_count": len(conv.messages),
                "assistant_message_count": len(conv.assistant_messages),
                "tools_detected": conv_tools,
                "business_case_key": conv.business_case_key,
                "assignment_key": conv.assignment_key,
                "step_agents": [asdict(sa) for sa in conv.step_agents],
                "prefilled_fields": conv.prefilled_fields,
                "case_stages": conv.case_stages,
            }

        # Offset turn numbers if we're combining multiple conversations
        offset = len(all_turns)
        for turn in turns:
            turn["turn"] = offset + turn["turn"]

        all_turns.extend(turns)

        # Conversation-level tools — prefer supplement data
        if supplement_registry:
            invoked_tools = [
                name for name, data in supplement_registry.items()
                if data.get("invoked") and name not in _INTERNAL_TOOL_NAMES
            ]
            all_tools.update(invoked_tools)
        else:
            all_tools.update(conv.tools_detected)
        all_step_agents.extend(asdict(sa) for sa in conv.step_agents)
        all_conversation_insights.append(conv)

        # Conversation-level tools for metadata
        conv_level_tools = (
            [name for name, data in supplement_registry.items()
             if data.get("invoked") and name not in _INTERNAL_TOOL_NAMES]
            if supplement_registry
            else conv.tools_detected
        )

        conversation_data.append({
            "conversation_id": conv_id,
            "status": conv.status,
            "message_count": len(conv.messages),
            "business_case_key": conv.business_case_key,
            "tools_detected": conv_level_tools,
        })

    # --- Print turn summary ---
    print(f"\n{'=' * 70}")
    print(f"  RECONSTRUCTED TURNS ({len(all_turns)} total)")
    print(f"{'=' * 70}")
    for turn in all_turns:
        user_preview = turn["input"][:60].replace("\n", " ") if turn["input"] else "(greeting)"
        asst_preview = turn["response"]["text"][:60].replace("\n", " ") if turn["response"]["text"] else "(empty)"
        tools = turn.get("expected_tools", [])
        print(f"  Turn {turn['turn']:2d}: [{user_preview}]")
        print(f"           → [{asst_preview}]")
        if tools:
            print(f"           Tools: {tools}")

    # --- Build session JSON ---
    session: Dict[str, Any] = {
        "recorded_at": datetime.now().isoformat(),
        "capture_method": "passive_from_pega_ui",
        "name": name or f"Golden Session — {len(all_turns)} turns",
        "agent_card_url": AGENT_CARD_URL,
        "conversation_ids": conversation_ids,
        "step_agent_case_id": case_id,
        "conversations": conversation_data,
        "turns": all_turns,
        "summary": {
            "total_turns": len(all_turns),
            "total_latency_ms": round(
                sum(t["response"]["latency_ms"] for t in all_turns), 1
            ),
            "avg_latency_ms": round(
                sum(t["response"]["latency_ms"] for t in all_turns) / max(len(all_turns), 1), 1
            ),
            "context_id": conversation_ids[0] if conversation_ids else None,
            "final_case_key": next(
                (c["business_case_key"] for c in reversed(conversation_data) if c.get("business_case_key")),
                None,
            ),
            "all_tools_used": sorted(all_tools),
            "step_agent_count": len(all_step_agents),
            "all_step_agents": all_step_agents,
        },
    }

    # --- Auto-sense: add wait_for_pattern and gate_timeout to each turn ---
    for turn in all_turns:
        resp_text = turn["response"]["text"]
        latency = turn["response"]["latency_ms"]

        pattern = _auto_wait_for_pattern(resp_text, turn["turn"])
        if pattern:
            turn["wait_for_pattern"] = pattern

        timeout = _auto_gate_timeout(latency)
        if timeout:
            turn["gate_timeout"] = timeout

    # Record deepeval version
    try:
        import deepeval
        session["deepeval_version"] = deepeval.__version__
    except ImportError:
        pass

    # --- Save golden session JSON ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c for c in (name or "capture") if c.isalnum() or c in "-_ ")[:40].strip().replace(" ", "_")
    filename = f"golden_{safe_name}_{ts}.json"
    filepath = out_dir / filename

    # Add profile reference to the session before saving
    profile_filename = f"profile_{safe_name}_{ts}.json"
    session["profile"] = profile_filename

    with open(filepath, "w") as f:
        json.dump(session, f, indent=2, default=str)

    # --- Generate evaluation profile ---
    profile = _build_profile(pcfg, session, all_conversation_insights, workflow_id=workflow_id, case_id=case_id)
    profile["golden_session"] = filename
    profile_path = out_dir / profile_filename

    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2, default=str)

    print(f"\n{'#' * 70}")
    print(f"  GOLDEN SESSION SAVED: {filepath}")
    print(f"  EVAL PROFILE SAVED:  {profile_path}")
    print(f"  Turns:       {len(all_turns)}")
    print(f"  Case:        {session['summary']['final_case_key']}")
    print(f"  Tools:       {session['summary']['all_tools_used']}")
    print(f"  Step Agents: {session['summary']['step_agent_count']}")
    print(f"  Auto-gated:  {sum(1 for t in all_turns if t.get('wait_for_pattern'))} turns")
    print(f"{'#' * 70}")
    print(f"\n  Next step — run the replay test:")
    print(f"    pytest test_golden_session.py -v -s --golden {filepath}")

    return session


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Capture a Golden Session from a Pega conversation you already ran in the UI.\n\n"
            "Example:\n"
            "  python3 capture_golden_session.py PXCONV-12345\n"
            "  python3 capture_golden_session.py PXCONV-12345 --name 'Zelle Campaign Flow'\n"
            "  python3 capture_golden_session.py --from-json agent_output.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "conversation_ids",
        nargs="*",
        help="One or more conversation IDs to capture (e.g., PXCONV-12345)",
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default=None,
        help="Friendly name for this golden session",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="golden_sessions",
        help="Directory to save recorded sessions (default: golden_sessions/)",
    )
    parser.add_argument(
        "--project-config", "-p",
        type=str,
        default=None,
        help=(
            "Path to a project_config.*.json file with agent identity, workflow stages, "
            "and hallucination context. If omitted, looks for PROJECT_CONFIG env var or "
            "project_config.json in the script directory."
        ),
    )
    parser.add_argument(
        "--from-json", "-j",
        type=str,
        default=None,
        help=(
            "Path to a structured Pega agent output JSON file. "
            "This captures tools directly from the tool_calls and plugins arrays "
            "instead of using regex pattern matching on message content. "
            "Use this when you have the full agent output from tracer or API."
        ),
    )
    parser.add_argument(
        "--supplement-json", "-s",
        type=str,
        default=None,
        help=(
            "Path to a structured Pega agent output JSON file used to SUPPLEMENT "
            "a conversation-ID capture.  The conversation is still fetched from "
            "D_pxAutopilotConversation (for message text, timestamps, etc.), but "
            "tool detection is taken from the structured tool_calls arrays in "
            "this file.  This is the recommended approach when the default regex "
            "patterns don't match the agent's tools (e.g., UplusRetailBankAssistant)."
        ),
    )
    parser.add_argument(
        "--workflow-id", "-w",
        type=str,
        default=None,
        help=(
            "ID of the workflow this session exercises (must match a 'workflows[].id' "
            "in your project config). Omit for FAQ or no-workflow sessions. "
            "Example: --workflow-id complaint_resolution"
        ),
    )
    parser.add_argument(
        "--case-id",
        type=str,
        default=None,
        help=(
            "Pega case ID the step agent was operating on during capture (metadata only). "
            "Recorded in the golden session for documentation — during replay a different "
            "case ID will be needed. Example: --case-id 'UPLUS-FS-WORK P-168004'"
        ),
    )
    parser.add_argument(
        "--list-recent",
        action="store_true",
        help="(placeholder) List recent conversations — requires Pega search API",
    )
    args = parser.parse_args()

    if args.list_recent:
        print(
            "Listing recent conversations is not yet implemented.\n"
            "To find your conversation ID:\n"
            "  1. Open the Surface chat UI in your browser\n"
            "  2. Open DevTools → Network tab\n"
            "  3. Look for requests to /api/insight/conversation/<ID>\n"
            "  4. The ID looks like PXCONV-12345"
        )
        return

    # --- Handle structured JSON input ---
    if args.from_json:
        json_path = Path(args.from_json)
        if not json_path.exists():
            print(f"[ERROR] JSON file not found: {args.from_json}")
            sys.exit(1)

        print(f"[Capture] Loading structured agent output: {json_path}")
        with open(json_path) as f:
            agent_output = json.load(f)

        # Load project config for profile enrichment
        pcfg = load_project_config(args.project_config)

        golden_path, profile_path = capture_from_structured_output(
            agent_output=agent_output,
            session_name=args.name,
            output_dir=args.output_dir,
            project_config=pcfg,
            workflow_id=args.workflow_id,
            case_id=args.case_id,
        )
        print(f"\n[Done] Golden session: {golden_path}")
        print(f"[Done] Profile: {profile_path}")
        return

    if not args.conversation_ids:
        parser.print_help()
        print("\n[ERROR] Provide at least one conversation ID or use --from-json.")
        print("  Example: python3 capture_golden_session.py PXCONV-12345")
        print("  Example: python3 capture_golden_session.py --from-json agent_output.json")
        print("  Example: python3 capture_golden_session.py PXCONV-12345 --supplement-json agent_output.json")
        sys.exit(1)

    # --- Load supplementary structured data (for tool overlay) ---
    supplement_data = None
    if args.supplement_json:
        supp_path = Path(args.supplement_json)
        if not supp_path.exists():
            print(f"[ERROR] Supplement JSON file not found: {args.supplement_json}")
            sys.exit(1)
        print(f"[Capture] Loading supplementary agent output: {supp_path}")
        with open(supp_path) as f:
            supplement_data = json.load(f)

    # --- Initialize Pega insight client ---
    base_url = os.environ.get("AGENTX_BASE_URL", "https://genai-cdh-demo.pega.net")
    insight = PegaInsight(
        base_url=base_url,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        token_url=TOKEN_URL,
    )

    # --- Load project config ---
    pcfg = load_project_config(args.project_config)

    capture_golden_session(
        conversation_ids=args.conversation_ids,
        insight=insight,
        name=args.name,
        output_dir=args.output_dir,
        project_config=pcfg,
        supplement_agent_output=supplement_data,
        workflow_id=args.workflow_id,
        case_id=args.case_id,
    )


if __name__ == "__main__":
    main()
