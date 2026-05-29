"""
Golden Session Recorder for Pega Surface Agent

Drives a multi-turn A2A conversation through the full Surface campaign flow,
capturing rich Pega-specific data at every turn:

  - Agent response text, contextId, messageId, latency
  - Conversation insight (D_pxAutopilotConversation)
  - Tool invocations detected from assistant messages
  - Step agent executions (field prefills, content patterns, stages API)
  - Case stages / steps progression
  - Pre-filled field values from document extraction

The recorded session is saved as a JSON file that can be:
  1. Replayed as a DeepEval ConversationalTestCase regression test
  2. Diff'd against future runs to detect agent regressions
  3. Used as a golden benchmark for the full campaign creation workflow

Usage:
    # Record a "proceed without document" flow
    python3 record_golden_session.py

    # Record with document upload (provide PDF path)
    python3 record_golden_session.py --with-document test_data/Zelle.pdf

    # Custom output directory
    python3 record_golden_session.py --output-dir golden_sessions/v8

    # Interactive mode — you type each message
    python3 record_golden_session.py --interactive
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Ensure imports work from this directory
# ---------------------------------------------------------------------------
_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir))
load_dotenv(dotenv_path=_this_dir / ".env")

from test_surface_agents import (
    AGENT_CARD_URL,
    CLIENT_ID,
    CLIENT_SECRET,
    TOKEN_URL,
    AgentResponse,
    ConversationInsight,
    PegaInsight,
    StepAgentExecution,
    SurfaceAgent,
)

# ============================================================================
# Pre-defined campaign flow scripts
# ============================================================================

FLOW_WITHOUT_DOCUMENT: List[Dict[str, Any]] = [
    {
        "message": "Create a new campaign",
        "wait": 3,
        "description": "Case creation — agent creates campaign case and asks about document",
        "expected_tools": ["pxCreateCaseWithAssignmentDetails", "SurfaceNewCampaignAutomation"],
    },
    {
        "message": "Proceed without one",
        "wait": 4,
        "description": "Skip document upload — agent presents Section 1: Campaign Basics",
        "expected_tools": ["pxPerformAssignment", "GetAssignmentDetailsFromAgent"],
    },
    {
        "message": "Yes, that looks correct",
        "wait": 3,
        "description": "Confirm Section 1 — agent presents Section 2: Goals & Metrics",
        "expected_tools": [],
    },
    {
        "message": "Looks good",
        "wait": 3,
        "description": "Confirm Section 2 — agent presents Section 3: Channels & Distribution",
        "expected_tools": [],
    },
    {
        "message": "Yes",
        "wait": 3,
        "description": "Confirm Section 3 — agent presents Section 4: Offer & Creative",
        "expected_tools": [],
    },
    {
        "message": "Approved",
        "wait": 3,
        "description": "Confirm Section 4 — agent presents Section 5: Compliance",
        "expected_tools": [],
    },
    {
        "message": "Yes",
        "wait": 3,
        "description": "Confirm Section 5 — agent presents Section 6: Final Decision",
        "expected_tools": [],
    },
    {
        "message": "Everything is correct, move forward",
        "wait": 4,
        "description": "Section 6 — Final Decision gate; agent submits assignment",
        "expected_tools": ["pxPerformAssignment", "PerformNonBackToBackAssignments"],
    },
]

FLOW_WITH_DOCUMENT: List[Dict[str, Any]] = [
    {
        "message": "Create a new campaign",
        "wait": 3,
        "description": "Case creation — agent creates campaign case and asks about document",
        "expected_tools": ["pxCreateCaseWithAssignmentDetails", "SurfaceNewCampaignAutomation"],
    },
    {
        "message": "Will provide document",
        "wait": 3,
        "description": "Indicate document upload — agent shows upload assignment",
        "expected_tools": ["pxPerformAssignment"],
        # NOTE: actual file upload requires extending SurfaceAgent with multipart support.
        # For now, you can manually upload through the UI and capture the flow from here.
    },
    # After upload, the GenAI step agent extracts fields and the flow continues
    # with the same confirmation sections as FLOW_WITHOUT_DOCUMENT[2:]
]


# ============================================================================
# Recorder
# ============================================================================


class GoldenSessionRecorder:
    """Drives a multi-turn Surface session and captures everything."""

    def __init__(
        self,
        agent: SurfaceAgent,
        insight: PegaInsight,
        output_dir: str = "golden_sessions",
    ):
        self.agent = agent
        self.insight = insight
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session: Dict[str, Any] = {
            "recorded_at": datetime.now().isoformat(),
            "agent_card_url": AGENT_CARD_URL,
            "deepeval_version": None,
            "turns": [],
            "summary": {},
        }
        self._context_id: Optional[str] = None

        # Record deepeval version
        try:
            import deepeval
            self.session["deepeval_version"] = deepeval.__version__
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Core: send one turn and capture everything
    # ------------------------------------------------------------------

    def record_turn(
        self,
        message: str,
        description: str = "",
        expected_tools: Optional[List[str]] = None,
        wait_seconds: float = 2.0,
    ) -> Dict[str, Any]:
        """Send a message and capture the full turn snapshot."""

        turn_num = len(self.session["turns"]) + 1
        print(f"\n{'=' * 70}")
        print(f"  TURN {turn_num}: {description or message}")
        print(f"{'=' * 70}")

        # --- Send via A2A (multi-turn: reuse contextId) ---
        response: AgentResponse = self.agent.run(
            task=message,
            context_id=self._context_id,
        )

        # Capture contextId from first turn
        if not self._context_id and response.context_id:
            self._context_id = response.context_id
            print(f"  [Session] contextId established: {self._context_id}")

        # --- Wait for Pega to finalize conversation state ---
        time.sleep(wait_seconds)

        # --- Query conversation insight ---
        insight: Optional[ConversationInsight] = None
        insight_data: Dict[str, Any] = {}

        if self._context_id:
            try:
                insight = self.insight.query_conversation(self._context_id)
                insight_data = {
                    "conversation_id": insight.conversation_id,
                    "status": insight.status,
                    "message_count": len(insight.messages),
                    "assistant_message_count": len(insight.assistant_messages),
                    "tools_detected": insight.tools_detected,
                    "business_case_key": insight.business_case_key,
                    "assignment_key": insight.assignment_key,
                    "step_agents": [asdict(sa) for sa in insight.step_agents],
                    "prefilled_fields": insight.prefilled_fields,
                    "case_stages": insight.case_stages,
                }
            except Exception as e:
                print(f"  [Warning] Insight query failed: {e}")
                insight_data = {"error": str(e)}

        # --- Build turn snapshot ---
        turn: Dict[str, Any] = {
            "turn": turn_num,
            "description": description,
            "input": message,
            "expected_tools": expected_tools or [],
            "response": {
                "text": response.text,
                "context_id": response.context_id,
                "message_id": response.message_id,
                "latency_ms": round(response.latency_ms, 1),
            },
            "insight": insight_data,
        }

        # Include last N assistant messages (for review)
        if insight and insight.assistant_messages:
            turn["latest_assistant_messages"] = [
                {
                    "content": m["content"][:500],
                    "message_id": m["message_id"],
                    "timestamp": m.get("timestamp", ""),
                }
                for m in insight.assistant_messages[-3:]
            ]

        self.session["turns"].append(turn)

        # --- Console summary ---
        print(f"  [Response] {response.text[:150]}...")
        print(f"  [Latency]  {response.latency_ms:.0f}ms")
        if insight:
            print(f"  [Tools]    {insight.tools_detected or 'none'}")
            if insight.business_case_key:
                print(f"  [Case]     {insight.business_case_key}")
            if insight.step_agents:
                for sa in insight.step_agents:
                    print(f"  [StepAgent] {sa.name} ({sa.source})")
            if insight.prefilled_fields:
                print(f"  [Prefilled] {list(insight.prefilled_fields.keys())}")

        return turn

    # ------------------------------------------------------------------
    # Run a scripted flow
    # ------------------------------------------------------------------

    def run_flow(self, flow: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a pre-defined flow and record all turns."""
        print(f"\n{'#' * 70}")
        print(f"  GOLDEN SESSION RECORDING — {len(flow)} turns")
        print(f"{'#' * 70}")

        for step in flow:
            self.record_turn(
                message=step["message"],
                description=step.get("description", ""),
                expected_tools=step.get("expected_tools"),
                wait_seconds=step.get("wait", 2),
            )

        return self._finalize()

    # ------------------------------------------------------------------
    # Interactive mode — user types each message
    # ------------------------------------------------------------------

    def run_interactive(self) -> Dict[str, Any]:
        """Interactive mode: user types each message at the terminal."""
        print(f"\n{'#' * 70}")
        print("  GOLDEN SESSION RECORDING — Interactive Mode")
        print("  Type your messages. Commands: 'done', 'undo', 'status'")
        print(f"{'#' * 70}")

        while True:
            try:
                user_input = input(f"\n[Turn {len(self.session['turns']) + 1}] You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Session interrupted.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("done", "quit", "exit"):
                break
            if user_input.lower() == "undo":
                if self.session["turns"]:
                    removed = self.session["turns"].pop()
                    print(f"  Removed turn: '{removed['input']}'")
                else:
                    print("  No turns to undo.")
                continue
            if user_input.lower() == "status":
                print(f"  Turns recorded: {len(self.session['turns'])}")
                if self._context_id:
                    print(f"  contextId: {self._context_id}")
                for t in self.session["turns"]:
                    print(f"    Turn {t['turn']}: {t['input'][:60]} → {t['response']['text'][:60]}...")
                continue

            desc = input("  Description (optional): ").strip() or user_input[:50]
            self.record_turn(message=user_input, description=desc)

        return self._finalize()

    # ------------------------------------------------------------------
    # Finalize and save
    # ------------------------------------------------------------------

    def _finalize(self) -> Dict[str, Any]:
        """Build summary and save to disk."""
        turns = self.session["turns"]

        if not turns:
            print("\n  No turns recorded. Nothing to save.")
            return self.session

        # Aggregate summary
        all_tools: set = set()
        all_step_agents: List[Dict] = []
        for t in turns:
            insight = t.get("insight", {})
            all_tools.update(insight.get("tools_detected", []))
            all_step_agents.extend(insight.get("step_agents", []))

        self.session["summary"] = {
            "total_turns": len(turns),
            "total_latency_ms": round(sum(t["response"]["latency_ms"] for t in turns), 1),
            "avg_latency_ms": round(
                sum(t["response"]["latency_ms"] for t in turns) / len(turns), 1
            ),
            "context_id": self._context_id,
            "final_case_key": turns[-1].get("insight", {}).get("business_case_key"),
            "all_tools_used": sorted(all_tools),
            "step_agent_count": len(all_step_agents),
            "all_step_agents": all_step_agents,
        }

        # Save JSON
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"golden_{ts}.json"
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(self.session, f, indent=2, default=str)

        print(f"\n{'#' * 70}")
        print(f"  GOLDEN SESSION SAVED: {filepath}")
        print(f"  Turns: {len(turns)}")
        print(f"  Total latency: {self.session['summary']['total_latency_ms']:.0f}ms")
        print(f"  Context: {self._context_id}")
        print(f"  Case: {self.session['summary']['final_case_key']}")
        print(f"  Tools: {self.session['summary']['all_tools_used']}")
        print(f"  Step Agents: {self.session['summary']['step_agent_count']}")
        print(f"{'#' * 70}")

        return self.session


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Record a Golden Session through the full Surface campaign flow"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode — type each message at the terminal",
    )
    parser.add_argument(
        "--with-document",
        type=str,
        default=None,
        help="Path to PDF for document-upload flow (not yet supported in A2A, recorded as metadata)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="golden_sessions",
        help="Directory to save recorded sessions (default: golden_sessions/)",
    )
    args = parser.parse_args()

    # --- Initialize clients ---
    print("Initializing Surface agent and Pega insight client...")
    agent = SurfaceAgent(AGENT_CARD_URL, CLIENT_ID, CLIENT_SECRET, TOKEN_URL)

    base_url = os.environ.get("AGENTX_BASE_URL", "https://genai-cdh-demo.pega.net")
    insight = PegaInsight(
        base_url=base_url,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        token_url=TOKEN_URL,
    )

    recorder = GoldenSessionRecorder(agent, insight, output_dir=args.output_dir)

    # --- Run ---
    if args.interactive:
        recorder.run_interactive()
    else:
        flow = FLOW_WITHOUT_DOCUMENT
        if args.with_document:
            print(f"  NOTE: Document upload path recorded as metadata: {args.with_document}")
            flow = FLOW_WITH_DOCUMENT
        recorder.run_flow(flow)


if __name__ == "__main__":
    main()
