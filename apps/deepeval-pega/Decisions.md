# Architectural Decisions

This document records the key design decisions made while building the DeepEval Surface testing framework, including context and trade-offs.

---

## ADR-001: A2A JSON-RPC as the Agent Protocol

**Decision:** Use the [Agent-to-Agent (A2A)](https://google.github.io/A2A/) JSON-RPC protocol to communicate with the Pega Surface agent.

**Context:** Pega Surface agents expose multiple interfaces — REST DX API, Constellation UI websockets, and the A2A JSON-RPC endpoint. We needed a protocol that mirrors how downstream systems (other agents, orchestration layers) would interact with the agent in production.

**Alternatives Considered:**
- **DX API REST calls** — Lower-level; would require manually managing conversation state and assignment flows
- **UI automation (Playwright)** — Brittle, slow, tests the UI not the agent
- **Direct LLM API calls** — Bypasses Pega orchestration entirely

**Trade-offs:**
- A2A is the standard inter-agent protocol and tests the full orchestration stack
- Requires OAuth2 authentication and agent card discovery
- Response format follows the A2A `message/send` spec with parts-based payloads

---

## ADR-002: DX API v2 for Conversation Insight (Not pyRole-Based)

**Decision:** After each agent turn, query `D_pxAutopilotConversation` via DX API v2 to verify tool invocations and conversation state.

**Context:** We needed definitive proof that specific tools (e.g., `glossary_agent`) were invoked — not just that the response *looks* correct. The SurfacePOC UI uses this same data view, making it the canonical source of truth.

**Key Finding:** The `pyMessages` array only contains `USER` and `ASSISTANT` role entries. There is no separate role for tool calls, function results, or step agents. Tool invocations must be inferred from assistant message content.

**Alternatives Considered:**
- **AI Tracer API** — Provides step-level trace data but is not exposed via public API
- **Constellation WebSocket events** — Would require a full UI session
- **Response text heuristics alone** — Unreliable; agent phrasing varies between runs

---

## ADR-003: Regex-Based Tool Detection (16 Patterns)

**Decision:** Detect tool invocations by applying regex patterns against assistant message content, ported from the SurfacePOC `insight-transformer.ts`.

**Context:** Since `D_pxAutopilotConversation` does not expose a structured `tools_used` field, we reverse-engineer tool usage from content patterns. The SurfacePOC UI already does this — we ported the same approach to Python for consistency.

**Pattern Examples:**
| Pattern | Detected Tool |
|---|---|
| `creat.*case.*\[S-\d+\]` | `pxCreateCaseWithAssignmentDetails` |
| `glossary\|terminolog\|acronym\|defin` | `glossary_agent` |
| `stage\|step\|progress` | `GetCaseStages` |

**Trade-offs:**
- Consistent with how the production UI interprets conversation data
- Patterns may need updates when agent response phrasing changes
- False positives possible but mitigated by specific regex anchoring

---

## ADR-004: Vertex AI Gemini 2.0 Flash as the Judge Model

**Decision:** Use Vertex AI Gemini 2.0 Flash (`gemini-2.0-flash-001`) as the LLM judge for DeepEval metrics.

**Context:** DeepEval metrics (AnswerRelevancy, Hallucination, custom GlossarySourceMetric) require an LLM to evaluate response quality. We needed a model accessible via Google Cloud ADC without additional API keys.

**Alternatives Considered:**
- **OpenAI GPT-4** — Requires a separate API key and billing; not available in all enterprise environments
- **Gemini 1.5 Pro** — Higher quality but significantly slower and more expensive for test-suite-scale evaluation
- **Local models (Ollama)** — Lower quality, inconsistent scoring

**Trade-offs:**
- Gemini 2.0 Flash is fast (~1-3s per evaluation) and cost-effective
- Uses Application Default Credentials — no extra secrets to manage
- Quality is sufficient for binary pass/fail metrics at 0.5–0.8 thresholds
- Agent non-determinism (different phrasing per run) occasionally causes marginal scores near thresholds

---

## ADR-005: Step Agent Detection from 4 Indirect Sources

**Decision:** Detect Step Agent executions by combining evidence from 4 sources rather than relying on a single definitive indicator.

**Context:** Step Agents are controlled agents executed by the Pega case workflow (not by user chat). They run as automated steps that pre-fill property values using Knowledge Base queries. **Critical finding: Step Agents do NOT have a unique `pyRole`** — only `USER` and `ASSISTANT` exist in `pyMessages`. There is no `STEP_AGENT`, `TOOL`, or `FUNCTION` role.

**The 4 Detection Sources:**

| # | Source | What It Detects | Reliability |
|---|---|---|---|
| 1 | **Content Patterns** | 9 regex patterns in assistant messages (🤖, "specialist agent", KB queries, Adobe audience) | Medium — depends on agent phrasing |
| 2 | **Field Prefills** | `defaultValue` entries in `pzAssignmentFieldsMetaData` | High — definitive proof a field was pre-filled |
| 3 | **Previous Assignment** | `pzPrevAssignmentFieldsMetaData` from completed prior steps | High — shows what earlier steps produced |
| 4 | **Stages API** | Completed steps with agent/auto/genai/KB keywords in `/cases/{key}/stages` | High — structural evidence from case lifecycle |

**Alternatives Considered:**
- **Single source (content only)** — Too fragile; step agents may execute silently
- **Stages API only** — Not all step agents register as named stages
- **AI Tracer** — Would be ideal but not available via public API

**Trade-offs:**
- Multi-source approach reduces false negatives at the cost of complexity
- Stages API returns 404 for fresh/initializing cases (handled gracefully)
- Field prefill detection is the most reliable signal
- Content patterns will need maintenance as agent prompts evolve

---

## ADR-006: Custom GlossarySourceMetric (Two-Phase Evaluation)

**Decision:** Build a custom DeepEval `BaseMetric` that combines hard-coded hedging phrase detection with an LLM judge evaluation.

**Context:** For glossary definition questions, we need to distinguish between "agent used the glossary tool and returned an authoritative answer" vs. "agent fell back to general LLM knowledge and hedged with multiple possibilities." Standard DeepEval metrics (AnswerRelevancy, Hallucination) don't capture this distinction.

**Two-Phase Design:**
1. **Hard-fail phase (deterministic):** Scan for 16 hedging phrases ("could refer to", "depending on context", "multiple meanings", etc.). Any match → score 0.0, immediate fail. This catches obvious general-knowledge fallbacks without burning a judge call.
2. **LLM judge phase:** If no hedging detected, Gemini scores 0–10 on glossary-sourced vs. general knowledge. Normalized to 0–1, threshold 0.8.

**Alternatives Considered:**
- **AnswerRelevancy alone** — Doesn't distinguish glossary source from general knowledge
- **Exact string matching** — Too brittle; glossary definitions may be paraphrased
- **Embedding similarity** — Requires reference embeddings; overkill for this use case

**Trade-offs:**
- Hard-fail phase is fast and deterministic — no LLM cost for obvious failures
- LLM judge handles nuanced cases where the answer sounds correct but isn't from the glossary
- Threshold of 0.8 is aggressive; agent non-determinism can cause flaky results near the boundary

---

## ADR-007: Module-Scoped Fixtures for Agent and Insight Clients

**Decision:** Use `@pytest.fixture(scope="module")` for both `surface_agent` and `pega_insight` fixtures.

**Context:** OAuth2 token acquisition and agent card discovery are expensive operations (~2-3s each). Running them per test would add significant overhead to a suite that already has 30s+ round-trip times per agent call.

**Trade-offs:**
- Shared authentication across all tests in the module — faster execution
- If token expires mid-suite, individual tests re-authenticate on 401
- No test isolation for auth state (acceptable for integration tests)

---

## ADR-008: Brief Delay Before Conversation Insight Queries

**Decision:** Insert a 2-second `time.sleep(2)` between the agent response and the `D_pxAutopilotConversation` query.

**Context:** After the A2A `message/send` response is returned, Pega's backend may still be finalizing the conversation state (writing messages, updating case processing data). Querying immediately can return incomplete data.

**Why 2 seconds:**
- Empirically determined: 1s was sometimes too early; 3s was unnecessarily slow
- The `PegaInsight.query_conversation()` method also has its own retry logic (3 retries, 2s delay) for cases where data isn't ready

**Alternatives Considered:**
- **Polling with exponential backoff** — More robust but adds complexity for a marginal improvement
- **No delay** — Resulted in occasional empty `pyMessages` arrays
- **Webhook/event-driven** — Not supported by the current DX API

---

## ADR-009: Single Test File Architecture

**Decision:** Keep all tests, fixtures, data classes, and utility functions in a single `test_surface_agents.py` file (~1241 lines).

**Context:** This is an evaluation suite, not a production application. The components are tightly coupled (agent → insight → metrics) and co-evolve together. Splitting into multiple files would add import complexity without meaningful separation of concerns at this stage.

**When to Refactor:**
- When the test count exceeds ~20 distinct test functions
- When multiple agent versions need parallel test suites
- When shared utilities (GeminiJudgeLLM, PegaInsight) are reused in other projects

---

## ADR-010: No Mocking — Live Integration Tests Only

**Decision:** All tests hit the live Pega Surface agent environment. There are no mocks, stubs, or recorded fixtures.

**Context:** The purpose of this suite is to evaluate the *actual* agent behavior including orchestration, tool routing, and response quality. Mocking would defeat this purpose.

**Trade-offs:**
- Tests require network access and valid credentials
- Agent non-determinism means tests can produce different scores across runs
- Test suite execution time is ~3-5 minutes (5 agent calls × 30s + insight queries + judge calls)
- Tests may fail due to environment issues (Pega downtime, expired creds)

**Mitigation:**
- Latency budget assertions (30s max) catch hung requests
- Retry logic in PegaInsight handles transient 401s
- Parametrized tests run the same assertion across multiple phrasings to reduce single-point-of-failure

---

## ADR-011: Passive Golden Session Capture (Not CLI-Driven)

**Decision:** Build a passive golden session recorder (`capture_golden_session.py`) that reconstructs the conversation from `D_pxAutopilotConversation` after the user runs the flow in the Pega UI, rather than driving the flow from a CLI recorder.

**Context:** The original CLI recorder (`record_golden_session.py`) required typing each turn into the terminal during a scripted multi-turn flow. This was impractical for the 8-step, 6-section campaign creation flow — the agent requires specific phrasing, document uploads, and assignment approvals that are awkward to drive via text input. Users naturally run the flow in the Pega UI's chat interface.

**How It Works:**
1. User completes the campaign flow in the Pega Surface UI (chat interface)
2. User copies the conversation ID (PXCONV-XXXXX) from DevTools or the UI
3. `capture_golden_session.py` queries `D_pxAutopilotConversation` and reconstructs turn pairs
4. Output JSON is identical in format to `record_golden_session.py`, compatible with `test_golden_session.py`

**Key Design Choices:**
- **Turn-pair reconstruction**: Pairs each USER message with the ASSISTANT response(s) that follow it, concatenating multi-part responses
- **Greeting handling**: Leading ASSISTANT messages (before any USER input) are recorded as turns with empty `input` — the replay test skips these
- **Latency estimation**: Estimated from `pxCreateDateTime` timestamps when available; falls back to 0ms
- **Auto-descriptions**: Heuristic descriptions generated from user input keywords and detected tools
- **Multi-conversation support**: Multiple PXCONV IDs can be combined into one golden session for multi-session flows

**Alternatives Considered:**
- **CLI recorder only** — Too difficult to use; required exact phrasing and couldn't handle file uploads
- **Browser extension** — Would capture raw HTTP but not the semantic turn structure
- **Pega event hooks** — No public webhook API for conversation events

**Trade-offs:**
- Passive capture only records what already happened — no real-time control
- Latency data is estimated (UI-to-UI timing, not precise A2A round-trip)
- Requires the conversation to still be queryable in `D_pxAutopilotConversation`

---

## ADR-012: conftest.py for Shared pytest Options

**Decision:** Move `pytest_addoption` (the `--golden` CLI flag) into `conftest.py` rather than keeping it in `test_golden_session.py`.

**Context:** pytest only picks up `pytest_addoption` hooks from `conftest.py` files or registered plugins — not from regular test files. Having `pytest_addoption` in `test_golden_session.py` caused `pytest: error: unrecognized arguments: --golden` when running via `deepeval test run` or with `pytest --golden`.

**Key Lesson:** The `deepeval test run` CLI wraps pytest internally and does not discover hooks from test modules. Only `conftest.py` placement works reliably across both `pytest` and `deepeval test run` invocations.

**Trade-offs:**
- One more file, but it's the standard pytest convention
- Any future shared options (e.g., `--agent-version`, `--env`) go in the same place


## ADR-013: Hybrid Transport — AgentX (default) + A2A via `--transport`

**Decision:** Default golden session replays to the AgentX Application v2 API (same path as the Pega UI), with A2A available via `--transport a2a`. An `auto` mode uses AgentX for file-upload turns and A2A for the rest.

**Context:** The original replay used A2A JSON-RPC exclusively. When the golden session included a file upload (e.g., `Zelle.pdf`), the A2A `message/send` sent only a text description (`"[I have attached a file: Zelle.pdf]"`) — no binary was attached to the case. The Pega agent correctly blocked workflow progression because no document existed on the case, causing 3 of 8 tests to fail (conversation completeness, tool invocations, hallucination). This was a framework limitation, not an agent bug.

**Why AgentX as Default:** The AgentX API (`/prweb/api/application/v2/ai-agents/`) is the same interface the Pega UI uses. It supports real file uploads via `/v2/attachments/upload` plus an `Attachments` payload on the message. Testing through this path exercises the full agent workflow — including document processing, audience assembly, and step agent delegation — exactly as a human marketer would experience it.

**Why Keep A2A:** A2A JSON-RPC is the external interop protocol. Other systems (partner agents, orchestrators, marketplaces) will call the agent via A2A, not AgentX. A2A-only testing already revealed the file-upload capability gap. Both protocols should be tested.

**Usage:**
```bash
# Default — AgentX (UI path), handles file uploads
deepeval test run test_golden_session.py -- --golden golden_sessions/golden_Zelle_...json

# Explicit A2A — tests external contract, no file support
deepeval test run test_golden_session.py -- --golden ... --transport a2a

# Auto — AgentX for file turns, A2A for everything else
deepeval test run test_golden_session.py -- --golden ... --transport auto
```

**File Attachment Convention:** Place test files in `golden_sessions/attachments/` (e.g., `Zelle.pdf`). The capture script auto-detects file turns from the `[I have attached a file: ...]` pattern and writes `file_attachment` metadata into the golden JSON with the expected path.

**Trade-offs:**
- AgentX is Pega-specific and may change between versions; A2A is a published standard
- AgentX requires `AGENT_NAME` env var in addition to the A2A agent card URL
- The `AgentXTransport` class is self-contained (uses `requests` directly — no SDK dependency)
- Both transports share the same DeepEval evaluation pipeline — only the message-send layer differs

**Validated 2026-02-19:** Both transports run end-to-end. A2A: 5/8 pass, AgentX: 4/8 pass. Failures are genuine agent findings, not transport issues. AgentX is slower on file-upload turns (~57s vs A2A within budget).

---

## ADR-014: Self-Contained AgentXTransport (No SDK Dependency)

**Decision:** Implement the AgentX transport as a self-contained `AgentXTransport` class using `requests` directly, rather than importing from the `AgentXTestSuites` project or requiring `pdstools`.

**Context:** The original plan was to reuse `AgentXTestClient` from `AgentXTestSuites/utils/agentx_client.py`, which imports `PegaOAuth` from `agent_sdk.utils`. However, the `agent_sdk` module depends on `pdstools` (Pega Data Scientist Tools), which has heavy dependencies and was not installed in the test venv. The import chain `agentx_client → agent_sdk → pdstools` caused `ModuleNotFoundError` at module load time — before any test code could run.

**Implementation:** The `AgentXTransport` class (~120 lines) in `test_golden_session.py` handles:
1. OAuth2 client_credentials token acquisition
2. `POST /api/application/v2/ai-agents/{agentID}/conversations` — create conversation
3. `POST /api/application/v2/attachments/upload` — file upload (for document turns)
4. `PATCH /api/application/v2/ai-agents/{agentID}/conversations/{conversationID}` — send message

**Alternatives Considered:**
- **Install pdstools** — Heavy dependency (~100+ packages) just for OAuth2 token flow
- **Monkey-patch imports** — Fragile; would break when agent_sdk internals change
- **Shared OAuth utility** — Over-engineering for a single OAuth2 grant type

**Trade-offs:**
- Self-contained: zero cross-project imports, works in any venv with `requests`
- Duplicates ~20 lines of OAuth2 logic that exists in agent_sdk
- File upload path exercises the same Pega API as the production UI

---

## ADR-015: Lazy Transport Fixture Initialization

**Decision:** Transport fixtures (`surface_agent` for A2A, `agentx_transport` for AgentX) return `None` when the selected transport doesn't need them, rather than eagerly constructing both clients.

**Context:** When running with `--transport agentx`, the A2A `SurfaceAgent` fixture would still try to fetch the agent card from the A2A endpoint. If that endpoint returned 404 (e.g., agent renamed), the entire test collection failed — even though A2A wasn't being used. Similarly, running `--transport a2a` would try to construct the AgentX client unnecessarily.

**Implementation:**
```python
@pytest.fixture(scope="module")
def surface_agent(transport_mode):
    if transport_mode in ("a2a", "auto"):
        return SurfaceAgent(AGENT_CARD_URL, CLIENT_ID, CLIENT_SECRET, TOKEN_URL)
    return None  # Not needed for agentx-only mode
```

**Trade-offs:**
- Prevents cross-transport failures during test collection
- Tests must handle `None` for the unused transport (enforced by the replay logic)
- Slightly less uniform fixture API, but the conditional is clear and well-documented

---

## ADR-017: Silent File-Upload Turns Require Explicit `file_attachment` Metadata in Golden JSON

**Status:** Known issue — fix pending next session.

**Context:** In the Pega Surface UI a user can upload a document without typing any text. The turn appears in `D_pxAutopilotConversation` with an empty USER message followed by an ASSISTANT response referencing the upload ("I've successfully uploaded the document..."). When `capture_golden_session.py` reconstructs the turn it records `input: ""` with no `file_attachment` key.

During replay, `_replay_session` skips turns where `input` is empty and `file_attachment` is absent. This means the PDF never reaches the case, the agent follows a no-document path, and subsequent responses trigger the wrong tool patterns.

**Symptom (2026-02-19):** `test_tool_invocations_match_golden` fails with `Missing tools {'Gradial_Agent'}. Got: {'SurfaceNewCampaignAutomation'}` on Turns 5, 6, 8, 9 — because Turns 2 and 11 (the silent uploads) were skipped.

**Fix (next session):**
1. In `capture_golden_session.py`, after pairing turns, inspect assistant content for `"uploaded the document"`, `"successfully uploaded"`, `"extracted.*from your document"`. If matched on an empty-input turn, inject `file_attachment` metadata pointing to `golden_sessions/attachments/<filename>`.
2. Update `_patch_golden_inputs.py` to support manually setting `file_attachment` on specific turns.
3. Re-capture or patch PXCONV-4081 golden to add `file_attachment` on Turns 2 and 11.

**Trade-offs:**
- Heuristic detection may miss silent uploads with unusual response phrasing
- Filename must be inferred from assistant response or manually specified — data view does not return it
- Alternative: support CLI annotation `--file-turn 2:Zelle.pdf 11:Marketing_Approval_Document.docx`

---

## ADR-018: `GetCaseStages` Regex Was Too Broad — Triggered on Normal Prose

**Decision:** Tightened the `GetCaseStages` detection regex in `_TOOL_DETECTION_PATTERNS` to require explicit stage-related phrases rather than any occurrence of the words "stage", "step", or "progress".

**Context:** The previous regex `(?:stage|step|progress|lifecycle|...)` matched the word "step" in ordinary sentences like "The **next step** requires 3rd party approval". This caused `GetCaseStages` to appear as a false positive in `expected_tools` for Turns 9 and 10 of the PXCONV-4081 golden — neither of which actually called the `GetCaseStages` API.

**Old pattern:**
```python
r"(?:stage|step|progress|lifecycle|where.*(?:am|are).*case)"
```

**New pattern (applied 2026-02-19):**
```python
r"(?:case\s+stages|(?:current|active)\s+stage|stages?\s+(?:API|endpoint|response)|GetCaseStages|lifecycle\s+(?:of|for)\s+(?:the\s+)?case|where.*(?:am|are).*(?:in\s+the\s+)?case)"
```

**Impact:** Golden re-captured after this fix shows `all_tools_used: ['Gradial_Agent']` only — no spurious `GetCaseStages`. Golden sessions captured before this fix may have stale false-positive entries.

---

## ADR-019: Hallucination Judge Context Must Cover Audience/Approval Stage Details

**Decision:** Enriched the `context` array in `test_no_hallucination_per_turn` with audience-approval-specific facts so Gemini does not flag Turn 8 (Adobe Audience results) as a hallucination.

**Context:** Turn 8's assistant response describes Adobe Audience segment sizes, match rates, and button choices like "Move forward with this audience". The original Judge context described the agent generically and didn't mention audience sizing data or approval button flows — Gemini scored it as hallucination.

**Lines added to context (2026-02-19):**
- "The agent presents Adobe Audience segment results with audience reach, size, and match rates during the Audience Approval stage."
- "Approval stages present the user with button choices like 'Move forward with this audience', 'Move forward with this content', or 'Move forward' to advance the workflow."
- "The agent may describe audience reach numbers, segment sizing, and channel details extracted from the Adobe integration."
- "The agent summarizes all campaign details (dates, channels, audience, content URLs) during final review and campaign approval stages."

**Result:** `test_no_hallucination_per_turn` PASSES (was FAIL). Rule of thumb: the Judge context must describe any data the agent is legitimately allowed to present — omitting a category causes the Judge to treat real output as fabricated.

---

## ADR-016: Pega OAuth Application Access Affects Agent Resolution

**Decision (operational):** Document that the Pega OAuth client's "Default Application Access" must be set to the application containing the agent rule (e.g., `Surface1Dev`), not a different application.

**Context:** During debugging, the AgentX API returned `500 Execution error` for every agent name (`V8`, `CLEAN`, `SURFACEFORAUTOMATION`) despite the agent working in the portal UI. After extensive diagnostics (testing multiple agent names, payloads, and endpoints), the root cause was that the OAuth client's user account had its default application access set to a different Pega application. The DX API resolves agent rules within the authenticated user's application context — if the agent rule lives in `Surface1Dev` but the user defaults to another app, the rule resolution fails server-side.

**Symptoms:**
- AgentX API: `500 Execution error` (agent found but cannot execute)
- A2A: Works independently (uses app-specific URL path `/prweb/app/surface1dev/...`)
- Portal UI: Works (user navigates to Surface1Dev explicitly)

**Resolution:** Set the OAuth user's default application access to `Surface1Dev` in Pega Admin Studio.

**Key Lesson:** When AgentX returns 500 but A2A and the portal work, check the OAuth client's application access configuration before debugging client code.

---

## ADR-020: Dual Vertex AI Authentication — Service Account Key or gcloud ADC

**Decision:** Support both Service Account JSON key files **and** `gcloud auth application-default login` for Vertex AI authentication, with SA key recommended for CI/CD.

**Context:** The original setup relied exclusively on user credentials via `gcloud`, which caused `RefreshError` and token expiration issues in CI/CD environments and long-running test suites. However, for local development, `gcloud ADC` is simpler — no key file to manage.

**Implementation:**
- If `GOOGLE_APPLICATION_CREDENTIALS` is set in `.env`, the SA key is used automatically.
- If unset, the Vertex AI client falls back to gcloud Application Default Credentials.
- The README documents both paths; the SA key approach is recommended for shared/CI environments.

**Trade-offs:**
- **SA Key:** Stable, non-interactive, works identically in local and CI/CD. Requires managing a sensitive key file (must be `.gitignore`d).
- **gcloud ADC:** Zero file management, quick for local dev. Tokens expire, require interactive `gcloud auth` refresh, problematic in CI/CD.

---

## ADR-021: Project Config System — `project_config.*.json`

**Decision:** Introduce a user-authored `project_config.*.json` file that describes the agent, workflow, and evaluation context for any Pega agent project — not just Surface.

**Date:** 2026-02-22

**Context:** The test suite was tightly coupled to the Surface marketing agent. Every LLM judge test had hardcoded strings: `chatbot_role` ("A Pega Surface marketing automation assistant for U+ Bank..."), `expected_outcome` (10 workflow stages), `hallucination_context` (18 Surface/Zelle-specific lines), and tool/step-agent detection patterns. This made the framework impossible to reuse for a different Pega agent without editing test code.

**Design:**
```
project_config.surface.json    ← user-authors this (one per project)
project_config.template.json   ← blank template with documentation
```

**Sections:**
| Section | Purpose |
|---|---|
| `connection` | Base URL, agent name, app path |
| `agent_identity` | Role description, domain, organization, off-topic guidance |
| `workflow` | Description + ordered stage list |
| `hallucination_context` | Ground-truth facts for the LLM hallucination judge |
| `tool_patterns` | Supplemental regex → tool-name mappings |
| `step_agent_patterns` | Supplemental regex → step-agent-name mappings |
| `silent_upload_patterns` | Patterns indicating silent file-upload turns |

**Resolution Chain (used by both capture script and tests):**
1. `--project-config` CLI flag
2. `PROJECT_CONFIG` environment variable
3. `project_config.json` in the test directory
4. First `project_config.*.json` glob match (skipping `template`)

**Alternatives Considered:**
- **YAML/TOML config** — JSON is native to the toolchain (golden sessions are JSON); no extra parser needed
- **Python config module** — Would require code changes per project; JSON is declarative
- **Inline in golden JSON** — Would grow the golden file and couple recording to config authoring

**Trade-offs:**
- Users must author the config once per project (agent identity + workflow + hallucination context)
- The auto-sense system (ADR-022) handles most mechanical patterns automatically
- Config is validated implicitly at test time — no schema enforcement yet (future improvement)

---

## ADR-022: Auto-Sensing During Golden Session Capture

**Decision:** During golden session capture, automatically derive gate patterns, timeouts, and a profile JSON from the recorded conversation data — reducing manual configuration to just the "fundamentals" in `project_config.*.json`.

**Date:** 2026-02-22

**Context:** The original capture script output only the raw golden JSON. All gate patterns (`wait_for_pattern`), timeout overrides, and evaluation profiles had to be manually added. This was error-prone and required deep knowledge of the conversation flow.

**What is Auto-Sensed:**

| Field | Source | Method |
|---|---|---|
| `wait_for_pattern` | Assistant response text | Extract markdown headings, choice prompts ("Yes / No"), and transition keywords; compose a regex that uniquely identifies the workflow stage |
| `gate_timeout` | Turn latency | If golden latency > 30s, add a timeout override (2× golden latency + 10s buffer) |
| Step agents | Assistant message content | Apply `_STEP_AGENT_CONTENT_PATTERNS` to each turn's assistant messages |
| Tool usage | Assistant message content | Apply `_TOOL_DETECTION_PATTERNS` to detect which tools were invoked |

**What is NOT Auto-Sensed (must be in project_config):**
- `agent_identity.role` — What the agent *should* do (can't be derived from what it *did*)
- `agent_identity.off_topic_guidance` — What the agent should refuse
- `workflow.stages` — The expected end-to-end workflow (not just what was observed)
- `hallucination_context` — Ground-truth facts about the agent's capabilities

**Output:** The capture script now produces two files:
```
golden_sessions/golden_<name>_<timestamp>.json    ← golden session with auto-sensed gates
golden_sessions/profile_<name>_<timestamp>.json   ← evaluation profile (auto-sensed + config merge)
```

**The golden JSON references the profile:** `"profile": "profile_<name>_<timestamp>.json"`

**Trade-offs:**
- Auto-sensed gate patterns may not always uniquely identify a stage — edge cases may need manual tuning
- The profile is a snapshot at capture time — if the agent changes behavior, re-capture is needed
- Auto-sensing depends on the quality of response text (markdown headings, button prompts)

---

## ADR-023: Gate-Based Response Synchronization for Replay

**Decision:** Implement `_gate_response()` — a pattern-based verification gate between each replayed turn — to prevent conversation desync during golden session replay.

**Date:** 2026-02-20

**Context:** The original replay used a fixed `wait=2s` delay between turns. The live agent takes 12–62 seconds for some turns (content assembly, audience assembly). After Turn 2 (file upload), the agent would acknowledge the upload but extraction hadn't finished. Turn 3 ("I approve.") would fire while the agent was still processing, causing the response to contain extraction results instead of the expected approval confirmation. Every subsequent turn was permanently out of sync.

**Gate Logic:**
1. After receiving the agent response, check it against the turn's `wait_for_pattern` regex
2. If match → proceed to next turn immediately
3. If no match → **passive wait** (golden `latency_ms` − elapsed time + 5s buffer)
4. If still no match after passive wait → send a nudge ("Please continue.") and retry
5. After 3 failed nudges → fail the turn with a desync error

**Why Pattern-Based, Not Time-Based:**
- Time-based gates require knowing exactly how long each turn takes — varies by 2–10× between runs
- Pattern-based gates proceed as soon as the right response arrives — no wasted waiting
- The pattern is auto-sensed during capture (ADR-022) from response headings and choice prompts

**Results:** With gate logic enabled, the test suite achieves 8/8 passing consistently. Typical total time is ~471s (~8 minutes) for 11 turns. No gate activations (passive wait or nudges) were needed in the successful runs — responses matched on first attempt.

**Trade-offs:**
- Adds complexity to the replay loop (30+ lines of gate logic)
- Nudge messages ("Please continue.") inject extra turns that don't exist in the golden — excluded from DeepEval evaluation
- If the agent fundamentally changes its response structure, gate patterns need updating (re-capture)

---

## ADR-024: Profile-Driven Test Evaluation

**Decision:** All 4 LLM-judge tests (`knowledge_retention`, `conversation_completeness`, `role_adherence`, `no_hallucination_per_turn`) now read their evaluation context from a `project_profile` pytest fixture instead of hardcoded strings.

**Date:** 2026-02-22

**Context:** Prior to this change, each test had Surface-specific strings baked into the test code:
- `test_knowledge_retention`: hardcoded `chatbot_role` mentioning "U+ Bank", "campaign creation", "document extraction"
- `test_conversation_completeness`: hardcoded `expected_outcome` listing 10 Surface workflow stages
- `test_role_adherence`: hardcoded `chatbot_role` with "marketing automation", "glossary", "taxonomy"
- `test_no_hallucination_per_turn`: 18-line `context` array with Surface/Zelle/Gradial specifics

**Implementation:**
```python
@pytest.fixture(scope="module")
def project_profile(request, golden_session):
    """Load evaluation profile — resolution: golden ref → CLI → env → glob."""
    cli_config = request.config.getoption("--project-config", default=None)
    return _load_profile(golden_session, cli_config)
```

Helper functions build the evaluation parameters from profile data:
- `_profile_role(profile)` → builds `chatbot_role` from `agent_identity.role` + `off_topic_guidance`
- `_profile_expected_outcome(profile)` → builds expected outcome from `workflow.stages`
- `_profile_hallucination_context(profile)` → returns `hallucination_context` lines with identity fallback

**Profile Resolution Order:**
1. `golden["profile"]` field → sibling file in `golden_sessions/`
2. `--project-config` CLI → builds profile from project config
3. `PROJECT_CONFIG` env → same
4. `project_config.json` or first `project_config.*.json` glob → same
5. Empty dict fallback → minimal defaults

**Trade-offs:**
- Tests are now fully project-agnostic — no Surface strings in test code
- Profile loading adds ~1 function call per test run (negligible overhead)
- The quality of LLM judge evaluations depends on how well the user authors `agent_identity.role` and `hallucination_context` — garbage in, garbage out
- Backward compatible: if no profile/config exists, fallback defaults produce reasonable (generic) evaluation

---

## ADR-025: Configurable Pattern Detection — Defaults + Profile Merge

**Decision:** Renamed hardcoded pattern constants to `_DEFAULT_*` and added `build_*_from_profile()` functions that merge profile patterns with defaults. Detection functions now accept optional pattern overrides.

**Date:** 2026-02-22

**Context:** `_TOOL_DETECTION_PATTERNS` (15 compiled regexes), `_TOOL_LABELS` (15 entries), and `_STEP_AGENT_CONTENT_PATTERNS` (11 compiled regexes) were all hardcoded for the Surface project. A different agent project would have entirely different tools and step agents.

**Design:**
```python
# Defaults preserved — backward compatible
_DEFAULT_TOOL_DETECTION_PATTERNS = [...]
_TOOL_DETECTION_PATTERNS = _DEFAULT_TOOL_DETECTION_PATTERNS  # alias

# Profile-aware builder — merges user patterns on top
def build_tool_patterns_from_profile(profile=None) -> List[tuple]:
    # Profile patterns first (higher priority), then defaults
    return extra + list(_DEFAULT_TOOL_DETECTION_PATTERNS)
```

**Detection functions updated to accept optional patterns:**
```python
def _detect_tools_from_messages(messages, patterns=None):
    if patterns is None:
        patterns = _DEFAULT_TOOL_DETECTION_PATTERNS
    ...
```

**Trade-offs:**
- Existing code referencing `_TOOL_DETECTION_PATTERNS` continues to work (backward-compatible alias)
- Profile patterns have higher priority (checked first) — project-specific patterns override defaults
- Default patterns are broad Pega patterns (case creation, stages, glossary) that apply to many agents
- Project-specific patterns (e.g., Surface's Gradial content agent) belong in the project config

---

## ADR-026: Report Generation Uses DeepEval In-Memory Results — Both Run Modes Produce Valid Reports

**Decision:** `conftest.py`'s `pytest_sessionfinish` hook reads DeepEval's `global_test_run_manager` in-memory state (populated by `assert_test()` calls) rather than `.deepeval/.latest_test_run.json`. This makes both `python3 -m pytest` and `deepeval test run` produce an accurate, current QA report.

**Date:** 2026-02-22

**Context:** An earlier version of the hook called `report_generator.py` with no arguments, which always read `.deepeval/.latest_test_run.json` — a file exclusively written by `deepeval test run`. Running `python3 -m pytest` triggered the hook but produced a report from stale data (in one case, a Feb 19 run against a different Pega instance, `genai-cdh-demo.pega.net`), causing false failure reports even when all 8 tests were passing against the current instance (`wellfb-surfce-dt1.pega.net`). This led to significant confusion: the CLI showed 8 PASSED while the report showed a critical Conversation Completeness failure (score 0.38) with conversation ID `PXCONV-17251` — which didn't even exist on the current instance.

**Root Cause of the Specific Incident:**
- `.deepeval/.latest_test_run.json` was 96KB and dated 2026-02-19 14:06 — from a run against `genai-cdh-demo.pega.net`
- Running `python3 -m pytest` produced 8 PASSED but did not overwrite the stale JSON
- The `pytest_sessionfinish` hook called `report_generator.py` → read stale JSON → generated wrong report

**Fix Design — Two hooks + priority strategy:**

```
pytest_runtest_logreport (fires after each test call)
  → appends {nodeid, outcome, duration, longrepr} to _pytest_test_results[]
  → covers ALL 8 tests regardless of whether they use assert_test() or plain assert

pytest_sessionfinish (fires at end of session)
  → serializes _pytest_test_results → temp file → passed to report_generator via --pytest-results
  → also reads global_test_run_manager in-memory state → temp file → passed via --data
  → falls back to .deepeval/.latest_test_run.json if in-memory data is empty

report_generator.py
  → receives --pytest-results (all 8 tests) + --data (LLM-judge detail for 3 semantic tests)
  → injects full suite table into Gemini prompt as primary pass/fail data
  → LLM-judge detail used for metric scores and failure analysis
```

**Why 3 tests use `assert_test()` and 5 don't:**

| Tests using `assert_test()` (LLM judge) | Tests using plain `assert` |
|---|---|
| `test_knowledge_retention` | `test_tool_invocations_match_golden` |
| `test_conversation_completeness` | `test_latency_regression` |
| `test_role_adherence` | `test_case_lifecycle` |
| | `test_step_agents_detected` |
| | `test_no_hallucination_per_turn` |

The plain-assert tests evaluate deterministic, structured data (tool lists, latency numbers, case IDs, step agent presence). They don't need a Gemini judge — they fail hard with a clear message. `assert_test()` is only needed when the evaluation requires semantic understanding.

**When to Use Each Run Mode:**

| Use Case | Command |
|---|---|
| Standard regression run (valid report, live console) | `python3 -m pytest test_golden_session.py -v -s` |
| Need results written to disk (CI artifact) | `deepeval test run test_golden_session.py -v` |
| Manual report from a prior `deepeval test run` | `python3 report_generator.py` |
| Manual report from a specific data file | `python3 report_generator.py --data path/to/run.json` |

**Alternatives Considered:**
- **Write `.latest_test_run.json` from `pytest_sessionfinish`** — would require reverse-engineering DeepEval's internal schema; brittle against version changes
- **Always require `deepeval test run`** — forces a less ergonomic command, loses `--golden` / `--transport` flag support (deepeval CLI doesn't forward custom pytest flags)
- **Remove the hook from `pytest`-only runs** — too aggressive; the hook is correct behaviour in both modes after this fix

**Trade-offs:**
- `global_test_run_manager` is a DeepEval internal — could change across versions (mitigated by try/except fallback)
- Temp file created and cleaned up in `pytest_sessionfinish` — negligible overhead
- `.latest_test_run.json` remains the fallback for standalone `report_generator.py` invocations
