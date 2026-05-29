# Agentic Regression Testing — Gaps & Expansion Plan

> **Context:** This document evaluates the current DeepEval-based testing framework for Pega Surface agents against the requirements of true agentic regression testing. It identifies what existing tools cover, what they don't, and what we'd need to build.

---

## 1. Two Testing Paradigms

Agentic applications require two fundamentally different testing approaches that no single tool handles well:

| Dimension | Quality Evaluation | Behavioral Regression |
|---|---|---|
| **Question** | "Was the response good?" | "Did the system do the right thing?" |
| **Asserts on** | Text output (relevancy, hallucination, role adherence) | System state (case fields, tool calls, assignments) |
| **Tolerance** | Fuzzy — LLM judge scores 0–1 | Deterministic — field X must equal Y |
| **Non-determinism** | Expected — scores vary ±10% across runs | Problematic — different tools ≠ wrong, but baseline breaks |
| **Our coverage** | DeepEval metrics (4 tests) | Custom pytest assertions (4 tests) |

---

## 2. Current Tool Landscape (Feb 2026)

### Quality Evaluation (mature)

| Tool | Approach | Covers |
|---|---|---|
| **DeepEval** (what we use) | LLM judge metrics, ConversationalTestCase | Output quality, hallucination, role, completeness, knowledge retention |
| **Ragas** | RAG-specific faithfulness/relevancy | Retrieval pipelines, not orchestration |
| **Braintrust** | Scoring functions + dataset management | Prompt-level eval, side-by-side comparison |
| **Promptfoo** | YAML-driven prompt testing | Fast iteration on prompt changes, red-teaming |

### Observability (mature, but passive)

| Tool | Approach | Covers |
|---|---|---|
| **LangSmith** | Trace-based evaluation | Full tool call traces, latency, token usage — LangChain-coupled |
| **Arize Phoenix** | OpenTelemetry spans | Span-level eval, tool call visibility — requires instrumentation |
| **Galileo** | Production monitoring | Drift detection, automated alerts — not assertion-based |

### Agentic-Specific (emerging, incomplete)

| Tool | Approach | Gap |
|---|---|---|
| **AgentEval (AutoGen)** | Multi-agent task completion | AutoGen-specific, not generalizable |
| **Patronus AI** | Red-teaming + regression | Hallucination/PII focus, not state-aware |
| **Guardrails AI** | Output contracts | Structural validation, not workflow regression |

### The Gap

**No tool provides state-aware behavioral regression for agentic workflows.** Every tool either evaluates text quality or observes execution traces — none asserts "after turn 5, the case field `AudienceTaxonomy` must equal `Financial Services`."

---

## 3. What We Have Today

### Quality Tests (DeepEval)

| Test | Metric | Validated |
|---|---|---|
| `test_knowledge_retention` | KnowledgeRetentionMetric | A2A ✅ AgentX ✅ |
| `test_conversation_completeness` | ConversationCompletenessMetric | Score 0.25 — real finding |
| `test_role_adherence` | RoleAdherenceMetric | A2A ✅ AgentX ✅ |
| `test_no_hallucination_per_turn` | HallucinationMetric | Turn 8-9 flagged — real finding |

### Behavioral Tests (custom)

| Test | Assertion Type | Validated |
|---|---|---|
| `test_tool_invocations_match_golden` | Set comparison: expected vs actual tools per turn | Fails on agent version change (rigid baseline) |
| `test_latency_regression` | Per-turn: actual ≤ 2× golden baseline (10s min) | A2A ✅, AgentX fails on file turns (57s) |
| `test_case_lifecycle` | Case key created + consistent across turns | A2A ✅ AgentX ✅ |
| `test_step_agents_detected` | Step agents found when golden had them | A2A ✅ AgentX ✅ |

### Infrastructure

- Dual transport: AgentX (UI-equivalent) + A2A (external interop)
- Self-contained `AgentXTransport` (no SDK dependencies)
- `PegaInsight`: conversation query, stages, field metadata, step agent detection
- Golden session capture from `D_pxAutopilotConversation`
- Vertex AI Gemini 2.0 Flash as LLM judge

---

## 4. What's Missing

### 4.1 State Assertions on Case Fields

**Problem:** We verify a case was *created*, but not that it was created *correctly*. The agent could create a Zelle campaign case and pass all current tests while setting `AudienceTaxonomy = "Healthcare"`.

**Solution:** After replay, query `GET /cases/{key}` via DX API. Assert field values against golden expectations.

```
Golden: turn 5 → expected_state: {CampaignName: "Zelle", AudienceTaxonomy: "Financial Services"}
Actual: query case properties → assert match
```

**Effort:** 2-3 hours. `PegaInsight` already has OAuth and DX API v2 access.

### 4.2 Multi-Path Tool Tolerance

**Problem:** `test_tool_invocations` uses rigid set comparison. When the agent takes a valid-but-different tool path (e.g., `GetMyAssignmentsAll` instead of `taxonomy_agent`), the test fails even though the outcome is correct.

**Solution:** Golden JSON defines multiple acceptable tool sets per turn. Test passes if actual matches ANY.

```json
{
  "turn": 5,
  "acceptable_tool_sets": [
    ["taxonomy_agent"],
    ["GetMyAssignmentsAll", "GetCaseStages"]
  ]
}
```

**Effort:** 4-6 hours. Schema change + matching logic + re-capture golden with alternatives.

### 4.3 Tool Call Ordering

**Problem:** Current detection is unordered — `{A, B}` == `{B, A}`. Can't catch "agent called GetCaseStages before creating the case."

**Solution:** Parse tool invocation timestamps from `pyMessages[].pxCreateDateTime`. Assert ordered sequences.

**Effort:** 1-2 hours. We already detect tools per turn; just need to preserve and compare order.

### 4.4 Side Effect Verification

**Problem:** The agent creates cases, triggers step agents, posts to Pulse, completes assignments — we only check conversation content, not whether these side effects actually occurred.

**Solution:** Post-replay verification queries:
- `GET /cases/{key}` — case exists, correct type, correct status
- `GET /cases/{key}/assignments` — assignments completed
- Query Pulse feed for expected posts
- Verify attachment presence on case

**Effort:** 4-6 hours. New `PegaInsight` methods for each endpoint.

### 4.5 Cross-Transport Diff Report

**Problem:** We run A2A and AgentX separately. Comparing results requires manual inspection.

**Solution:** Single command runs both transports, produces a structured comparison:

```
Turn 2: A2A 5.7s ✅ | AgentX 57.0s ❌ (10× slower)
Turn 5: A2A tools {GetCaseStages} | AgentX tools {pxPerformAssignment, GetCaseStages}
```

**Effort:** 3-4 hours. Orchestrator that runs both and diffs `ReplayResult` objects.

### 4.6 Regression Trend Tracking

**Problem:** Each test run is independent. No historical context — "did this score regress from last week?"

**Solution:** Persist results to JSON/SQLite per run. Compare against rolling p50/p95. Alert on trend breaks.

**Effort:** 4-6 hours. Storage layer + basic stats + optional HTML dashboard.

---

## 5. Expansion Priority

Ordered by impact-to-effort ratio:

| Priority | Capability | Effort | Impact | Rationale |
|---|---|---|---|---|
| **P0** | State assertions (case fields) | 2-3 hrs | High | Catches the real regression: "workflow completed but values are wrong" |
| **P0** | Multi-path tool tolerance | 4-6 hrs | High | Eliminates false positives from valid non-determinism |
| **P1** | Tool call ordering | 1-2 hrs | Medium | Catches sequencing bugs (tool called before prerequisite) |
| **P1** | Cross-transport diff | 3-4 hrs | Medium | One command, full protocol comparison |
| **P2** | Side effect verification | 4-6 hrs | Medium | Validates system mutations, not just conversation |
| **P2** | Regression trend tracking | 4-6 hrs | Medium | Historical context for score interpretation |
| **P3** | Auto-capture from UI | 1-2 days | Low | Nice-to-have; passive capture works well enough |
| **P3** | OTel trace integration | Blocked | High | Requires Pega platform instrumentation — not in our control |

---

## 6. Architectural Constraint

All behavioral assertions depend on **what Pega exposes via DX API v2**. The agent runtime is a black box — we cannot instrument it, add trace spans, or intercept tool calls. Everything we know comes from:

1. `D_pxAutopilotConversation` — message content, message timestamps
2. `/cases/{key}/stages` — case lifecycle steps
3. `/cases/{key}` — case property values (not yet used)
4. `pzAssignmentFieldsMetaData` — pre-filled field values from step agents
5. A2A/AgentX response payloads — `contextId`, `messageId`, latency

This is both a limitation and a design principle: **we test the agent the same way a real consumer would — through its public APIs.** The tests are portable across Pega versions and environments. Any assertion that requires internal instrumentation is out of scope unless Pega exposes it.
