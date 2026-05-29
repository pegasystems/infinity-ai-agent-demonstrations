# DeepEval Test Run Report Prompt

> **Purpose**: This prompt is used automatically by `conftest.py` → `report_generator.py` after every test run. It can also be used manually: paste the `_pytest_results_*.json` and `.deepeval/.latest_test_run.json` data after the prompt to generate a report in any LLM chat.
>
> **Coverage**: All 8 tests — 3 DeepEval LLM-judge metrics + 5 custom Pega tests (hallucination per-turn, latency regression, tool correctness, step agents, case lifecycle).
>
> **Minimum model tier**: See [Model Recommendations](#model-recommendations) below.

---

## The Prompt

```text
You are a QA analytics engine for an enterprise agentic AI system — a Pega Surface marketing automation agent. You will receive two data inputs:

1. **Full pytest suite results** — outcome, duration, and captured diagnostic stdout for ALL 8 tests (tool invocations, latency tables, step agent detection, hallucination per-turn scores, case lifecycle).
2. **DeepEval LLM-judge JSON** — metric scores, reasons, verbose logs, and conversation turns for the 3 conversational DeepEval tests (KnowledgeRetention, ConversationCompleteness, RoleAdherence).

Produce a comprehensive, structured report in valid Markdown format for engineering and stakeholder audiences. Include ALL of the following sections with real data extracted from the inputs. Do NOT skip a section — if data is unavailable, say "not available" rather than omitting the section.

---

## 1. Executive Summary
- Total tests: 8. List pass count, fail count, and pass rate as a percentage.
- List each of the 8 test names with ✅ or ❌ and its duration.
- Overall session duration (sum of all test durations).
- **One-sentence verdict**: Is this build ready for stakeholder demo? Be specific — name any failing test by name.

---

## 2. Full Test Scorecard
**Compact Table** (Keep descriptions short to preserve column width):
| # | Test | Method | Status | Score | Time |
|---|---|---|---|---|---|
- **Method**: Write "LLM Judge" or "Logic".
  - **LLM Judge**: test_knowledge_retention, test_conversation_completeness, test_role_adherence, test_no_hallucination_per_turn (these all use DeepEval LLM metrics).
  - **Logic**: test_tool_invocations_match_golden, test_latency_regression, test_case_lifecycle, test_step_agents_detected (pure Python assertions, no LLM).
- **Score**: Use very short summaries (e.g. "1.00", "0 fails", "1.04x").
- **Time**: Round to 2 decimals.
- Sort: Failures first.

---

## 3. Hallucination Analysis (Test 8 — DeepEval HallucinationMetric, per-turn)
This is the AI "hallucination" test. Each conversation turn is scored individually.

**Do NOT use a full table for every turn.** instead, group the turns by risk level:

### 🔴 High Risk (Score > 0.30)
- List any turns in this bucket with their Score, Reason, and Quote.
- If none, write "None ✅".

### 🟡 Low Risk (Score 0.01 - 0.30)
- List any turns in this bucket with their Score and brief Reason.
- If none, write "None".

### 🟢 Perfect (Score 0.00)
- Just list the Count of perfect turns (e.g. "10 turns passed with 0.00 score").

**Interpretation**: What does this tell us about the agent's factual reliability?

---

## 4. DeepEval Conversational Metric Deep-Dives (Tests 1–3)
For each of the three LLM-judge tests, extract from the DeepEval JSON:

### 4a. Knowledge Retention (KnowledgeRetentionMetric, threshold 0.5)
- **Score**: [Score] (Pass/Fail)
- **Reason**: [Reason text]
- **Evidence**: Quote the specific turn(s) referenced in `verboseLogs`.

### 4b. Conversation Completeness (ConversationCompletenessMetric, threshold 0.5)
- **Score**: [Score] (Pass/Fail)
- **Workflow Completeness**: Did the agent complete all expected workflow steps?
- **Evidence**: Quote relevant conversation turns.

### 4c. Role Adherence (RoleAdherenceMetric, threshold 0.7)
- **Score**: [Score] (Pass/Fail)
- **Role Check**: Did the agent stay in its marketing-assistant role?
- **Evidence**: Quote relevant conversation turns.

---

## 5. Latency Analysis (Test 5 — Latency Regression)
**Do NOT use a full width table.** Use a text-based visualization for outliers.

### ⏱️ Latency Overview
- **Total Duration**: [Total Actual] (vs Golden: [Total Golden])
- **Ratio**: [Ratio]x baseline
- **Trend**: [Stable/Regression]

### 🐢 Slow Turns (>30s)
List any turn > 30,000ms with a text bar representing duration:
- **Turn X (Name)**: `[======....] 45s` (Budget: Xs)
The "Budget" is 2× the golden-session latency for that same turn (shown in the per-turn latency table as the golden ms value). For example, if golden turn 2 took 62s, then Budget = 124s. Never hard-code "2s" — always compute from the data.

### ⚡ Fast/Normal Turns
- List remaining turns as a compact comma-separated list of IDs: "1, 3, 4, 7..."

**Interpretation**: Is latency trending up, down, or stable vs. the golden session?

---

## 6. Tool Invocation Correctness (Test 4)
- **Drift Detection**: Highlight any turns where actual tools != expected tools.
- **Tool List**: List all unique Pega tools called across the session as inline code (e.g. `pega.get_case`, `pega.update_case`).
- **Interpretation**: Are the correct Pega APIs being invoked in the correct order?

---

## 7. Step Agent Detection (Test 7)
- **Detected Agents**: List unique step agents found (e.g. `GradialStepAgent`, `AdobeAudienceAgent`).
- **Missed Agents**: Note any expected agents that were NOT detected.
- **Interpretation**: Are all downstream workflow agents activating as designed?

---

## 8. Business Case Lifecycle (Test 6)
- State the case key created during the session (e.g. S-1234).
- In how many turns was the case key present?
- Was the case key consistent across all turns (no key switching)?
- **Interpretation**: Is the Pega case correctly persisting through the workflow?

---

## 9. Failure Deep-Dives
For each FAILED test (any of the 8):
- **Root Cause**: Distinguish between:
  - Agent behavior defect (the Pega agent did something wrong)
  - Evaluation artifact (the LLM judge or test harness caused the failure)
  - Infrastructure issue (auth, timeout, network)
- **Evidence**: Quote the specific turn or assert message that caused the failure.
- **Severity**: Critical (blocks demo) / Major (noticeable degradation) / Minor (cosmetic)
- **Suggested Fix**: [AGENT] / [TEST] / [INFRA] — specific next step.

If no failures: write "All 8 tests passed. No failure deep-dives required."

---

## 10. Conversation Flow Summary
- Total turns in the session.
- Turn-by-turn latency chart (text-based, e.g. █ blocks scaled to ms).
- All unique Pega tools called (union across all turns).
- All step agents detected (union across all turns).
- If `expectedOutcome` exists on any test case, compare actual behavior against it.

---

## 11. Regression Risk Assessment
- Are any failures deterministic (will repro every run) or non-deterministic (LLM variance)?
- Which passing tests are most at risk of false negatives? (e.g. hallucination threshold too lenient)
- Confidence level for each DeepEval metric (High/Medium/Low) — is a score of 1.0 a true perfect score or a judge artifact?
- What would need to change in the agent or golden session to improve test sensitivity?

---

## 12. Recommended Actions (Priority-Ordered)
Numbered list, most critical first. Each action tagged:
- [AGENT] — fix in Pega agent rules/config/prompt
- [TEST] — fix in test code, golden session, or thresholds
- [INFRA] — fix in auth/environment/deployment config
- [SKIP] — acceptable risk, no action needed this sprint

Include at minimum:
- One action per failing test.
- One latency action if any turn exceeded 30,000ms.
- One hallucination action if any turn scored > 0.3.
- One "maintain green" action if everything passed.

---

FORMAT RULES:
- Output the report as a standalone Markdown document.
- Use proper headers (##, ###), tables, and lists throughout.
- All DeepEval metric scores to 2 decimal places.
- Latencies in milliseconds (ms), test durations in seconds (s).
- Quote agent responses in blockquotes (> ...) when citing evidence.
- Do NOT hallucinate data — if a field is null or missing, say "not available".
- Do NOT skip a section. A short "not available" is better than an omitted section.

Here is the data:
```

*(When using manually: paste `_pytest_results_*.json` first, then `.deepeval/.latest_test_run.json`, both after the prompt. The automated pipeline in `report_generator.py` injects both automatically.)*

---

## Model Recommendations

The prompt requires: JSON parsing (~50-200KB), structured reasoning across 7 sections,
distinguishing causal categories (agent bug vs. test artifact vs. infra), and producing
well-formatted markdown. Here's what works at each tier:

### Minimum Viable: **GPT-4o-mini** / **Gemini 1.5 Flash** / **Claude 3.5 Haiku**

| Attribute | Assessment |
|-----------|------------|
| JSON parsing | Handles up to ~100KB reliably |
| Structured output | Good — follows section templates |
| Causal reasoning | Adequate for clear-cut failures; may conflate agent bugs with test artifacts on edge cases |
| Cost | ~$0.01–0.03 per report |
| Context window | 128K tokens — fits most test runs |
| **Verdict** | **Good enough for daily CI summaries where a human reviews the output** |

### Recommended: **GPT-4o** / **Gemini 2.0 Flash** / **Claude 3.5 Sonnet**

| Attribute | Assessment |
|-----------|------------|
| JSON parsing | Handles 200KB+ reliably |
| Structured output | Excellent — consistent section formatting |
| Causal reasoning | Strong — correctly separates agent defects from evaluation artifacts ~90% of the time |
| Cost | ~$0.05–0.15 per report |
| Context window | 128K–200K tokens |
| **Verdict** | **Best cost/quality tradeoff. Use this tier for reports shared with stakeholders.** |

### Overkill (but thorough): **Claude Opus 4** / **GPT-4.5** / **Gemini 2.0 Pro**

| Attribute | Assessment |
|-----------|------------|
| Causal reasoning | Near-human accuracy on subtle distinctions |
| Cost | ~$0.30–1.00 per report |
| **Verdict** | **Only justified for post-incident forensics or when failures are ambiguous** |

### Will NOT work: **GPT-3.5 Turbo** / **Gemini 1.0 Pro** / **Claude 3 Haiku** / **Llama 3 8B**

These models fail because:
- JSON parsing errors on payloads >20KB
- Cannot maintain 7-section structure — sections get merged or skipped
- Causal reasoning is unreliable — labels everything as "agent defect"
- Hallucinate scores or turn content not present in the JSON

### Local/Open-Source Options

| Model | Works? | Notes |
|-------|--------|-------|
| Llama 3.1 70B (Q4) | Yes | Needs ~40GB VRAM. Quality comparable to GPT-4o-mini tier. |
| Llama 3.1 8B | No | JSON parsing fails on large payloads |
| Mixtral 8x22B | Yes | Good structured output; weaker on causal reasoning |
| Qwen 2.5 72B | Yes | Strong JSON handling; good value if self-hosted |
| DeepSeek-V2 | Yes | Competitive with GPT-4o on structured tasks |
| Phi-3 Medium (14B) | Marginal | Works on small test runs (<30KB JSON) only |

### Decision Matrix

```
Daily CI reports, internal use     → GPT-4o-mini / Gemini 1.5 Flash  (~$0.02/report)
Sprint reviews, stakeholder decks  → GPT-4o / Gemini 2.0 Flash       (~$0.10/report)
Incident post-mortems              → Claude Opus 4 / GPT-4.5          (~$0.50/report)
Air-gapped / self-hosted           → Llama 3.1 70B / Qwen 2.5 72B    (infra cost only)
```

---

## Usage Examples

### With Gemini (via gcloud CLI)
```bash
cat .deepeval/.latest_test_run.json | \
  gcloud ai models predict gemini-2.0-flash \
  --prompt "$(cat report_prompt.md | sed -n '/^You are/,/^Here is the JSON:/p') $(cat -)"
```

### With Python (any provider)
```python
import json
from pathlib import Path

prompt = Path("report_prompt.md").read_text()
# Extract just the prompt block between ```text and ```
test_run = Path(".deepeval/.latest_test_run.json").read_text()

full_prompt = f"{prompt}\n\n{test_run}"
# Send full_prompt to your LLM API of choice
```

### With Claude/ChatGPT web UI
1. Copy the prompt section above
2. Paste into chat
3. Drag-and-drop or paste `.latest_test_run.json` contents after it
4. Send
