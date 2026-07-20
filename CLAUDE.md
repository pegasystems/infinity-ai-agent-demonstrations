# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

**DeepEval_Pega** is an end-to-end regression testing pipeline for Pega AI agents. It replays recorded multi-turn "golden sessions" against live Pega agents, evaluates 13 quality dimensions using DeepEval with a pluggable LLM judge (Google Gemini, OpenAI, AWS Bedrock, or GitHub Copilot), stores results in SQLite, and exposes them via an MCP server and a Reflex web UI.

## Key Commands

### Environment Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in credentials
python create_db.py   # initialize qa_results.db
python verify_connection.py  # smoke test OAuth2 + Pega connectivity
```

### Running Tests
```bash
# Run full 13-test suite (uses defaults from .env)
pytest test_golden_session.py -v -s

# Target a specific golden file
pytest test_golden_session.py -v -s --golden golden_sessions/<file>.json

# Select specific metrics
pytest test_golden_session.py -v -s --metrics answer_relevancy,hallucination,pega_tool_correctness

# Override transport (agentx | a2a | auto)
pytest test_golden_session.py -v -s --transport a2a

# Step agent evaluation (provide case ID for agent to operate on)
pytest test_golden_session.py -v -s --case-id "UPLUS-FS-WORK P-168004"
```

### Recording Golden Sessions
```bash
# Capture from a Pega conversation ID (get from Pega UI after a live run)
python capture_golden_session.py PXCONV-12345 --name "Flow Name"

# Annotate which workflow the session exercises (must match a workflows[].id in config)
python capture_golden_session.py PXCONV-12345 --name "Complaint Flow" --workflow-id complaint_resolution

# Capture from structured agent output JSON (better tool detection than regex)
python capture_golden_session.py --from-json agent_output.json --workflow-id complaint_resolution

# Capture a step agent session (records case ID as metadata)
python capture_golden_session.py PXCONV-12345 --name "Step Agent Flow" --case-id "UPLUS-FS-WORK P-168004"

# List recent conversations available to capture
python capture_golden_session.py --list-recent
```

### Utilities
```bash
# Annotate golden sessions with detection_mode ("structured" vs "regex")
python _migrate_golden_sessions.py [--dir golden_sessions] [--dry-run]

# Re-run DB ETL from a saved pytest results JSON (iterate on reports without re-running tests)
python db_etl.py --pytest-results _pytest_results_from_log.json
python db_etl.py --init-db   # initialize schema only
python db_etl.py --pytest-results results.json --run-id 20260303_120000 --dry-run
```

### Web UI
```bash
reflex run
# Frontend: http://localhost:3000  Backend: http://localhost:8000
```

### REST API
```bash
python run_api.py                         # http://localhost:8100 (development, auto-reload)
uvicorn api.app:create_app --factory --host 0.0.0.0 --port 8100  # production

# Configure clients: copy api_clients.example.json → api_clients.json and fill in secrets
# Or set env vars: API_CLIENT_ID, API_CLIENT_SECRET, API_JWT_SECRET
```

| Method | Path | Auth Scope | Purpose |
|---|---|---|---|
| `POST` | `/oauth/token` | — | Obtain JWT (client_credentials grant) |
| `GET` | `/projects` | `projects:read` | List project configurations |
| `POST` | `/projects` | `projects:write` | Create a project configuration |
| `GET` | `/datasets` | `datasets:read` | List all golden datasets |
| `GET` | `/datasets/by-project/{name}` | `datasets:read` | List golden datasets for a project |
| `GET` | `/datasets/{filename}` | `datasets:read` | Get full golden dataset JSON |
| `POST` | `/datasets` | `datasets:write` | Create golden dataset from agent output |
| `PATCH` | `/datasets/{filename}` | `datasets:write` | Rename a golden dataset |
| `PUT` | `/datasets/{filename}` | `datasets:write` | Replace golden dataset content |
| `DELETE` | `/datasets/{filename}` | `datasets:write` | Delete golden dataset and profile |
| `POST` | `/evaluations` | `evaluations:write` | Start an evaluation (returns 202 + run_id) |
| `GET` | `/evaluations/{run_id}` | `evaluations:read` | Poll evaluation status/results |
| `GET` | `/evaluations` | `evaluations:read` | List recent evaluations |
| `GET` | `/metrics` | — | List available metrics |

OpenAPI docs: `http://localhost:8100/docs`

### MCP Server (exposes test results to LLM agents)
```bash
python qa_results_mcp_server.py          # stdio (Claude Desktop)
PORT=8090 python qa_results_mcp_server.py --transport http
fastmcp dev qa_results_mcp_server.py     # inspector UI
```

## Required Environment Variables (`.env`)

| Variable | Purpose |
|---|---|
| `AGENTX_BASE_URL` | Pega instance base URL |
| `AGENT_NAME` | Pega agent rule name (`@CLASSNAME!NAME` format) |
| `PEGA_CLIENT_ID` / `PEGA_CLIENT_SECRET` | OAuth2 credentials |
| `LLM_PROVIDER` | LLM judge provider: `gemini` (default), `bedrock`, `openai`, `copilot`, or `anthropic` |
| `GEMINI_API_KEY` | Google Gemini API key (when `LLM_PROVIDER=gemini`) |
| `GEMINI_MODEL_ID` | Gemini model name (default `gemini-2.5-flash`; when `LLM_PROVIDER=gemini`) |
| `OPENAI_AUTH_METHOD` | OpenAI auth: `api_key` (default) or `oauth` (Sign in with ChatGPT subscription) |
| `OPENAI_API_KEY` | OpenAI API key (when `LLM_PROVIDER=openai` and `OPENAI_AUTH_METHOD=api_key`) |
| `OPENAI_MODEL_ID` | OpenAI model name (default `gpt-4o`; when `LLM_PROVIDER=openai`) |
| `COPILOT_AUTH_METHOD` | Copilot auth: `api_key` (default, GitHub PAT) or `oauth` (Copilot subscription) |
| `GITHUB_COPILOT_TOKEN` | GitHub PAT with Copilot access (when `COPILOT_AUTH_METHOD=api_key`) |
| `GITHUB_COPILOT_MODEL_ID` | GitHub Copilot model (default `openai/gpt-4o`; uses `publisher/model` format) |
| `ANTHROPIC_AUTH_METHOD` | Anthropic auth: `api_key` (default) or `oauth` (Sign in with Claude subscription) |
| `ANTHROPIC_API_KEY` | Anthropic API key (when `LLM_PROVIDER=anthropic` and `ANTHROPIC_AUTH_METHOD=api_key`) |
| `ANTHROPIC_MODEL_ID` | Anthropic model name (default `claude-sonnet-4-5`; when `LLM_PROVIDER=anthropic`) |
| `AWS_AUTH_METHOD` | Bedrock auth: `access_keys` (default) or `sso_profile` |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | AWS credentials (when `AWS_AUTH_METHOD=access_keys`) |
| `AWS_PROFILE` | Named AWS CLI/SSO profile (when `AWS_AUTH_METHOD=sso_profile`) |
| `AWS_REGION` | AWS region for Bedrock (default `us-east-1`) |
| `AWS_BEDROCK_MODEL_ID` | Bedrock model ID (default `anthropic.claude-3-5-sonnet-20241022-v2:0`) |
| `PROJECT_CONFIG` | Project config filename (auto-discovers if omitted) |

For SSO profile auth, run `aws sso login --profile <name>` before starting the app or running tests.

Optional: `DEFAULT_TIMEOUT`, `MAX_RETRIES`, `VERIFY_SSL`, `TRANSPORT`, `GOLDEN_FILE`, `EVAL_METRICS`, `PEGA_CASE_ID`.

## Architecture

```
Reflex Web UI (DeepEval_Pega/DeepEval_Pega.py)
    ├─► DX API v2 (OAuth2) ──► D_pxAutopilotConversations (list) + D_pxAutopilotConversation (details)
    └─► pytest subprocess (test_golden_session.py)
            ├─► AgentXTransport (AgentX v2 or A2A JSON-RPC) ──► Pega Agent (OAuth2)
            ├─► PegaInsight (DX API v2) ──► D_pxAutopilotConversation + /cases/{key}/stages
            ├─► create_judge_llm() ──► GeminiJudgeLLM         ──► Google Gemini
            │                     ├─► OpenAIJudgeLLM        ──► OpenAI
            │                     ├─► BedrockJudgeLLM       ──► AWS Bedrock (Claude/Titan/Llama)
            │                     ├─► GitHubCopilotJudgeLLM ──► GitHub Models
            │                     └─► AnthropicJudgeLLM     ──► Anthropic API (Claude)
            └─► conftest.py hooks
                    ├─► report_generator.py ──► QA_Report_<timestamp>.md
                    └─► db_etl.py ──► qa_results.db (SQLite)
                                          └─► qa_results_mcp_server.py ──► LLM Agents
```

**Dual transport:** AgentX v2 API (same path as Pega UI) or A2A JSON-RPC — selected per `TRANSPORT` env or `--transport` flag.

**Tool & step-agent detection:** Hybrid approach — regex patterns (16 rules ported from SurfacePOC UI) + DX API v2 structured data + 4-source step agent detection (content patterns, field prefills, previous assignments, stages API).

**Business case detection:** Three-tier approach for `test_business_case_adherence`:
1. Response text regex patterns (5 patterns covering standard agent phrasings)
2. DX API `case_stages` data (stage name = case type)
3. DX API `pzCaseKey` change detection (new key appearing between turns confirms case creation)

**DX API v2 data views used:**
- `D_pxAutopilotConversations` — list all conversation instances (POST, returns paginated `data[]` with `pyID`, `pyStatusWork`, `pxCreateDateTime`, `pxCreateOperator`, `pyLabel`)
- `D_pxAutopilotConversation` — single conversation details (GET with `dataViewParameters={"InteractionID":"PXCONV-XXXXX"}`, returns `pyMessages[]` with turn history)

## The 13 Golden Session Tests

| Test | Method | What It Checks |
|---|---|---|
| `test_knowledge_retention` | LLM Judge | Context from early turns persists later |
| `test_conversation_completeness` | LLM Judge | All workflow stages completed; skipped if no `workflow_id` annotated |
| `test_role_adherence` | LLM Judge | Agent stays in role; respects off-topic guidance |
| `test_no_hallucination_per_turn` | LLM Judge | Each response grounded in conversation context |
| `test_contextual_precision` | LLM Judge | Retrieved context contains only relevant information |
| `test_contextual_recall` | LLM Judge | Retrieved context covers all necessary information |
| `test_toxicity` | LLM Judge | Agent responses are free of toxic or harmful content |
| `test_bias` | LLM Judge | Agent responses are free of discriminatory bias |
| `test_tool_invocations_match_golden` | Logic | Per-turn tool calls match baseline (regex + structured detection) |
| `test_latency_regression` | Logic | Each turn ≤ 2× golden baseline latency |
| `test_case_lifecycle` | Logic | Business case created and consistent across turns |
| `test_business_case_adherence` | Logic | Correct business case type created per turn (3-tier detection) |
| `test_step_agents_detected` | Logic | Step agents found when golden recorded them |

## Key Files

| File | Purpose |
|---|---|
| `test_golden_session.py` | 13-test suite + `AgentXTransport` class |
| `test_surface_agents.py` | Shared utilities: `_JudgeLLMBase`, `GeminiJudgeLLM`, `BedrockJudgeLLM`, `OpenAIJudgeLLM`, `GitHubCopilotJudgeLLM`, `AnthropicJudgeLLM`, `create_judge_llm()`, `PegaInsight`; dataclasses `AgentResponse`, `StepAgentExecution`, `CanonicalToolEvent`, `ConversationInsight`; tool detection functions |
| `llm_oauth.py` | OAuth/subscription sign-in for the LLM judge (OpenAI ChatGPT, GitHub Copilot device-code, Anthropic Claude): PKCE/device-code flows, token storage/refresh in the `__oauth__` vault namespace, and `get_*_token`/`*_generate` helpers consumed by the judge classes |
| `capture_golden_session.py` | Record golden sessions; `_normalize_workflows()` coerces legacy configs; `--workflow-id` flag annotates sessions |
| `conftest.py` | Pytest hooks: `pytest_addoption` (`--golden`, `--transport`, `--metrics`), `pytest_runtest_logreport` (accumulate results), `pytest_sessionfinish` (generate report + ETL to SQLite) |
| `report_generator.py` | Generate QA markdown reports from accumulated test results |
| `db_etl.py` | ETL: parse pytest results → SQLite; called by conftest.py post-session; also has standalone CLI for replaying from saved JSON |
| `insert_results.py` | Legacy SQLite insert path (superseded by db_etl.py for new runs) |
| `qa_results_mcp_server.py` | MCP server with 8 tools for querying test results |
| `create_db.py` | Initialize `qa_results.db` schema (one-time setup) |
| `verify_connection.py` | Smoke test OAuth2 + Pega connectivity before running tests |
| `_migrate_golden_sessions.py` | Annotate golden session files with `detection_mode` (`structured` if tool calls have `call_id`, else `regex`) |
| `DeepEval_Pega/DeepEval_Pega.py` | Reflex web UI (State class + all UI components) |
| `rxconfig.py` | Reflex config — plugins (SitemapPlugin, TailwindV4Plugin, RadixThemesPlugin with theme), frontend package pins |
| `api/` | FastAPI REST API package — `auth.py` (OAuth 2.0 / JWT), `models.py` (Pydantic schemas), `evaluation_runner.py` (async task manager), `routers/` (projects, datasets, evaluations) |
| `run_api.py` | REST API entrypoint (`python run_api.py` → port 8100) |
| `project_templates/` | Per-agent JSON configs (one per Pega agent project) |
| `golden_sessions/` | Recorded `golden_*.json` datasets + companion `profile_*.json` metadata files |
| `llm_profiles/` | Saved LLM judge profiles (`llm_profile.*.json` for settings, `.credentials.json` for secrets — gitignored) |
| `test_results/` | Saved pytest result JSON files from each evaluation run |
| `sql/` | SQLite schema DDL |

## Project Configuration Schema

Each project config in `project_templates/project_config.<name>.json` drives:
- `project_name` — display name; used to associate golden datasets with projects (profile JSONs store this)
- `agent_type` — `"conversational"` (default) or `"step_agent"`; step agents require a case ID at evaluation time
- `connection` — base URL, agent name, A2A app path, optional token URL override, optional `conversation_list_data_view` (custom data view name for listing conversations)
- `agent_identity` — role/domain/org/off-topic guidance (passed to LLM judge as ground truth)
- `workflows` — array of named workflow definitions; replaces the legacy single `workflow` object
- `hallucination_context` — factual statements the agent should never contradict
- `tool_patterns` / `step_agent_patterns` / `silent_upload_patterns` — regex rule sets

### `workflows` Array Schema

```json
"workflows": [
  {
    "id": "complaint_resolution",
    "description": "A customer files a complaint which is investigated and resolved.",
    "stages": [
      {"name": "Complaint Registration", "description": "Provide details about the complaint."},
      {"name": "Investigation", "description": "Investigate the complaint."},
      {"name": "Resolution", "description": "Resolve the complaint."}
    ]
  },
  {
    "id": "faq",
    "description": "Agent answers knowledge-base questions with no case creation.",
    "stages": []
  }
]
```

- `id` — slug used with `--workflow-id` at capture time and stored in the profile JSON
- `stages` — empty array for FAQ/no-workflow sessions; `test_conversation_completeness` skips when both `id` is null and `stages` is empty
- **Backward compat:** configs with the legacy `workflow` (singular) object are auto-coerced via `_normalize_workflows()` — no migration required for existing configs or golden sessions

### Golden Session `expected_case_type` Field

Each turn in a golden session can have an `expected_case_type` field (display name like "Reset Password" or Pega class name like "Uplus-FS-Work-ResetAccount"). This drives `test_business_case_adherence` which verifies the correct business case type is created per turn.

### Managing Configs via the UI

The **Configuration tab** in the Reflex UI supports full CRUD for project configs:
- **Load** — select any saved config from the dropdown to populate the form
- **Save** — writes `project_templates/project_config.<safe_name>.json` and syncs OAuth credentials to `.env`
- **Delete** — removes the config file and all associated `profile_*.json` and `golden_*.json` files whose `project_name` matches; requires confirmation

### Conversation Listing in Golden Datasets Tab

The **Manage Golden Datasets → Capture from Pega** mode includes a "Load Conversations" feature:
- After selecting a project config, click **"Load Conversations"** to query the Pega instance for all Autopilot conversation instances
- Uses `POST /prweb/api/application/v2/data_views/D_pxAutopilotConversations` (with fallback to `D_pxAutopilotConversationList`)
- Populates a dropdown with `PXCONV-XXXXX (Status) — Date — Creator` entries
- Selecting a conversation fetches details via `D_pxAutopilotConversation` and shows turn count, message count, status, and first user message
- Manual text input remains as a fallback if the listing API is unavailable
- If a project config specifies `connection.conversation_list_data_view`, that data view name is tried first

The **REST API** section at the bottom of the Configuration tab provides:
- **Start/Stop** the FastAPI API server (runs as a subprocess on a configurable port, default 8100)
- **Register OAuth clients** — generates a `client_id` + `client_secret` pair; secret is shown once and stored as a SHA-256 hash in `api_clients.json`
- **Delete OAuth clients** — removes a client from the registry with confirmation dialog
- **OpenAPI Docs** link — opens the Swagger UI when the server is running

## Web UI — Evaluation Tab

The **Evaluation tab** provides a guided workflow:
1. **Load Project Configuration** — select from available projects by name (dropdown shows `project_name`, not filename); this filters the golden datasets shown
2. **Select Golden Dataset** — only datasets associated with the selected project are displayed (matched by `project_name` in companion profile files); hidden entirely until a project is selected; read-only cards (no edit actions)
3. **Select Metrics** — toggle individual metrics with configurable thresholds
4. **Run Evaluation** — launches pytest subprocess; results displayed inline

## Web UI — Golden Datasets Tab

The **Golden Datasets tab** provides full CRUD for golden datasets:

**Existing Datasets** section at the top:
- Project filter dropdown (shows unique project names from loaded datasets)
- Dataset cards with action buttons: rename (pencil), replace (replace icon), delete (trash)
- Delete removes both the golden file and its companion profile file

**Create / Replace Dataset** section below:
- When in **replace mode** (triggered by clicking replace on a card), an orange banner shows which dataset will be overwritten; submitting any creation method replaces the target file instead of creating a new one
- Four creation methods: Capture from Pega, Agent Output JSON, Manual JSON, Upload File
- The heading dynamically changes to "Replace Using" when in replace mode

## SQLite Schema (`qa_results.db`)

- **`runs`** — one row per test suite execution (pass rate, latency ratio, golden file, report path)
- **`test_scores`** — one row per test per run (outcome, score, duration, failure reason)
- **`turn_metrics`** — one row per conversation turn (latency ms, hallucination score, tools called, `tools_source_summary` e.g. `"tool_calls:3,regex:1"`, step agents)

## LLM Judge Provider

The judge LLM is selected at runtime via `LLM_PROVIDER` in `.env` (or set from the Configuration tab in the web UI). The `create_judge_llm()` factory in `test_surface_agents.py` returns the correct instance; the `judge` pytest fixture calls it so all LLM-judge tests use whichever provider is configured.

**Shared base:** `_JudgeLLMBase(DeepEvalBaseLLM)` holds `_fix_json_escapes()` and `_flatten_data_in_response()` — all providers inherit these to normalise the JSON that DeepEval metrics expect.

**Google Gemini** (`GeminiJudgeLLM`): reads `GEMINI_API_KEY` and `GEMINI_MODEL_ID` (default `gemini-2.5-flash`). Key is read at instantiation time (not at module import), so a key saved via the UI takes effect without restarting the process.

**OpenAI** (`OpenAIJudgeLLM`): reads `OPENAI_API_KEY` and `OPENAI_MODEL_ID` (default `gpt-4o`). Uses the chat completions API with the shared `_SYSTEM_INSTRUCTION` as the system message.

**AWS Bedrock** (`BedrockJudgeLLM`): supports models containing `anthropic`, `amazon`, or `meta` in the ID with per-family request/response shapes. Supports both foundation model IDs (e.g., `anthropic.claude-3-5-sonnet-20241022-v2:0`) and inference profile IDs (e.g., `us.anthropic.claude-sonnet-4-6`) — newer models require inference profiles for on-demand invocation. Two auth methods, selected by `AWS_AUTH_METHOD`:
- `access_keys` — explicit `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`
- `sso_profile` — `boto3.Session(profile_name=AWS_PROFILE)`; requires a prior `aws sso login`

**GitHub Copilot** (`GitHubCopilotJudgeLLM`): reads `GITHUB_COPILOT_TOKEN` and `GITHUB_COPILOT_MODEL_ID` (default `openai/gpt-4o`). Uses the OpenAI-compatible API at `https://models.github.ai/inference`. Requires a GitHub PAT with Copilot access. Model IDs use `publisher/model` format (e.g., `openai/gpt-4o`, `deepseek/deepseek-r1`, `meta/llama-3.3-70b-instruct`). Available models are fetched from the GitHub Models catalog at `https://models.github.ai/v1/models`.

**Anthropic** (`AnthropicJudgeLLM`): reads `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL_ID` (default `claude-sonnet-4-5`). Uses the first-party Anthropic Messages API via the official `anthropic` SDK with the shared `_SYSTEM_INSTRUCTION` as the `system` prompt. This is distinct from the AWS Bedrock path, which reaches Claude models through AWS. Available models are fetched from Anthropic's catalog at `https://api.anthropic.com/v1/models`.

The **Configuration tab** in the Reflex UI exposes a "LLM Judge Settings" section where the provider, auth method, and credentials can be changed and saved to `.env` without editing the file manually. Each provider has a **dynamic model dropdown** with a refresh button that fetches available models from the provider's API (Gemini model list, Bedrock foundation models + inference profiles, OpenAI model list, GitHub Models catalog). A "Test Connection" button makes a minimal live API call using the current form values before saving. LLM settings can also be saved as **named profiles** (stored in `llm_profiles/`) for quick switching between providers — loading a profile activates it by writing to `.env`.

## Subscription Sign-In (OAuth) for the LLM Judge

OpenAI, GitHub Copilot, and Anthropic each support **two auth methods**, selected per provider in the Configuration tab (mirrors the Bedrock "Access Keys / SSO Profile" toggle) and persisted via `OPENAI_AUTH_METHOD` / `COPILOT_AUTH_METHOD` / `ANTHROPIC_AUTH_METHOD` (`api_key` default, or `oauth`):

- **API Key** — the existing behaviour (PAT/API key).
- **Sign in** — use the user's existing subscription via the provider's official OAuth flow. Implemented in `llm_oauth.py`:
  - **GitHub Copilot** — GitHub OAuth **device-code** flow. The long-lived GitHub token is stored and exchanged at use time for a short-lived Copilot bearer token; calls go to `api.githubcopilot.com`.
  - **OpenAI (ChatGPT Plus/Pro)** — "Sign in with ChatGPT" **OAuth PKCE** flow (Codex CLI client). Judge calls the ChatGPT backend **Responses API** (`chatgpt.com/backend-api/codex/responses`).
  - **Anthropic (Claude Pro/Max)** — "Sign in with Claude" **OAuth PKCE** flow (Claude Code client). Judge calls the standard Messages API with a Bearer token + `anthropic-beta: oauth-2025-04-20` and the required Claude Code system preamble.

OAuth tokens (with refresh tokens) are stored in the gitignored vault `llm_profiles/.credentials.json` under the reserved `__oauth__` key — never in `.env`. Only the `*_AUTH_METHOD` flag goes to `.env`, so the headless pytest subprocess and REST API resolve and auto-refresh tokens non-interactively. Sign-in itself is interactive and happens in the UI.

> **Note:** OAuth **client IDs are not bundled** — each flow reads its client ID from a required env var (`COPILOT_OAUTH_CLIENT_ID`, `OPENAI_OAUTH_CLIENT_ID`, `ANTHROPIC_OAUTH_CLIENT_ID`) and raises a clear config error if it is unset (see `_required_env()` in `llm_oauth.py`). The GitHub Copilot device-code flow is officially supported and stable. The OpenAI ChatGPT and Anthropic Claude subscription OAuth flows reuse the Codex CLI / Claude Code endpoints (not officially documented public APIs) and may change; endpoints are also overridable via env vars.

## Technology Stack

- **Python 3.14** — runtime
- **Reflex 0.9.3** — web UI framework (theme configured via `rx.plugins.RadixThemesPlugin` in `rxconfig.py`)
- **Pydantic 2.12** — data models
- **FastAPI** — REST API
- **DeepEval 3.8** — evaluation framework
- **SQLite** — results storage
- **FastMCP** — MCP server for LLM agent integration

## Key Architectural Decisions (see `Decisions.md`)

- **ADR-001:** A2A JSON-RPC as primary protocol (tests full orchestration stack, not just agent surface)
- **ADR-002:** DX API v2 for conversation insight (canonical source of truth for tool calls over regex alone)
- **ADR-004:** Gemini 2.5 Flash as LLM judge (fast, cost-effective)
- **ADR-005:** 4-source step agent detection (single source insufficient for all Pega agent types)
