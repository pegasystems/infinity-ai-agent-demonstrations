# DeepEval Pega — Agent Evaluation Framework

**Project-agnostic evaluation suite for Pega AI agents** using [DeepEval](https://github.com/confident-ai/deepeval), a pluggable LLM judge (Google Gemini, AWS Bedrock, OpenAI, or GitHub Copilot), and Pega DX API conversation insight. Includes a [Reflex](https://reflex.dev) web UI for managing configurations, golden datasets, LLM judge profiles, and running evaluations — a [REST API](#rest-api) for programmatic access — plus an MCP server so LLM agents can query test results directly.

---

## Table of Contents

- [What This Does](#what-this-does)
- [Prerequisites](#prerequisites)
- [Setting Up the Application](#setting-up-the-application)
- [Creating the Database](#creating-the-database)
- [Running the Reflex Application](#running-the-reflex-application)
- [Navigating the Reflex Application](#navigating-the-reflex-application)
- [Creating a Configuration](#creating-a-configuration)
- [Creating a Golden Dataset](#creating-a-golden-dataset)
- [Running an Evaluation](#running-an-evaluation)
- [REST API](#rest-api)
- [Running the MCP Server](#running-the-mcp-server)
- [Project Structure](#project-structure)
- [Key Components](#key-components)

---

## What This Does

This project provides an end-to-end testing pipeline that:

1. **Sends tasks** to a live Pega agent via **dual transport** — AgentX Application v2 API (same path as the Pega UI) or [A2A (Agent-to-Agent) JSON-RPC protocol](https://google.github.io/A2A/)
2. **Queries conversation insight** from Pega's `D_pxAutopilotConversation` data view and Stages API
3. **Detects tool invocations** by pattern-matching assistant message content or extracting structured `tool_calls` from agent output (configurable per-project)
4. **Evaluates response quality** using DeepEval metrics backed by a pluggable LLM judge (Google Gemini, AWS Bedrock, OpenAI, or GitHub Copilot)
5. **Records golden sessions** with auto-sensing — captures a multi-turn flow from the Pega UI and automatically derives gate patterns, latency baselines, and an evaluation profile
6. **Replays & regresses** golden sessions with 13 conversational tests (knowledge retention, completeness, role adherence, tool fidelity, latency, case lifecycle, business case adherence, step agents, hallucination, contextual precision, contextual recall, toxicity, bias)
7. **Provides a Reflex web UI** for managing project configurations, golden datasets, LLM judge profiles, and triggering evaluations without the command line
8. **Exposes a REST API** (FastAPI) secured with OAuth 2.0 for programmatic access to project configs, golden datasets, evaluations, and LLM profiles
9. **Generates agent-ready analytics** — parses test results into SQLite and serves them via an MCP server so conversation agents can answer questions like "Is this build ready?" or "Show me the slow turns."

```
                                     ┌───────────────────────────┐
                                     │  Reflex Web UI            │
                                     │  • Configuration editor   │
                                     │  • LLM judge profiles     │
                                     │  • Golden dataset mgmt    │
                                     │  • Evaluation runner      │
                                     │  • REST API management    │
                                     └─────────┬─────────────────┘
                                               │
                                     ┌─────────▼─────────────────┐
                                     │  FastAPI REST API (:8100)  │
                                     │  • OAuth 2.0 (JWT)        │
                                     │  • /projects, /datasets   │
                                     │  • /evaluations           │
                                     │  • /metrics, /llm-profiles│
                                     └─────────┬─────────────────┘
                                               │
┌─────────────┐  --transport agentx    ┌───────▼──────────────────┐
│  pytest      │ ──── AgentX v2 API ──►│  Pega Agent              │
│  test suite  │ ──── A2A JSON-RPC ──► │  (any project)           │
│              │ ◄───────────────────  │                          │
└──────┬───────┘     AgentResponse     └──────────────────────────┘
       │
       │  Query DX API v2
       ▼
┌──────────────────────────────────┐
│  D_pxAutopilotConversation       │──► Tool Detection (configurable)
│  /cases/{key}/stages             │──► Step Agent Detection
└──────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐     ┌──────────────────────────────────┐
│  DeepEval Metrics                │     │  Results Pipeline                │
│  • KnowledgeRetention            │     │                                  │
│  • Hallucination / Faithfulness  │     │  conftest.py → SQLite ETL        │
│  • Contextual Precision/Recall   │     │  → MCP Server (qa_results)       │
│  • Toxicity / Bias               │     │  → Markdown QA Report            │
│  Judge: Gemini/Bedrock/OpenAI/GH │     │                                  │
└──────────────────────────────────┘     └──────────────────────────────────┘
```

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.11+ (tested on 3.14) |
| **Node.js** | Required by Reflex for the frontend build |
| **LLM API Key** | At least one of: Google Gemini API key, OpenAI API key, AWS Bedrock credentials, or GitHub PAT with Copilot access |
| **Pega credentials** | OAuth2 client ID + secret for the agent's environment |
| **Network access** | Connectivity to the Pega instance hosting the agent |

---

## Setting Up the Application

### 1. Clone and Create a Virtual Environment

```bash
git clone <repo-url> DeepEval_Pega
cd DeepEval_Pega
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

The key dependencies are:
- `reflex==0.9.3` — web UI framework
- `fastapi` — REST API framework
- `google-genai` — Google Gemini client (LLM judge)
- `openai` — OpenAI and GitHub Copilot client (LLM judge)
- `boto3` — AWS Bedrock client (LLM judge)
- `deepeval` — evaluation metrics framework
- `pydantic==2.12` — data models
- `pytest` — test runner
- `PyJWT` — JWT token handling for API auth
- `python-dotenv` — `.env` file loading
- `fastmcp` — MCP server framework

### 3. Configure the `.env` File

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your values:

```dotenv
# ── Pega Connection ──────────────────────────────────────────
AGENTX_BASE_URL=https://your-pega-instance.example.com/
AGENT_NAME=@BASECLASS!YOURAGENTNAME

# ── OAuth2 Credentials ──────────────────────────────────────
PEGA_CLIENT_ID=<your-client-id>
PEGA_CLIENT_SECRET=<your-client-secret>

# ── Project Config ───────────────────────────────────────────
PROJECT_CONFIG=project_config.myagent.json

# ── LLM Judge Provider ──────────────────────────────────────
# Options: "gemini", "bedrock", "openai", "copilot", or "anthropic"
LLM_PROVIDER=gemini

# Google Gemini (when LLM_PROVIDER=gemini)
GEMINI_API_KEY=<your-gemini-api-key>
GEMINI_MODEL_ID=gemini-2.5-flash

# OpenAI (when LLM_PROVIDER=openai)
OPENAI_AUTH_METHOD=api_key           # "api_key" or "oauth" (Sign in with ChatGPT subscription)
OPENAI_API_KEY=<your-openai-api-key>
OPENAI_MODEL_ID=gpt-4o

# AWS Bedrock (when LLM_PROVIDER=bedrock)
AWS_AUTH_METHOD=access_keys          # "access_keys" or "sso_profile"
AWS_ACCESS_KEY_ID=<your-key-id>
AWS_SECRET_ACCESS_KEY=<your-secret>
AWS_REGION=us-east-1
AWS_BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
# For SSO: set AWS_AUTH_METHOD=sso_profile, AWS_PROFILE=<name>, then run: aws sso login --profile <name>

# GitHub Copilot (when LLM_PROVIDER=copilot)
COPILOT_AUTH_METHOD=api_key          # "api_key" (GitHub PAT) or "oauth" (Copilot subscription)
GITHUB_COPILOT_TOKEN=<your-github-pat>
GITHUB_COPILOT_MODEL_ID=openai/gpt-4o

# Anthropic (when LLM_PROVIDER=anthropic)
ANTHROPIC_AUTH_METHOD=api_key        # "api_key" or "oauth" (Sign in with Claude subscription)
ANTHROPIC_API_KEY=<your-anthropic-api-key>
ANTHROPIC_MODEL_ID=claude-sonnet-4-5
```

| Variable | Required | Description |
|---|---|---|
| `AGENTX_BASE_URL` | Yes | Base URL of your Pega instance |
| `AGENT_NAME` | Yes | The Pega agent rule name (`@CLASSNAME!NAME` format) |
| `PEGA_CLIENT_ID` | Yes | OAuth2 client ID for the Pega environment |
| `PEGA_CLIENT_SECRET` | Yes | OAuth2 client secret |
| `PROJECT_CONFIG` | No | Project config filename in `project_templates/` (auto-discovered if omitted) |
| `LLM_PROVIDER` | No | LLM judge provider: `gemini` (default), `bedrock`, `openai`, `copilot`, or `anthropic` |
| `GEMINI_API_KEY` | If Gemini | Google AI Gemini API key |
| `GEMINI_MODEL_ID` | No | Gemini model name (default `gemini-2.5-flash`) |
| `OPENAI_AUTH_METHOD` | If OpenAI | `api_key` (default) or `oauth` (Sign in with ChatGPT subscription) |
| `OPENAI_API_KEY` | If OpenAI + api_key | OpenAI API key |
| `OPENAI_MODEL_ID` | No | OpenAI model name (default `gpt-4o`) |
| `AWS_AUTH_METHOD` | If Bedrock | `access_keys` (default) or `sso_profile` |
| `AWS_ACCESS_KEY_ID` | If Bedrock + access_keys | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | If Bedrock + access_keys | AWS secret key |
| `AWS_PROFILE` | If Bedrock + sso_profile | Named AWS CLI/SSO profile |
| `AWS_REGION` | No | AWS region for Bedrock (default `us-east-1`) |
| `AWS_BEDROCK_MODEL_ID` | No | Bedrock model ID (default `anthropic.claude-3-5-sonnet-20241022-v2:0`) |
| `COPILOT_AUTH_METHOD` | If Copilot | `api_key` (default, GitHub PAT) or `oauth` (Copilot subscription) |
| `GITHUB_COPILOT_TOKEN` | If Copilot + api_key | GitHub PAT with Copilot access |
| `GITHUB_COPILOT_MODEL_ID` | No | GitHub Copilot model (default `openai/gpt-4o`; uses `publisher/model` format) |
| `ANTHROPIC_AUTH_METHOD` | If Anthropic | `api_key` (default) or `oauth` (Sign in with Claude subscription) |
| `ANTHROPIC_API_KEY` | If Anthropic + api_key | First-party Anthropic API key |
| `ANTHROPIC_MODEL_ID` | No | Anthropic model name (default `claude-sonnet-4-5`) |

### 4. Verify Your Connection

```bash
python verify_connection.py
```

This confirms OAuth2 authentication and basic connectivity to your Pega instance.

---

## Creating the Database

The framework stores structured test results in a SQLite database (`qa_results.db`) for querying via the MCP server.

```bash
python create_db.py
```

This reads the schema from `sql/sqlite_schema.sql` and creates three tables:

| Table | Description |
|---|---|
| `runs` | One row per test suite execution (pass rate, latency ratio, golden file, etc.) |
| `test_scores` | One row per test per run (outcome, score, duration, failure reason) |
| `turn_metrics` | One row per conversation turn per run (latency, hallucination score, tools called) |

> The database is also auto-initialized when `conftest.py` pushes results after a test run, so manual creation is optional but recommended for first-time setup.

---

## Running the Reflex Application

The web UI is built with [Reflex](https://reflex.dev) and provides a browser-based interface for the entire evaluation workflow.

### Start the Application

```bash
reflex run
```

This starts both the frontend and backend. By default:
- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:8000

To use a custom backend port:

```bash
reflex run --backend-port 3001
```

> On first run, Reflex will install Node.js dependencies and build the frontend — this may take a minute or two.

---

## Navigating the Reflex Application

The web UI has three main tabs accessible from the top navigation bar:

### 1. Evaluation Tab

The primary workspace for running DeepEval evaluations.

**Workflow:**
1. **Load Project Configuration** — Select a project from the dropdown (shows project names). This filters the golden datasets shown in step 3.
2. **Select Metrics** — Choose which DeepEval metrics to include. Available metrics:
   - Knowledge Retention, Hallucination, Conversation Completeness, Role Adherence
   - Pega Tool Correctness, Business Case Lifecycle, Business Case Adherence
   - Contextual Precision, Contextual Recall
   - Toxicity, Bias
   - Each metric has a configurable pass/fail threshold (0.0-1.0)
3. **Select Golden Dataset** — Only datasets associated with the selected project are displayed (matched by `project_name` in companion profile files). Hidden entirely until a project is selected. Cards show turn count, tools used, and recording date.
4. **Run Evaluation** — Click "Run Evaluation" to execute `test_golden_session.py` with your selections. Live log output streams in the UI.

### 2. Golden Datasets Tab

Full CRUD management for golden datasets, organized into two sections:

#### Existing Datasets

Browse all golden datasets with a project filter dropdown. Each dataset card shows:
- Dataset name, turn count, tools count, and recording date
- **Rename** (pencil icon) — Change the dataset's display name
- **Replace** (replace icon) — Enter replace mode, then use any creation method below to overwrite the dataset's content while preserving its name and metadata
- **Delete** (trash icon) — Remove the dataset and its companion profile file (with confirmation)

#### Create / Replace Dataset

Create new golden datasets (or replace an existing one when in replace mode) using four methods:

- **Capture from Pega** — Enter a conversation ID (PXCONV-XXXXX) from a completed Pega agent session. The script pulls the full conversation from `D_pxAutopilotConversation` and auto-senses gate patterns, tool usage, and latency baselines.
- **Agent Output JSON** — Paste raw Pega agent output JSON (must contain `conversation_history` with `tool_calls`). Tools are extracted from the structured data rather than regex patterns.
- **Manual JSON** — Write or paste golden session JSON by hand. A template is provided.
- **Upload File** — Drag-and-drop an existing `golden_*.json` file.

When in replace mode, an orange banner indicates which dataset will be overwritten. Submitting any creation method replaces the target dataset instead of creating a new file. Click "Cancel" to exit replace mode.

### 3. Configuration Tab

Create or edit project configurations through a guided form:

- **Project Information** — Name and version
- **Connection Settings** — Pega base URL, agent name, A2A app path, optional token URL override
- **Agent Identity** — Role description (used by the LLM judge), domain, organization, off-topic guidance
- **Workflow Configuration** — Multiple workflows with IDs, descriptions, and stages arrays
- **Hallucination Context** — Factual statements the LLM judge uses to determine grounding (one per line)

Configs are saved to `project_templates/project_config.<name>.json` and become available in the Evaluation and Golden Datasets tabs.

#### LLM Judge Settings

The Configuration tab includes an **LLM Judge Settings** section where you can:

- **Select provider** — Google Gemini, AWS Bedrock, OpenAI, or GitHub Copilot
- **Configure credentials** — API keys, AWS auth method (access keys or SSO profile), GitHub PATs
- **Select model** — Each provider has a dynamic model dropdown with a refresh button that fetches available models from the provider's API using your credentials. Bedrock lists both foundation models and inference profiles; GitHub Copilot fetches from the GitHub Models catalog.
- **Test connection** — Make a minimal live API call to verify credentials and model selection before saving
- **Save/load named profiles** — Store LLM configurations as named profiles in `llm_profiles/` for quick switching between providers. Loading a profile activates it by writing to `.env`.

#### REST API Management

The bottom of the Configuration tab provides an **REST API** section for:

- **Start/Stop** the FastAPI API server (runs as a subprocess on a configurable port, default 8100)
- **Register OAuth clients** — Generates a `client_id` + `client_secret` pair; the secret is shown once and stored as a SHA-256 hash in `api_clients.json`
- **Download credentials** — One-time download of generated client credentials as a text file
- **Delete OAuth clients** — Remove a client from the registry
- **OpenAPI Docs** link — Opens the Swagger UI when the server is running

---

## Creating a Configuration

You can create a project configuration either through the **Reflex UI** (Configuration tab) or by editing JSON files directly.

### Option A: Using the Reflex UI

1. Navigate to the **Configuration** tab
2. Fill in the form fields (project name, connection details, agent identity, workflow stages, hallucination context)
3. Click **Save Configuration**
4. The config is saved to `project_templates/project_config.<name>.json`

### Option B: Editing JSON Directly

```bash
cp project_templates/project_config.template.json \
   project_templates/project_config.myagent.json
```

Edit the file:

```jsonc
{
  "project_name": "My Agent Project",
  "version": "1.0",
  "connection": {
    "base_url": "https://your-pega-instance.example.com",
    "agent_name": "YOUR-AGENT-NAME",
    "a2a_app_path": "your-app",
    "token_url_override": null
  },
  "agent_identity": {
    "role": "Describe what this agent does — used by the LLM judge for evaluation.",
    "domain": "Business domain",
    "organization": "Organization name",
    "off_topic_guidance": "What topics should the agent refuse?"
  },
  "workflows": [
    {
      "id": "my_workflow",
      "description": "High-level description of the multi-step workflow.",
      "stages": [
        {"name": "Stage 1", "description": "What happens in this stage."},
        {"name": "Stage 2", "description": "What happens in this stage."}
      ]
    }
  ],
  "hallucination_context": [
    "Factual statements about what the agent can and should do.",
    "Used by the LLM judge to determine whether responses are grounded."
  ],
  "tool_patterns": { "patterns": [], "labels": {} },
  "step_agent_patterns": { "patterns": [] },
  "silent_upload_patterns": { "patterns": [] }
}
```

**What each section does for the tests:**

| Section | Used By | Purpose |
|---|---|---|
| `connection` | Transport layer | Where to find and authenticate with the agent |
| `agent_identity.role` | Knowledge retention, completeness, role adherence tests | Tells the LLM judge what behavior is "correct" |
| `agent_identity.off_topic_guidance` | Role adherence test | What the agent should refuse |
| `workflows[].stages` | Conversation completeness test | Expected end-to-end workflow steps |
| `hallucination_context` | Per-turn hallucination test | Ground-truth facts for evaluating responses |
| `tool_patterns` | Tool invocation test | Supplemental regex patterns (auto-sensing covers most) |
| `step_agent_patterns` | Step agent detection test | Supplemental patterns for case-driven agents |

Then update your `.env` to reference the new config:

```dotenv
PROJECT_CONFIG=project_config.myagent.json
```

> **Backward compatibility:** Configs with the legacy singular `workflow` object are auto-coerced to the `workflows` array format via `_normalize_workflows()` — no migration required for existing configs.

---

## Creating a Golden Dataset

Golden datasets are recorded conversations that serve as the baseline for regression testing. You can create them via the Reflex UI, the REST API, or the command line.

### Option A: Using the Reflex UI

1. Navigate to the **Golden Datasets** tab
2. Select a creation method:
   - **Capture from Pega** — Enter the conversation ID from a completed Pega session
   - **Agent Output JSON** — Paste structured agent output with `conversation_history`
   - **Manual JSON** — Write the golden session JSON by hand (use "Load Template" for the expected format)
   - **Upload File** — Upload an existing `.json` file
3. Select a project configuration (required for Capture and Agent Output modes)
4. Click the action button to create the dataset

### Option B: REST API

```bash
# Obtain a token
TOKEN=$(curl -s -X POST http://localhost:8100/oauth/token \
  -d "grant_type=client_credentials&client_id=YOUR_ID&client_secret=YOUR_SECRET" \
  | jq -r .access_token)

# Create a golden dataset from agent output
curl -X POST http://localhost:8100/datasets \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"agent_output": {...}, "session_name": "My Flow", "project_config_filename": "project_config.myagent.json"}'
```

### Option C: Command Line Capture

After running a conversation in the Pega UI, capture it:

```bash
python capture_golden_session.py PXCONV-12345 --name "My Flow"

# Annotate which workflow the session exercises
python capture_golden_session.py PXCONV-12345 --name "Complaint Flow" --workflow-id complaint_resolution

# Capture from structured agent output JSON
python capture_golden_session.py --from-json agent_output.json --workflow-id complaint_resolution

# List recent conversations to find the right ID
python capture_golden_session.py --list-recent
```

The capture script **auto-senses** from the recorded conversation:
- **Gate patterns** — regex derived from response headings, choice prompts, and transition keywords (prevents conversation desync during replay)
- **Gate timeouts** — extended timeouts for slow turns (>30s)
- **Tool usage** — detected from assistant message content patterns
- **Step agents** — detected from content patterns

Output files:
```
golden_sessions/golden_<name>_<timestamp>.json   <- golden session
golden_sessions/profile_<name>_<timestamp>.json  <- auto-sensed evaluation profile
```

> **File Attachments:** Place test files in `golden_sessions/attachments/` (e.g., PDFs). The capture script auto-detects silent file-upload turns and records the expected path.

---

## Running an Evaluation

### Option A: Using the Reflex UI

1. Go to the **Evaluation** tab
2. Select a project configuration
3. Choose metrics and adjust thresholds
4. Select a golden dataset
5. Click **Run Evaluation**
6. Results and log output appear in the UI

### Option B: REST API

```bash
# Start an evaluation (returns immediately with a run_id)
curl -X POST http://localhost:8100/evaluations \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "golden_dataset": "golden_my_flow_20260303_094655.json",
    "metrics": [
      {"name": "hallucination", "threshold": 0.5},
      {"name": "role_adherence", "threshold": 0.7}
    ],
    "project_config_filename": "project_config.myagent.json",
    "llm_profile": "My Gemini Profile"
  }'

# Poll for results
curl http://localhost:8100/evaluations/{run_id} \
  -H "Authorization: Bearer $TOKEN"
```

The `llm_profile` field is optional — when omitted, the evaluation uses whichever LLM settings are active in the server's `.env`. When provided, it loads the named profile from `llm_profiles/` and injects those credentials into the evaluation subprocess.

### Option C: Command Line (pytest)

```bash
# Run with the default golden file and project config from .env
python -m pytest test_golden_session.py -v -s

# Specify a golden file and project config explicitly
python -m pytest test_golden_session.py -v -s \
  --golden golden_sessions/golden_my_flow_20260303_094655.json \
  --project-config project_templates/project_config.myagent.json

# Select specific metrics
python -m pytest test_golden_session.py -v -s \
  --metrics hallucination,pega_tool_correctness,role_adherence

# Override transport
python -m pytest test_golden_session.py -v -s --transport a2a
```

**Available pytest flags:**

| Flag | Env Fallback | Description |
|---|---|---|
| `--golden <path>` | `GOLDEN_FILE` | Path to a golden session JSON |
| `--transport <type>` | `TRANSPORT` | `agentx` (default), `a2a`, or `auto` |
| `--project-config <path>` | `PROJECT_CONFIG` | Path to a project config JSON |
| `--metrics <ids>` | `EVAL_METRICS` | Comma-separated list of metric IDs |

### The 13 Golden Session Tests

| # | Test | Method | What It Evaluates |
|---|------|--------|-------------------|
| 1 | `test_knowledge_retention` | LLM Judge | Context from early turns persists in later turns |
| 2 | `test_conversation_completeness` | LLM Judge | Full workflow completed across all turns; skipped if no `workflow_id` annotated |
| 3 | `test_role_adherence` | LLM Judge | Agent stays in its designated role |
| 4 | `test_no_hallucination_per_turn` | LLM Judge | Each response grounded in conversation context |
| 5 | `test_contextual_precision` | LLM Judge | Retrieved context contains only relevant information |
| 6 | `test_contextual_recall` | LLM Judge | Retrieved context covers all necessary information |
| 7 | `test_toxicity` | LLM Judge | Agent responses are free of toxic or harmful content |
| 8 | `test_bias` | LLM Judge | Agent responses are free of discriminatory bias |
| 9 | `test_tool_invocations_match_golden` | Logic | Per-turn tool invocations match the baseline |
| 10 | `test_latency_regression` | Logic | Each turn within 2x golden baseline |
| 11 | `test_case_lifecycle` | Logic | Business case created and consistent |
| 12 | `test_business_case_adherence` | Logic | Correct business case type created per turn (3-tier detection: text patterns, DX API stages, case key tracking) |
| 13 | `test_step_agents_detected` | Logic | Step agents found when golden had them |

### Auto-Generated QA Report

After every test run, `conftest.py` automatically:

1. **Generates a Markdown QA report** via the LLM judge — includes a pass/fail scorecard, hallucination analysis, latency visualization, tool drift detection, and recommended actions
2. **Pushes structured results to SQLite** — populates `qa_results.db` for MCP server queries

Report files:
- `latest_qa_report.md` — overwritten with each run
- `QA_Report_<YYYYMMDD_HHMMSS>.md` — timestamped archive

---

## REST API

The REST API provides programmatic access to project configurations, golden datasets, evaluations, and LLM profiles. It is built with FastAPI and secured with OAuth 2.0 (client_credentials grant).

### Starting the API Server

```bash
# Development (auto-reload)
python run_api.py

# Production
uvicorn api.app:create_app --factory --host 0.0.0.0 --port 8100
```

The server runs on port 8100 by default. OpenAPI docs are available at http://localhost:8100/docs.

You can also start/stop the API server from the **Configuration tab** in the Reflex UI.

### Authentication

The API uses OAuth 2.0 with the `client_credentials` grant type. Clients authenticate with a `client_id` and `client_secret` to obtain a JWT access token.

**Setting up clients:**

1. **Via the Reflex UI** — Go to Configuration > REST API > OAuth Clients and click "Register Client"
2. **Via config file** — Copy `api_clients.example.json` to `api_clients.json` and fill in secrets
3. **Via environment variables** — Set `API_CLIENT_ID`, `API_CLIENT_SECRET`, and optionally `API_JWT_SECRET`

**Obtaining a token:**

```bash
curl -X POST http://localhost:8100/oauth/token \
  -d "grant_type=client_credentials&client_id=YOUR_ID&client_secret=YOUR_SECRET"
```

Response:
```json
{
  "access_token": "<jwt>",
  "token_type": "bearer",
  "expires_in": 3600,
  "scope": "projects:read projects:write datasets:read datasets:write evaluations:read evaluations:write"
}
```

Use the token in subsequent requests:
```bash
curl -H "Authorization: Bearer <jwt>" http://localhost:8100/projects
```

### API Endpoints

| Method | Path | Auth Scope | Description |
|---|---|---|---|
| `POST` | `/oauth/token` | — | Obtain JWT access token |
| `GET` | `/projects` | `projects:read` | List all project configurations |
| `POST` | `/projects` | `projects:write` | Create a project configuration |
| `GET` | `/datasets` | `datasets:read` | List all golden datasets |
| `GET` | `/datasets/by-project/{name}` | `datasets:read` | List golden datasets for a specific project |
| `GET` | `/datasets/{filename}` | `datasets:read` | Get full golden dataset JSON |
| `POST` | `/datasets` | `datasets:write` | Create golden dataset from agent output |
| `PATCH` | `/datasets/{filename}` | `datasets:write` | Rename a golden dataset |
| `PUT` | `/datasets/{filename}` | `datasets:write` | Replace golden dataset content |
| `DELETE` | `/datasets/{filename}` | `datasets:write` | Delete golden dataset and companion profile |
| `POST` | `/evaluations` | `evaluations:write` | Start an evaluation (returns 202 + run_id) |
| `GET` | `/evaluations/{run_id}` | `evaluations:read` | Get evaluation status and results |
| `GET` | `/evaluations` | `evaluations:read` | List recent evaluations |
| `GET` | `/metrics` | — | List available metrics with default thresholds |
| `GET` | `/llm-profiles` | — | List available LLM judge profiles |

### Example: Run an Evaluation via API

```bash
# 1. Get a token
TOKEN=$(curl -s -X POST http://localhost:8100/oauth/token \
  -d "grant_type=client_credentials&client_id=YOUR_ID&client_secret=YOUR_SECRET" \
  | jq -r .access_token)

# 2. List available golden datasets
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8100/datasets | jq .

# 3. List available metrics
curl -s http://localhost:8100/metrics | jq .

# 4. List available LLM profiles
curl -s http://localhost:8100/llm-profiles | jq .

# 5. Start an evaluation
RUN_ID=$(curl -s -X POST http://localhost:8100/evaluations \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "golden_dataset": "golden_my_flow_20260303_094655.json",
    "metrics": [
      {"name": "hallucination", "threshold": 0.5},
      {"name": "role_adherence", "threshold": 0.7},
      {"name": "pega_tool_correctness", "threshold": 1.0}
    ],
    "llm_profile": "My Gemini Profile"
  }' | jq -r .run_id)

# 6. Poll for results
curl -s -H "Authorization: Bearer $TOKEN" \
  http://localhost:8100/evaluations/$RUN_ID | jq .
```

---

## Running the MCP Server

The MCP server (`qa_results_mcp_server.py`) exposes test results from `qa_results.db` so LLM agents can query them via natural language.

### Available Tools

| Tool | Description |
|---|---|
| `query_qa_data` | Run a read-only SQL SELECT against the qa_results database |
| `get_report_section` | Get a specific section from the latest QA report |
| `list_recent_runs` | List recent test runs with summary info |
| `get_slow_turns` | Find turns that exceeded a latency threshold |
| `get_hallucination_details` | Get per-turn hallucination scores and reasons |
| `compare_runs` | Compare two test runs side by side |
| `get_failed_tests` | Get details about failed tests |
| `get_tools_per_turn` | Get the tools invoked per conversation turn |

### Running Locally (stdio — for Claude Desktop)

```bash
python qa_results_mcp_server.py
```

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "qa-results": {
      "command": "/path/to/DeepEval_Pega/.venv/bin/python",
      "args": ["/path/to/DeepEval_Pega/qa_results_mcp_server.py"]
    }
  }
}
```

### Running as HTTP Server (for remote clients)

```bash
PORT=8090 python qa_results_mcp_server.py --transport http
```

### Running as HTTP Server with the SSE transport (for use with Pega)

```bash
PORT=8090 python qa_results_mcp_server.py --transport sse
```

### Running with FastMCP Inspector

```bash
fastmcp dev qa_results_mcp_server.py
```

---

## Project Structure

```
DeepEval_Pega/
├── .env                              # Credentials & config (git-ignored)
├── .env.example                      # Template for .env
├── rxconfig.py                       # Reflex app configuration (plugins, theme, frontend packages)
├── requirements.txt                  # Python dependencies
│
├── DeepEval_Pega/                    # Reflex web application
│   ├── __init__.py
│   └── DeepEval_Pega.py              # Main Reflex app: state, UI components, pages
│
├── api/                              # FastAPI REST API
│   ├── __init__.py
│   ├── app.py                        # FastAPI factory, router registration, CORS
│   ├── auth.py                       # OAuth 2.0 token endpoint, JWT, scope enforcement
│   ├── models.py                     # Pydantic request/response schemas
│   ├── evaluation_runner.py          # Async background evaluation task manager
│   └── routers/
│       ├── __init__.py
│       ├── projects.py               # POST/GET /projects
│       ├── datasets.py               # POST/GET /datasets, GET /datasets/by-project/{name}
│       └── evaluations.py            # POST/GET /evaluations, GET /evaluations/{run_id}
│
├── run_api.py                        # REST API entrypoint (python run_api.py -> port 8100)
├── api_clients.example.json          # Example OAuth client registry
├── api_clients.json                  # Active OAuth client registry (git-ignored)
│
├── conftest.py                       # Shared pytest config, result collection, auto-report + DB insert
├── test_golden_session.py            # Multi-turn golden session replay (13 tests)
├── test_surface_agents.py            # Shared library: transports, LLM judges, PegaInsight, tool detection
│
├── capture_golden_session.py         # Passive recorder: captures from Pega conversation history
├── record_golden_session.py          # Active CLI recorder (legacy)
│
├── report_generator.py               # QA report: reads test results -> LLM judge -> Markdown
├── report_prompt.md                  # Prompt template for report generation
│
├── create_db.py                      # Initialize qa_results.db SQLite database
├── db_etl.py                         # ETL: parse test results -> SQLite
├── insert_results.py                 # Insert structured results into SQLite (used by conftest.py)
├── qa_results_mcp_server.py          # MCP server: exposes test results to LLM agents
│
├── golden_sessions/                  # Captured golden sessions + profiles
│   ├── attachments/                  # Test files (PDFs, DOCX) for upload turns
│   ├── golden_*.json                 # Golden session recordings
│   └── profile_*.json                # Auto-sensed evaluation profiles
│
├── project_templates/                # Project configurations
│   ├── project_config.template.json  # Blank template for new projects
│   └── project_config.*.json         # Agent-specific configs
│
├── llm_profiles/                     # LLM judge profiles
│   ├── llm_profile.*.json            # Named profile settings (provider, model, region)
│   └── .credentials.json             # Credential vault for all profiles (git-ignored)
│
├── sql/                              # SQL scripts
│   ├── sqlite_schema.sql             # Database schema (runs, test_scores, turn_metrics)
│   ├── insert_run.sql                # Insert a test run
│   ├── insert_test_score.sql         # Insert a test score
│   └── insert_turn_metric.sql        # Insert a turn metric
│
├── test_results/                     # Pytest result JSON files (auto-generated)
│
├── verify_connection.py              # Quick connection test for Pega
├── _migrate_golden_sessions.py       # Annotate golden sessions with detection_mode
├── _diag.py                          # AgentX API diagnostic
├── _test_a2a_card.py                 # Agent card reachability check
├── Decisions.md                      # Architectural decision log
└── README.md                         # This file
```

---

## Key Components

### Reflex Web UI (`DeepEval_Pega/DeepEval_Pega.py`)

The main web application built with Reflex. Contains:
- **State** — Application state management (selected metrics, datasets, configs, evaluation status, LLM profiles, API server state)
- **Evaluation Section** — Select project config, choose metrics, pick golden dataset, run evaluation
- **Golden Dataset Section** — Full CRUD: browse existing datasets with project filter, rename, replace (reuses creation UI), delete; four creation modes (Pega capture, agent output JSON, manual JSON, file upload)
- **Configuration Section** — Form-based project config editor with load/save/delete, LLM judge settings with dynamic model dropdowns and named profiles, REST API management with OAuth client registration

### LLM Judge (`test_surface_agents.py`)

Pluggable LLM judge with five provider implementations sharing a common `_JudgeLLMBase(DeepEvalBaseLLM)` base class:

| Class | Provider | Default Model |
|---|---|---|
| `GeminiJudgeLLM` | Google Gemini | `gemini-2.5-flash` (configurable) |
| `OpenAIJudgeLLM` | OpenAI | `gpt-4o` (configurable) |
| `BedrockJudgeLLM` | AWS Bedrock | Anthropic, Amazon, or Meta models (supports inference profiles) |
| `GitHubCopilotJudgeLLM` | GitHub Copilot | `openai/gpt-4o` (any model from GitHub Models catalog) |
| `AnthropicJudgeLLM` | Anthropic API | `claude-sonnet-4-5` (any model from Anthropic catalog) |

The `create_judge_llm()` factory reads `LLM_PROVIDER` from the environment and returns the appropriate instance. All providers share `_fix_json_escapes()` and `_flatten_data_in_response()` for robust DeepEval integration.

#### Subscription sign-in (OAuth)

OpenAI, GitHub Copilot, and Anthropic each support an **API key** or **Sign in** auth method (set in the Configuration tab, persisted as `OPENAI_AUTH_METHOD` / `COPILOT_AUTH_METHOD` / `ANTHROPIC_AUTH_METHOD` = `api_key` | `oauth`). The **Sign in** option lets you evaluate using your existing subscription via each provider's official OAuth flow (`llm_oauth.py`): GitHub Copilot device-code, "Sign in with ChatGPT" (PKCE → ChatGPT Responses API), and "Sign in with Claude" (PKCE → Messages API). OAuth tokens are stored with refresh tokens in the gitignored `llm_profiles/.credentials.json` vault and auto-refreshed, so headless test/REST runs stay non-interactive. The GitHub Copilot flow is officially supported; the ChatGPT and Claude subscription flows reuse the Codex CLI / Claude Code clients and may change.

LLM settings can be saved as **named profiles** (stored in `llm_profiles/`) and selected per evaluation run via the REST API's `llm_profile` field.

### `AgentXTransport` (`test_golden_session.py`)

Self-contained AgentX Application v2 API client. Handles OAuth2 token management, conversation creation, file upload, and message send — the same API path used by the Pega chat UI.

### `SurfaceAgent` (`test_surface_agents.py`)

A2A JSON-RPC client that discovers the agent endpoint from the A2A agent card. Used when `--transport a2a` or `--transport auto`.

### `PegaInsight` (`test_surface_agents.py`)

DX API v2 client that queries `D_pxAutopilotConversation` for full conversation state and `/cases/{key}/stages` for case lifecycle data.

### REST API (`api/`)

FastAPI-based REST API for programmatic access. Key modules:
- **`auth.py`** — OAuth 2.0 token endpoint, JWT creation/validation (HS256), scope-based access control, client registry management
- **`evaluation_runner.py`** — Background task manager that spawns pytest subprocesses, tracks run status, parses results, and resolves LLM profiles into subprocess environment variables
- **`models.py`** — All Pydantic request/response schemas
- **`routers/`** — Endpoint implementations for projects, datasets, and evaluations

### `capture_golden_session.py`

Passive recorder that pulls a completed conversation from Pega's DX API and reconstructs a golden session JSON. Auto-senses gate patterns, timeouts, tool usage, and step agents. Supports both conversation-ID-based capture and structured agent output JSON ingestion via `capture_from_structured_output()`.

### `conftest.py`

Shared pytest configuration that:
- Collects all test results (including stdout via a tee buffer that works with `-s`)
- Triggers QA report generation via `report_generator.py` after each session
- Pushes structured results to SQLite via `insert_results.py`

### MCP Server (`qa_results_mcp_server.py`)

FastMCP-based server exposing 8 tools for querying `qa_results.db`. Supports stdio (Claude Desktop), HTTP (remote clients), and SSE transports.

### Project Config Resolution

Project configurations are loaded with this resolution chain:

1. `--project-config` CLI flag
2. `PROJECT_CONFIG` environment variable
3. `project_config.json` in the project templates directory
4. First `project_config.*.json` glob match (skipping `template`)
