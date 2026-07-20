"""OAuth / subscription sign-in for LLM judge providers.

Provides an alternative to API keys for three providers, using each provider's
official OAuth flow so users can evaluate with their existing subscription:

  - GitHub Copilot : GitHub OAuth **device-code** flow. The long-lived GitHub
                     OAuth token is stored and exchanged at use time for a
                     short-lived Copilot bearer token.
  - OpenAI         : "Sign in with ChatGPT" **OAuth PKCE** flow (the Codex CLI
                     flow). Calls the ChatGPT backend Responses API.
  - Anthropic      : "Sign in with Claude" **OAuth PKCE** flow (the Claude Code
                     flow). Calls the standard Messages API with a Bearer token.

Tokens are persisted in the gitignored credentials vault
(``llm_profiles/.credentials.json``) under the reserved ``__oauth__`` key, so
both the Reflex UI and the headless pytest subprocess can resolve (and refresh)
them without re-prompting the user.

NOTE: The OpenAI ChatGPT and Anthropic Claude subscription OAuth flows are not
officially documented public APIs — they reuse the Codex CLI / Claude Code
client IDs and endpoints, which may change without notice. The GitHub Copilot
device-code flow is stable and widely used. All endpoints/client IDs below can
be overridden via environment variables if a provider changes them.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests

# --------------------------------------------------------------------------
# Vault storage (shared with the UI credential vault, namespaced under __oauth__)
# --------------------------------------------------------------------------

_VAULT_PATH = Path(__file__).resolve().parent / "llm_profiles" / ".credentials.json"
_OAUTH_NS = "__oauth__"


def _read_vault() -> dict:
    if not _VAULT_PATH.exists():
        return {}
    try:
        return json.loads(_VAULT_PATH.read_text())
    except Exception:
        return {}


def _write_vault(vault: dict) -> None:
    _VAULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _VAULT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(vault, indent=2) + "\n")
    tmp.replace(_VAULT_PATH)
    try:
        _VAULT_PATH.chmod(0o600)
    except OSError:
        pass


def _load(provider: str) -> dict:
    return _read_vault().get(_OAUTH_NS, {}).get(provider, {})


def _store(provider: str, data: dict) -> None:
    vault = _read_vault()
    oauth = vault.get(_OAUTH_NS, {})
    oauth[provider] = data
    vault[_OAUTH_NS] = oauth
    _write_vault(vault)


# Pending PKCE material (verifier + state) is held server-side between building
# the authorize URL and completing the exchange, so it survives independently of
# any UI/session state and is never sent to the browser.
_PENDING_NS = "__oauth_pending__"


def _store_pending(provider: str, data: dict) -> None:
    vault = _read_vault()
    pending = vault.get(_PENDING_NS, {})
    pending[provider] = data
    vault[_PENDING_NS] = pending
    _write_vault(vault)


def _load_pending(provider: str) -> dict:
    return _read_vault().get(_PENDING_NS, {}).get(provider, {})


def _clear_pending(provider: str) -> None:
    vault = _read_vault()
    pending = vault.get(_PENDING_NS, {})
    if provider in pending:
        pending.pop(provider, None)
        vault[_PENDING_NS] = pending
        _write_vault(vault)


def sign_out(provider: str) -> None:
    """Remove stored OAuth credentials for a provider."""
    vault = _read_vault()
    oauth = vault.get(_OAUTH_NS, {})
    if provider in oauth:
        oauth.pop(provider, None)
        vault[_OAUTH_NS] = oauth
        _write_vault(vault)


def is_signed_in(provider: str) -> bool:
    creds = _load(provider)
    if provider == "copilot":
        return bool(creds.get("github_token"))
    return bool(creds.get("refresh") or creds.get("access"))


def status_label(provider: str) -> str:
    """Human-readable signed-in indicator for the UI."""
    creds = _load(provider)
    if not is_signed_in(provider):
        return ""
    who = creds.get("account_label") or creds.get("account_id") or ""
    return f"Signed in{(' as ' + who) if who else ''}"


# --------------------------------------------------------------------------
# PKCE helpers
# --------------------------------------------------------------------------

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def generate_pkce() -> Tuple[str, str]:
    """Return (verifier, challenge) for an OAuth PKCE flow."""
    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def _new_state() -> str:
    # Hex (not base64url) and 32 bytes wide, matching the maintained Claude Code
    # subscription clients. claude.ai's authorize endpoint rejects shorter /
    # non-hex state values with "Invalid request format".
    return secrets.token_hex(32)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _decode_jwt_payload(token: str) -> dict:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return {}
        pad = "=" * (-len(parts[1]) % 4)
        return json.loads(base64.urlsafe_b64decode(parts[1] + pad).decode("utf-8"))
    except Exception:
        return {}


def _required_env(name: str, purpose: str) -> str:
    """Return a required environment variable or raise a clear config error."""
    value = (os.environ.get(name) or "").strip()
    if value:
        return value
    raise RuntimeError(
        f"Missing required environment variable: {name}. "
        f"Set it to use {purpose}."
    )


# ==========================================================================
# GitHub Copilot — OAuth device-code flow
# ==========================================================================

GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
COPILOT_TOKEN_URL = os.environ.get(
    "COPILOT_TOKEN_URL", "https://api.github.com/copilot_internal/v2/token"
)
COPILOT_API_BASE = os.environ.get("COPILOT_API_BASE", "https://api.githubcopilot.com")


def _copilot_client_id() -> str:
    return _required_env("COPILOT_OAUTH_CLIENT_ID", "GitHub Copilot OAuth sign-in")

_COPILOT_HEADERS = {
    "Editor-Version": "vscode/1.96.0",
    "Editor-Plugin-Version": "copilot-chat/0.23.0",
    "Copilot-Integration-Id": "vscode-chat",
    "User-Agent": "GitHubCopilotChat/0.23.0",
}


def copilot_request_headers() -> Dict[str, str]:
    """Editor identification headers required by the Copilot API."""
    return dict(_COPILOT_HEADERS)


def copilot_start_device_login(timeout: int = 15) -> Dict[str, Any]:
    """Begin the device-code flow. Returns user_code, verification_uri, device_code, interval."""
    client_id = _copilot_client_id()
    resp = requests.post(
        GITHUB_DEVICE_CODE_URL,
        data={"client_id": client_id, "scope": "read:user"},
        headers={"Accept": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    if "device_code" not in data:
        raise RuntimeError(f"GitHub device-code request failed: {data}")
    return data


def copilot_poll_once(device_code: str, timeout: int = 15) -> Tuple[str, Optional[str]]:
    """Poll the token endpoint once.

    Returns (status, error_or_none) where status is one of:
    "ok" (login complete + stored), "pending", "slow_down", or "error".
    """
    client_id = _copilot_client_id()
    resp = requests.post(
        GITHUB_ACCESS_TOKEN_URL,
        data={
            "client_id": client_id,
            "device_code": device_code,
            "grant_type": GITHUB_DEVICE_GRANT,
        },
        headers={"Accept": "application/json"},
        timeout=timeout,
    )
    data = resp.json()
    if data.get("access_token"):
        creds = _load("copilot")
        creds["github_token"] = data["access_token"]
        creds.pop("copilot_token", None)
        creds.pop("copilot_expires_ms", None)
        _store("copilot", creds)
        return "ok", None
    err = data.get("error")
    if err == "authorization_pending":
        return "pending", None
    if err == "slow_down":
        return "slow_down", None
    return "error", data.get("error_description") or err or "unknown error"


def _copilot_exchange_token(github_token: str, timeout: int = 20) -> Tuple[str, int, str]:
    """Exchange the GitHub OAuth token for a short-lived Copilot bearer token.

    GitHub rejects this endpoint (403 "only use approved clients") unless the
    request carries the editor identification headers, so they are always sent.
    Returns (copilot_token, expires_at_ms, api_base) — the API base is taken
    from the response's ``endpoints.api`` (business/enterprise accounts use a
    different host than the public default).
    """
    headers = copilot_request_headers()
    headers["Authorization"] = f"token {github_token}"
    headers["Accept"] = "application/json"
    resp = requests.get(COPILOT_TOKEN_URL, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    token = data.get("token")
    if not token:
        raise RuntimeError(f"Copilot token exchange failed: {data}")
    expires_at = int(data.get("expires_at", 0)) * 1000
    api_base = ""
    endpoints = data.get("endpoints")
    if isinstance(endpoints, dict):
        api_base = endpoints.get("api", "") or ""
    return token, expires_at, api_base


def get_copilot_token(skew_ms: int = 120_000) -> str:
    """Return a valid Copilot bearer token, exchanging/refreshing as needed."""
    creds = _load("copilot")
    github_token = creds.get("github_token")
    if not github_token:
        raise RuntimeError(
            "GitHub Copilot is not signed in. Use 'Sign in with GitHub' in the LLM "
            "Judge settings, or switch this provider back to API-key auth."
        )
    cached = creds.get("copilot_token")
    expires = int(creds.get("copilot_expires_ms", 0))
    if cached and expires > _now_ms() + skew_ms:
        return cached
    token, expires_at, api_base = _copilot_exchange_token(github_token)
    creds["copilot_token"] = token
    creds["copilot_expires_ms"] = expires_at or (_now_ms() + 25 * 60 * 1000)
    if api_base:
        creds["copilot_api_base"] = api_base
    _store("copilot", creds)
    return token


def get_copilot_api_base() -> str:
    """Return the Copilot API base for the signed-in account.

    Falls back to the public default until a token has been exchanged (which is
    when the account-specific endpoint becomes known).
    """
    base = _load("copilot").get("copilot_api_base", "")
    return base or COPILOT_API_BASE


def copilot_list_models(timeout: int = 20) -> List[str]:
    """List models available to the signed-in Copilot subscription."""
    token = get_copilot_token()
    api_base = get_copilot_api_base()
    headers = copilot_request_headers()
    headers["Authorization"] = f"Bearer {token}"
    headers["Accept"] = "application/json"
    resp = requests.get(f"{api_base}/models", headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("data", data) if isinstance(data, dict) else data
    ids = []
    for it in items or []:
        mid = it.get("id") if isinstance(it, dict) else None
        if mid:
            ids.append(mid)
    return sorted(set(ids))


# ==========================================================================
# OpenAI — "Sign in with ChatGPT" OAuth PKCE flow (Codex)
# ==========================================================================

OPENAI_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
OPENAI_TOKEN_URL = "https://auth.openai.com/oauth/token"
OPENAI_REDIRECT_URI = os.environ.get(
    "OPENAI_OAUTH_REDIRECT_URI", "http://localhost:1455/auth/callback"
)
OPENAI_SCOPE = "openid profile email offline_access"
OPENAI_CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
_OPENAI_AUTH_CLAIM = "https://api.openai.com/auth"


def _openai_client_id() -> str:
    return _required_env("OPENAI_OAUTH_CLIENT_ID", "OpenAI ChatGPT OAuth sign-in")


def openai_build_authorize_url() -> Tuple[str, str, str]:
    """Return (authorize_url, state, code_verifier) for the ChatGPT sign-in flow.

    The verifier/state are also persisted server-side so the exchange does not
    depend on UI state surviving the round-trip.
    """
    client_id = _openai_client_id()
    verifier, challenge = generate_pkce()
    state = _new_state()
    _store_pending("openai", {"verifier": verifier, "state": state})
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": OPENAI_REDIRECT_URI,
        "scope": OPENAI_SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": "codex_cli_rs",
    }
    return f"{OPENAI_AUTHORIZE_URL}?{urlencode(params)}", state, verifier


def parse_code_input(value: str) -> Tuple[Optional[str], Optional[str]]:
    """Accept a full redirect URL, a ``code#state`` pair, or a raw code."""
    v = (value or "").strip()
    if not v:
        return None, None
    if v.startswith("http://") or v.startswith("https://"):
        from urllib.parse import parse_qs, urlparse

        qs = parse_qs(urlparse(v).query)
        return (qs.get("code") or [None])[0], (qs.get("state") or [None])[0]
    if "#" in v:
        code, st = v.split("#", 1)
        return code or None, st or None
    if "code=" in v:
        from urllib.parse import parse_qs

        qs = parse_qs(v)
        return (qs.get("code") or [None])[0], (qs.get("state") or [None])[0]
    return v, None


def _openai_account_id(access_token: str) -> str:
    payload = _decode_jwt_payload(access_token)
    auth = payload.get(_OPENAI_AUTH_CLAIM) if isinstance(payload, dict) else None
    acct = auth.get("chatgpt_account_id") if isinstance(auth, dict) else None
    return acct if isinstance(acct, str) else ""


def _openai_account_label(access_token: str) -> str:
    payload = _decode_jwt_payload(access_token)
    return payload.get("email") or payload.get("preferred_username") or ""


def openai_complete_login(code: str, verifier: Optional[str] = None, timeout: int = 30) -> None:
    """Exchange an authorization code for tokens and persist them.

    The PKCE verifier is read from the server-side pending store; the optional
    ``verifier`` argument overrides it (kept for backward compatibility).
    """
    client_id = _openai_client_id()
    if not verifier:
        verifier = _load_pending("openai").get("verifier")
    if not verifier:
        raise RuntimeError("Sign-in was not started (no PKCE verifier). Click sign in again.")
    resp = requests.post(
        OPENAI_TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": OPENAI_REDIRECT_URI,
        },
        timeout=timeout,
    )
    if not resp.ok:
        raise RuntimeError(f"OpenAI token exchange failed: HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    access = data.get("access_token")
    refresh = data.get("refresh_token")
    expires_in = data.get("expires_in")
    if not (isinstance(access, str) and isinstance(refresh, str) and expires_in):
        raise RuntimeError("OpenAI token exchange failed: missing fields in response")
    _store(
        "openai",
        {
            "access": access,
            "refresh": refresh,
            "expires_ms": _now_ms() + int(expires_in) * 1000,
            "account_id": _openai_account_id(access),
            "account_label": _openai_account_label(access),
        },
    )
    _clear_pending("openai")


def _openai_refresh(refresh_token: str, timeout: int = 30) -> dict:
    client_id = _openai_client_id()
    resp = requests.post(
        OPENAI_TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        },
        timeout=timeout,
    )
    if not resp.ok:
        raise RuntimeError(f"OpenAI token refresh failed: HTTP {resp.status_code}: {resp.text}")
    return resp.json()


def get_openai_credentials(skew_ms: int = 300_000) -> Tuple[str, str]:
    """Return (access_token, chatgpt_account_id), refreshing if near expiry."""
    creds = _load("openai")
    if not creds.get("refresh") and not creds.get("access"):
        raise RuntimeError(
            "OpenAI is not signed in. Use 'Sign in with ChatGPT' in the LLM Judge "
            "settings, or switch this provider back to API-key auth."
        )
    if creds.get("access") and int(creds.get("expires_ms", 0)) > _now_ms() + skew_ms:
        return creds["access"], creds.get("account_id", "")
    data = _openai_refresh(creds["refresh"])
    access = data.get("access_token", creds.get("access"))
    refresh = data.get("refresh_token", creds["refresh"])
    expires_in = data.get("expires_in", 3600)
    creds.update(
        {
            "access": access,
            "refresh": refresh,
            "expires_ms": _now_ms() + int(expires_in) * 1000,
            "account_id": _openai_account_id(access) or creds.get("account_id", ""),
            "account_label": _openai_account_label(access) or creds.get("account_label", ""),
        }
    )
    _store("openai", creds)
    return creds["access"], creds.get("account_id", "")


def openai_chatgpt_generate(model: str, system: str, prompt: str, timeout: int = 120) -> str:
    """Run a single-turn completion against the ChatGPT backend Responses API.

    The ChatGPT subscription backend requires the Responses API format and a
    streaming (SSE) response, so this assembles the request and accumulates the
    output text from the stream.
    """
    access, account_id = get_openai_credentials()
    headers = {
        "Authorization": f"Bearer {access}",
        "ChatGPT-Account-Id": account_id,
        "OpenAI-Beta": "responses=v1",
        "OpenAI-Originator": "codex",
        "originator": "codex_cli_rs",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    body = {
        "model": model,
        "instructions": system,
        "input": [
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": prompt}]}
        ],
        "store": False,
        "stream": True,
    }
    text_parts: List[str] = []
    with requests.post(
        OPENAI_CODEX_RESPONSES_URL, headers=headers, json=body, stream=True, timeout=timeout
    ) as resp:
        if not resp.ok:
            raise RuntimeError(
                f"ChatGPT Responses API failed: HTTP {resp.status_code}: {resp.text[:500]}"
            )
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            payload = raw[len("data:"):].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            etype = event.get("type", "")
            if etype == "response.output_text.delta":
                delta = event.get("delta")
                if isinstance(delta, str):
                    text_parts.append(delta)
            elif etype == "response.completed" and not text_parts:
                text_parts.append(_extract_responses_text(event.get("response", {})))
    return "".join(text_parts)


def _extract_responses_text(response_obj: dict) -> str:
    out = []
    for item in response_obj.get("output", []) or []:
        if item.get("type") == "message":
            for part in item.get("content", []) or []:
                if part.get("type") in ("output_text", "text"):
                    out.append(part.get("text", ""))
    return "".join(out)


# Curated model list for the OpenAI OAuth (ChatGPT subscription) path. There is
# no clean public models endpoint for the ChatGPT backend, so this is a static
# default the user can override by typing a model id.
OPENAI_OAUTH_MODELS = ["gpt-5", "gpt-5-codex", "gpt-4o", "o4-mini", "o3"]


# ==========================================================================
# Anthropic — "Sign in with Claude" OAuth PKCE flow (Claude Code)
# ==========================================================================

ANTHROPIC_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
ANTHROPIC_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
# Claude *subscription* (Pro/Max) flow: the code is displayed on the
# platform.claude.com callback page and the scope is inference-only. (The
# console.anthropic.com redirect + org:create_api_key scope is the separate
# Console/API-key flow and fails for subscription accounts.)
ANTHROPIC_REDIRECT_URI = os.environ.get(
    "ANTHROPIC_OAUTH_REDIRECT_URI", "https://platform.claude.com/oauth/code/callback"
)
ANTHROPIC_SCOPES = os.environ.get(
    "ANTHROPIC_OAUTH_SCOPE", "user:inference user:profile"
)
ANTHROPIC_OAUTH_BETA = "oauth-2025-04-20"
# OAuth/subscription tokens require this Claude Code system preamble as the
# first system block, otherwise the Messages API rejects the request.
ANTHROPIC_OAUTH_SYSTEM_PREFIX = (
    "You are Claude Code, Anthropic's official CLI for Claude."
)


def _anthropic_client_id() -> str:
    return _required_env("ANTHROPIC_OAUTH_CLIENT_ID", "Anthropic Claude OAuth sign-in")

ANTHROPIC_OAUTH_MODELS = [
    "claude-opus-4-1-20250805",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-20250514",
    "claude-3-5-haiku-20241022",
]


def anthropic_build_authorize_url() -> Tuple[str, str, str]:
    """Return (authorize_url, state, code_verifier) for the Claude sign-in flow.

    The verifier/state are persisted server-side so the exchange always has the
    exact ``state`` Anthropic requires, regardless of UI state.
    """
    client_id = _anthropic_client_id()
    verifier, challenge = generate_pkce()
    state = _new_state()
    _store_pending("anthropic", {"verifier": verifier, "state": state})
    params = {
        "code": "true",
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": ANTHROPIC_REDIRECT_URI,
        "scope": ANTHROPIC_SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    return f"{ANTHROPIC_AUTHORIZE_URL}?{urlencode(params)}", state, verifier


def anthropic_complete_login(
    code: str, verifier: Optional[str] = None, state: Optional[str] = None, timeout: int = 30
) -> None:
    """Exchange an authorization code (the bare code) for tokens.

    The Anthropic token endpoint **requires** the ``state`` field — omitting it
    yields HTTP 400 "Invalid request format". The verifier/state are read from
    the server-side pending store by default so they are always present and
    correct; the optional arguments override them (backward compatible).
    """
    client_id = _anthropic_client_id()
    pending = _load_pending("anthropic")
    verifier = verifier or pending.get("verifier")
    state = state or pending.get("state")
    if not verifier:
        raise RuntimeError("Sign-in was not started (no PKCE verifier). Click sign in again.")
    if not state:
        raise RuntimeError("Sign-in state missing. Click sign in again to restart the flow.")
    # Strip any trailing "#state" the user may have pasted along with the code.
    if "#" in code:
        code = code.split("#", 1)[0]
    # Use form-encoding + a browser-like user agent, matching the maintained
    # Claude Code OAuth implementations (JSON is also accepted but form is the
    # canonical shape used by Claude Code / opencode).
    resp = requests.post(
        ANTHROPIC_TOKEN_URL,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "DeepEval-Pega/1.0 (oauth)",
        },
        data={
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": code,
            "state": state,
            "redirect_uri": ANTHROPIC_REDIRECT_URI,
            "code_verifier": verifier,
        },
        timeout=timeout,
    )
    if not resp.ok:
        raise RuntimeError(f"Anthropic token exchange failed: HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    access = data.get("access_token")
    refresh = data.get("refresh_token")
    expires_in = data.get("expires_in")
    if not (isinstance(access, str) and isinstance(refresh, str) and expires_in):
        raise RuntimeError("Anthropic token exchange failed: missing fields in response")
    _store(
        "anthropic",
        {
            "access": access,
            "refresh": refresh,
            "expires_ms": _now_ms() + int(expires_in) * 1000,
            "account_label": data.get("account", {}).get("email_address", "")
            if isinstance(data.get("account"), dict) else "",
        },
    )
    _clear_pending("anthropic")


def _anthropic_refresh(refresh_token: str, timeout: int = 30) -> dict:
    client_id = _anthropic_client_id()
    resp = requests.post(
        ANTHROPIC_TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        },
        timeout=timeout,
    )
    if not resp.ok:
        raise RuntimeError(f"Anthropic token refresh failed: HTTP {resp.status_code}: {resp.text}")
    return resp.json()


def get_anthropic_token(skew_ms: int = 120_000) -> str:
    """Return a valid Anthropic OAuth access token, refreshing if near expiry."""
    creds = _load("anthropic")
    if not creds.get("refresh") and not creds.get("access"):
        raise RuntimeError(
            "Anthropic is not signed in. Use 'Sign in with Claude' in the LLM Judge "
            "settings, or switch this provider back to API-key auth."
        )
    if creds.get("access") and int(creds.get("expires_ms", 0)) > _now_ms() + skew_ms:
        return creds["access"]
    data = _anthropic_refresh(creds["refresh"])
    access = data.get("access_token")
    if not access:
        raise RuntimeError("Anthropic token refresh failed: missing access_token")
    creds.update(
        {
            "access": access,
            "refresh": data.get("refresh_token", creds["refresh"]),
            "expires_ms": _now_ms() + int(data.get("expires_in", 3600)) * 1000,
        }
    )
    _store("anthropic", creds)
    return access


def anthropic_list_models(timeout: int = 20) -> List[str]:
    """List models available to the signed-in Claude subscription.

    Uses the OAuth access token (Bearer) plus the OAuth beta header against the
    standard Anthropic models endpoint. Falls back to the curated static list on
    any error.
    """
    try:
        token = get_anthropic_token()
        resp = requests.get(
            "https://api.anthropic.com/v1/models",
            headers={
                "Authorization": f"Bearer {token}",
                "anthropic-beta": ANTHROPIC_OAUTH_BETA,
                "anthropic-version": "2023-06-01",
            },
            params={"limit": 100},
            timeout=timeout,
        )
        resp.raise_for_status()
        ids = [m.get("id") for m in resp.json().get("data", []) if m.get("id")]
        if ids:
            return sorted(ids)
    except Exception:
        pass
    return list(ANTHROPIC_OAUTH_MODELS)
