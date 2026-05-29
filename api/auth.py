from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import string
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import jwt
from fastapi import Depends, Form, HTTPException, Request, status
from fastapi.openapi.models import OAuthFlowClientCredentials, OAuthFlows
from fastapi.security import OAuth2

from .models import ClientInfo, TokenResponse

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CLIENTS_FILE = _PROJECT_ROOT / "api_clients.json"

ALL_SCOPES = [
    "projects:read", "projects:write",
    "datasets:read", "datasets:write",
    "evaluations:read", "evaluations:write",
]

oauth2_scheme = OAuth2(
    flows=OAuthFlows(
        clientCredentials=OAuthFlowClientCredentials(
            tokenUrl="/oauth/token",
        ),
    ),
)


def _load_registry() -> dict:
    if _CLIENTS_FILE.exists():
        return json.loads(_CLIENTS_FILE.read_text())

    client_id = os.environ.get("API_CLIENT_ID")
    client_secret = os.environ.get("API_CLIENT_SECRET")
    jwt_secret = os.environ.get("API_JWT_SECRET", secrets.token_hex(32))
    if not client_id or not client_secret:
        raise RuntimeError(
            "No api_clients.json found and API_CLIENT_ID / API_CLIENT_SECRET "
            "env vars are not set. See api_clients.example.json."
        )
    return {
        "jwt_secret_key": jwt_secret,
        "token_expiry_minutes": 60,
        "clients": [
            {
                "client_id": client_id,
                "client_secret": client_secret,
                "scopes": [
                    "projects:read", "projects:write",
                    "datasets:read", "datasets:write",
                    "evaluations:read", "evaluations:write",
                ],
            }
        ],
    }


def _find_client(registry: dict, client_id: str) -> Optional[dict]:
    for c in registry.get("clients", []):
        if c["client_id"] == client_id:
            return c
    return None


def _verify_secret(provided: str, stored: str) -> bool:
    if stored.startswith("sha256:"):
        hashed = "sha256:" + hashlib.sha256(provided.encode()).hexdigest()
        return hmac.compare_digest(hashed, stored)
    return hmac.compare_digest(provided, stored)


def create_access_token(registry: dict, client_id: str, scopes: list[str]) -> tuple[str, int]:
    expiry_minutes = registry.get("token_expiry_minutes", 60)
    now = datetime.now(timezone.utc)
    payload = {
        "sub": client_id,
        "scopes": scopes,
        "iat": now,
        "exp": now + timedelta(minutes=expiry_minutes),
    }
    token = jwt.encode(payload, registry["jwt_secret_key"], algorithm="HS256")
    return token, expiry_minutes * 60


async def oauth_token(
    request: Request,
    grant_type: str = Form("client_credentials"),
    client_id: Optional[str] = Form(None),
    client_secret: Optional[str] = Form(None),
    username: Optional[str] = Form(None),
    password: Optional[str] = Form(None),
) -> TokenResponse:
    if grant_type != "client_credentials":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only grant_type=client_credentials is supported",
        )

    resolved_id = client_id or username or ""
    resolved_secret = client_secret or password or ""

    if not resolved_id or not resolved_secret:
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("basic "):
            import base64
            try:
                decoded = base64.b64decode(auth_header[6:]).decode()
                resolved_id, resolved_secret = decoded.split(":", 1)
            except Exception:
                pass

    if not resolved_id or not resolved_secret:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide client_id/client_secret as form fields or Basic auth header",
        )

    registry = _load_registry()
    client = _find_client(registry, resolved_id)
    if not client or not _verify_secret(resolved_secret, client["client_secret"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid client credentials",
        )

    token, expires_in = create_access_token(registry, resolved_id, client["scopes"])
    return TokenResponse(
        access_token=token,
        expires_in=expires_in,
        scope=" ".join(client["scopes"]),
    )


async def get_current_client(authorization: str = Depends(oauth2_scheme)) -> ClientInfo:
    if authorization.lower().startswith("bearer "):
        token = authorization[7:]
    else:
        token = authorization
    registry = _load_registry()
    try:
        payload = jwt.decode(token, registry["jwt_secret_key"], algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    return ClientInfo(
        client_id=payload["sub"],
        scopes=payload.get("scopes", []),
    )


def require_scope(scope: str):
    async def _check(client: ClientInfo = Depends(get_current_client)) -> ClientInfo:
        if scope not in client.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Scope '{scope}' is required",
            )
        return client
    return _check


# --- Client registry management (called from Reflex UI) ---

def _generate_client_id() -> str:
    prefix = "deepeval"
    suffix = secrets.token_hex(8)
    return f"{prefix}-{suffix}"


def _generate_client_secret() -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(48))


def _save_registry(registry: dict) -> None:
    _CLIENTS_FILE.write_text(json.dumps(registry, indent=2) + "\n")


def _ensure_registry() -> dict:
    if _CLIENTS_FILE.exists():
        return json.loads(_CLIENTS_FILE.read_text())
    registry = {
        "jwt_secret_key": secrets.token_hex(32),
        "token_expiry_minutes": 60,
        "clients": [],
    }
    _save_registry(registry)
    return registry


def register_client(description: str = "") -> dict:
    registry = _ensure_registry()
    client_id = _generate_client_id()
    client_secret = _generate_client_secret()
    secret_hash = "sha256:" + hashlib.sha256(client_secret.encode()).hexdigest()
    entry = {
        "client_id": client_id,
        "client_secret": secret_hash,
        "description": description,
        "scopes": list(ALL_SCOPES),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    registry["clients"].append(entry)
    _save_registry(registry)
    return {"client_id": client_id, "client_secret": client_secret}


def list_clients() -> list[dict]:
    registry = _ensure_registry()
    return [
        {
            "client_id": c["client_id"],
            "description": c.get("description", ""),
            "scopes": c.get("scopes", []),
            "created_at": c.get("created_at", ""),
        }
        for c in registry.get("clients", [])
    ]


def delete_client(client_id: str) -> bool:
    registry = _ensure_registry()
    original_count = len(registry.get("clients", []))
    registry["clients"] = [
        c for c in registry.get("clients", []) if c["client_id"] != client_id
    ]
    if len(registry["clients"]) < original_count:
        _save_registry(registry)
        return True
    return False


def get_clients_file_path() -> Path:
    return _CLIENTS_FILE
