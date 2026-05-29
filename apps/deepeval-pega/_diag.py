"""Quick diagnostic to test AgentX conversation creation."""
import requests, os, json, time
from dotenv import load_dotenv
load_dotenv()

base = os.environ.get("AGENTX_BASE_URL", "").rstrip("/")
client_id = os.environ.get("PEGA_CLIENT_ID", "")
client_secret = os.environ.get("PEGA_CLIENT_SECRET", "")
token_url = os.environ.get("TOKEN_URL", f"{base}/prweb/PRRestService/oauth2/v1/token")

# Get token
tok_resp = requests.post(token_url, data={
    "grant_type": "client_credentials",
    "client_id": client_id,
    "client_secret": client_secret,
}, verify=False)
print(f"token status: {tok_resp.status_code}")
if tok_resp.status_code != 200:
    print(f"token body: {tok_resp.text[:500]}")
    exit(1)
token = tok_resp.json()["access_token"]
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# Agent names to try
agents = [
    "OPK0KG-SURFACE1-UIPAGES!SURFACEFORAUTOMATION",
    "OPK0KG-SURFACE1-UIPAGES!SURFACEORCHESTRATIONAGENTV8",
]

# Payloads: minimal (schema-only) vs full (with enableTracer)
payloads = {
    "minimal (schema only)": {
        "contextID": "DiagTest",
        "interactionID": f"diag_{int(time.time())}",
        "activeChannel": "web",
        "activeChannelID": "diag",
        "executeStarterQuestion": True,
    },
    "with enableTracer": {
        "contextID": "DiagTest",
        "interactionID": f"diag2_{int(time.time())}",
        "activeChannel": "web",
        "activeChannelID": "diag",
        "executeStarterQuestion": True,
        "enableTracer": True,
    },
    "empty body": {},
}

for agent in agents:
    print(f"\n{'='*60}")
    print(f"AGENT: {agent}")
    print(f"{'='*60}")
    for label, payload in payloads.items():
        url = f"{base}/prweb/api/application/v2/ai-agents/{agent}/conversations"
        r = requests.post(url, headers=headers, json=payload, verify=False)
        body_preview = r.text[:200].replace("\n", " ")
        print(f"  [{r.status_code}] {label:25s} → {body_preview}")
