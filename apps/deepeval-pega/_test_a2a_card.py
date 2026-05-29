"""Quick check: can we reach the SURFACEFORAUTOMATION A2A agent card?"""
import os, sys, requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

TOKEN_URL = "https://genai-cdh-demo.pega.net/prweb/PRRestService/oauth2/v1/token"
CID = os.environ["PEGA_CLIENT_ID"]
CSEC = os.environ["PEGA_CLIENT_SECRET"]

CARD_URL = (
    "https://genai-cdh-demo.pega.net/prweb/app/surface1dev/"
    "api/agent2agent/v1/ai-agents/"
    "OPK0KG-SURFACE1-UIPAGES!SURFACEFORAUTOMATION/"
    ".well-known/agent.json"
)

# 1. Get token
r = requests.post(TOKEN_URL, data={"grant_type": "client_credentials"},
                   auth=(CID, CSEC), verify=False, timeout=15)
print(f"Token status: {r.status_code}")
tok = r.json().get("access_token", "")
print(f"Token prefix: {tok[:20]}" if tok else "Token: NONE")

# 2. Fetch agent card
r2 = requests.get(CARD_URL, headers={"Authorization": f"Bearer {tok}"}, verify=False, timeout=15)
print(f"Agent card status: {r2.status_code}")
print(f"Agent card body:\n{r2.text[:800]}")

# 3. Quick A2A message/send test if card loaded
if r2.status_code == 200:
    import json
    card = r2.json()
    a2a_url = card.get("url", "")
    print(f"\nA2A endpoint: {a2a_url}")

    payload = {
        "jsonrpc": "2.0",
        "id": "test-1",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "Hello, what can you help me with?"}]
            }
        }
    }
    r3 = requests.post(a2a_url, json=payload,
                       headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
                       verify=False, timeout=60)
    print(f"\nmessage/send status: {r3.status_code}")
    print(f"Response:\n{r3.text[:1000]}")
