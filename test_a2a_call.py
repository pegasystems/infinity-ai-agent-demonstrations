"""Quick script to test the A2A execute endpoint payload format."""
import requests
import json

# Get token
token_url = "https://genai-cdh-demo.pega.net/prweb/PRRestService/oauth2/v1/token"
resp = requests.post(token_url, data={
    "grant_type": "client_credentials",
    "client_id": "15050789044613319261",
    "client_secret": "3D820EAB37AB35F4A55EDBD6F63F83CB",
})
token = resp.json()["access_token"]
print(f"[Auth] Token acquired")

# Call the execute endpoint with A2A JSON-RPC message format
execute_url = "https://genai-cdh-demo.pega.net/prweb/app/surface1dev/api/agent2agent/v1/ai-agents/UPLUS-UBANK!MARKETINGAUTOMATIONAGENT/execute"

payload = {
    "jsonrpc": "2.0",
    "id": "test-1",
    "method": "message/send",
    "params": {
        "message": {
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": "Generate a tweet for our new product launch."
                }
            ]
        }
    }
}

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

print(f"[Request] POST {execute_url}")
print(f"[Request] Payload: {json.dumps(payload, indent=2)}")

resp = requests.post(execute_url, json=payload, headers=headers)
print(f"\n[Response] Status: {resp.status_code}")
print(f"[Response] Headers: {dict(resp.headers)}")
print(f"[Response] Body: {resp.text[:3000]}")
