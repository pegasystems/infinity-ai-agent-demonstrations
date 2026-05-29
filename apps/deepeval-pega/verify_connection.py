import os
import requests
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
base_url = os.getenv("AGENTX_BASE_URL", "").rstrip("/")
agent_name = os.getenv("AGENT_NAME")
client_id = os.getenv("PEGA_CLIENT_ID")
client_secret = os.getenv("PEGA_CLIENT_SECRET")
token_url = f"{base_url}/prweb/PRRestService/oauth2/v1/token"

print(f"--- Connection Verification Diagnostic ---")
print(f"Target Server: {base_url}")
print(f"Agent Name:    {agent_name}")
print(f"Client ID:     {client_id[:4]}...{client_id[-4:] if client_id else ''}")

# 1. Authenticate
print(f"\n[1] Requesting OAuth Token from: {token_url}")
try:
    auth_resp = requests.post(
        token_url,
        data={"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret},
        verify=False
    )
    if auth_resp.status_code != 200:
        print(f"❌ Auth Failed: {auth_resp.status_code}")
        print(f"Response: {auth_resp.text}")
        exit(1)
    
    access_token = auth_resp.json().get("access_token")
    print(f"✅ Auth Successful. Token obtained.")
except Exception as e:
    print(f"❌ Auth Exception: {e}")
    exit(1)


# 2. Start Conversation
# Try multiple base paths for the API
api_base_paths = [
    "/prweb/api/application/v2/ai-agents", # Application API v2 (What works in your screenshot)
    "/prweb/api/AgentX/v1/agents",          # Old AgentX V1
    "/prweb/PRRestService/AgentX/v1/agents" # Generic
]

context_id = f"TestConn_{int(time.time())}"
payload = {
    "contextID": context_id,
    "activeChannel": "web",
    "activeChannelID": "test_script",
    "executeStarterQuestion": True
}
headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}

from urllib.parse import quote

# List of agents to try
agents_to_try = [
    agent_name,  # The one from .env
]

print(f"\n[2] Attempting to start conversation with correct API path...")

success = False
for api_path in api_base_paths:
    if success: break
    create_url_base = f"{base_url}{api_path}"
    print(f"\nScanning API Base: {create_url_base}")
    
    for agent in agents_to_try:
        if not agent: continue
        
        # Pega API often needs URL encoded IDs (esp '!')
        encoded_agent = quote(agent)
        
        # Note: App V2 uses 'conversations' (plural), AgentX V1 uses 'conversation' (singular)
        endpoint_suffix = "conversations" if "application/v2" in api_path else "conversation"
        
        create_url = f"{create_url_base}/{encoded_agent}/{endpoint_suffix}"
        print(f"   > Endpoint: {create_url}")
        
        try:
            start_resp = requests.post(create_url, json=payload, headers=headers, verify=False)
            
            if start_resp.status_code in [200, 201]:
                data = start_resp.json()
                print(f"   ✅ SUCCESS! Conversation Started.")
                
                # Extract ID details
                context_id_val =  data.get("ID") or data.get("contextID") 
                
                print(f"\n--- SERVER RESPONSE DETAILS ---")
                print(f"Full Response Keys: {list(data.keys())}")
                print(f"Server Conversation/Context ID: {context_id_val}")
                print(f"-------------------------------")
                success = True
                break
            elif start_resp.status_code != 404:
                 print(f"   ⚠️ Found endpoint but failed ({start_resp.status_code}): {start_resp.text[:100]}")
        except Exception as e:
            print(f"     ❌ Exception: {e}")

if not success:
    print("\n❌ Could not start conversation.")
