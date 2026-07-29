import os
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from azure.identity import AzureCliCredential
from agent_framework.openai import OpenAIChatClient

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

# ------------------------------------------------------------------
# 1. ENTERPRISE CONFIGURATION (Using Passwordless Managed Identity)
# ------------------------------------------------------------------
# To run this, login locally using the Azure CLI: `az login`
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-resource.openai.azure.com")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

# Create local Azure credentials (no API keys hardcoded in code!)
credential = AzureCliCredential()

# ------------------------------------------------------------------
# 2. DEFINE SYSTEM TOOLS (Function Calling)
# ------------------------------------------------------------------
# In enterprise settings, agents must query secure transactional databases
async def get_customer_tier(customer_id: str) -> str:
    """Retrieves the customer support tier from internal database."""
    # Mock database lookup
    db = {"cust_101": "VIP Enterprise", "cust_102": "Standard"}
    return db.get(customer_id, "Standard")

# ------------------------------------------------------------------
# 3. INITIALIZE THE AGENT & CLIENT
# ------------------------------------------------------------------
# MAF uses OpenAIChatClient pointing to Azure for native, fast orchestration
chat_client = OpenAIChatClient(
    model=DEPLOYMENT_NAME,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=API_VERSION,
    credential=credential
)

# Package our tool into a structure the agent can understand
tools = [get_customer_tier]

# Create our AI Agent with defined instructions and tools
support_agent = chat_client.as_agent(
    name="EnterpriseSupportAgent",
    instructions=(
        "You are an enterprise assistant. ALWAYS look up the customer's tier "
        "using the 'get_customer_tier' tool before resolving their issue."
    ),
    tools=tools
)

# ------------------------------------------------------------------
# 4. EXPOSE VIA BACKEND API (FastAPI)
# ------------------------------------------------------------------
app = FastAPI(title="Enterprise AI Gateway")

class QueryRequest(BaseModel):
    customer_id: str
    user_query: str

@app.post("/v1/chat")
async def process_user_query(payload: QueryRequest):
    formatted_prompt = f"Customer ID: {payload.customer_id}. Issue: {payload.user_query}"
    
    try:
        # Run the agent through the Microsoft Agent Framework execution loop
        result = await support_agent.run(formatted_prompt)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # To run locally: python app.py
    uvicorn.run(app, host="0.0.0.0", port=8000)