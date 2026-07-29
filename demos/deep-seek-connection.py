import re
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI

endpoint = "https://rogerromualdoj-5195-resource.services.ai.azure.com/openai/v1"
deployment_name = "DeepSeek-V3.2-Speciale"

# Generate Azure AD token provider
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://ai.azure.com/.default"
)

# Initialize client using azure_ad_token_provider
# Call token_provider() with brackets () to retrieve the actual string token
client = OpenAI(
    base_url=endpoint,
    api_key=token_provider(),
)

completion = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ],
)

# 1. Extract content safely (fallback to empty string if None)
raw_content = completion.choices[0].message.content or ""

# 2. Extract everything after </think> (if present)
if "</think>" in raw_content:
    clean_answer = raw_content.split("</think>")[-1].strip()
else:
    clean_answer = raw_content.strip()

print(clean_answer)