import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    HttpOptions,
    Tool,
    ToolCodeExecution,
)

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)
os.environ.pop("GOOGLE_API_KEY", None)

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=HttpOptions(api_version="v1"),
)

model_id = "gemini-3.5-flash"
code_execution_tool = Tool(code_execution=ToolCodeExecution())
response = client.models.generate_content(
    model=model_id,
    contents="Calculate 20th fibonacci number. Then find the nearest palindrome to it.",
    config=GenerateContentConfig(
        tools=[code_execution_tool],
        temperature=0,
    ),
)
print("# Code:")
print(response.executable_code)
print("# Outcome:")
print(response.code_execution_result)
