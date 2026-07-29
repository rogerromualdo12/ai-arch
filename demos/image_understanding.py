import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai.types import HttpOptions, Part

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)
os.environ.pop("GOOGLE_API_KEY", None)

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=HttpOptions(api_version="v1"),
)

image_path = ROOT / "data" / "output_folder" / "example-image-qosqo.png"
image_bytes = image_path.read_bytes()

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=[
        "What is shown in this image?",
        Part.from_bytes(data=image_bytes, mime_type="image/png"),
    ],
)
print(response.text)
