import os
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions, Modality
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)
os.environ.pop("GOOGLE_API_KEY", None)

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=HttpOptions(api_version="v1"),
)

response = client.models.generate_content(
    model="gemini-3.1-flash-image",
    contents=("Generate an image of Cuzco with fireworks in the background."),
    config=GenerateContentConfig(
        response_modalities=[Modality.TEXT, Modality.IMAGE],
    ),
)
for part in response.candidates[0].content.parts:
    if part.text:
        print(part.text)
    elif part.inline_data:
        image = Image.open(BytesIO(part.inline_data.data))
        output_dir = ROOT / "data" / "output_folder"
        output_dir.mkdir(parents=True, exist_ok=True)
        image.save(output_dir / "example-image-qosqo.png")
