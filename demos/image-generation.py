import base64
import os
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Modelos típicos: "gpt-image-1", "dall-e-3"
result = client.images.generate(
    model=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1"),
    prompt="Genera una imagen de cuando consiga un contrato como Director de AI",
    size="1024x1024",
    # n=1,
)

b64 = result.data[0].b64_json
if not b64 and getattr(result.data[0], "url", None):
    # dall-e-3 a veces devuelve URL en vez de b64
    import urllib.request
    with urllib.request.urlopen(result.data[0].url) as resp:
        raw = resp.read()
else:
    raw = base64.b64decode(b64)

image = Image.open(BytesIO(raw))
output_dir = ROOT / "data" / "output_folder"
output_dir.mkdir(parents=True, exist_ok=True)
out_path = output_dir / "example-image-qosqo.png"
image.save(out_path)
print(f"Saved: {out_path}")