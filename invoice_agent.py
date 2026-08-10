"""
Google Gemini invoice agent.

Same workflow as the Azure Form Recognizer + Azure OpenAI pipeline:
1) Read PDFs from input_pdfs/
2) Extract + structure invoice fields with Gemini (multimodal PDF)
3) Save JSON under output_files/

Credentials come from .env (GEMINI_API_KEY) — never hardcode secrets.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions, Part

# ---------------------------
# Setup
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)
os.environ.pop("GOOGLE_API_KEY", None)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = os.getenv("GEMINI_INVOICE_MODEL", "gemini-2.5-flash")

INPUT_PDF_DIR = BASE_DIR / data / input_pdfs
OUTPUT_JSON_DIR = BASE_DIR / data / output_files
INPUT_PDF_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

client = genai.Client(
    api_key=GEMINI_API_KEY,
    # JSON response_mime_type requires v1beta (not v1)
    http_options=HttpOptions(api_version="v1beta"),
)

# ---------------------------
# Invoice agent prompt (same rules as Azure version)
# ---------------------------
INVOICE_PROMPT = """
Role: You are an expert invoice processing agent.

Task: Extract structured invoice details into the JSON format below.
Rules(Vendor Prompt):

Do not extract any value from “Vehicle ID #” or “Truck #” as “customer” or “shipTo”.
Do not extract any value from “Load Origin” as “shipToAddress” or “shipToCity”.
Do not extract any value from “VA SALES TAX” as “PRODUCT DESCRIPTION” or “Amount”.
Do not extract any value from “Notes” or “Subtotal” as “PRODUCT DESCRIPTION”.
Do not extract any value from “Contract #” as “PONumber” or “BOLNumber”.
If only one date is present on the invoice, assume it is the "InvoiceDate".
Do not extract the dates for "DeliveryDate", leave this field empty, in the majority of the cases this field is not present on the invoice.
Do not extract the numbers of the address lines as an "orderNumber".
Do not extract "customerAddress", leave this field empty.
Do not extract customer data if it is not clearly specified as customer, leave the fields empty(customer data are the fields starting with customer, including "customer").
The field "shipTo" must be the name of the client, present commonly in the Load Destination section.
Extract the customer field from the store section, in this section there are the words "STORE" folowed by a number and below them is the name of the customer, commonly SUNBELT.
Extract what is below "Unit price" as the "Products" "Rate" for each one of the products.
Extract what is below "Total price" as the "Products" "Amount" for each one of the products.
Do not extract the "Total price as Products{["Rate"]}, if there is only one numeric value for each product on the description table extract it as the Products{["Amount"]}
If there is a value under Products{["Rate"]}, but not a value under Products{["Amount"]}, delete the value from Products{["Rate"]}(leave the field empty) and add it to Products{["Amount"]}.

Return ONLY valid JSON (no markdown) with this shape:
{
    "DocumentType": "",
    "VendorName": "",
    "DeliveryDate": "",
    "InvoiceNumber": "",
    "InvoiceDate": "",
    "terms": "",
    "orderNumber": "",
    "dueDate": "",
    "customer": "",
    "customerAddress": "",
    "customerCity": "",
    "customerState": "",
    "customerZip": "",
    "shipTo": "",
    "shipToAddress": "",
    "shipToCity": "",
    "shipToState": "",
    "shipToZip": "",
    "PONumber": "",
    "BOLNumber": "",
    "ShiptoLocationNumber": "",
    "Products": [
        {
            "PRODUCT DESCRIPTION": "",
            "PRODUCT CODE": "",
            "BOL": "",
            "GROSS GALS": "",
            "Rate": "",
            "Amount": "",
            "Date": ""
        }
    ],
    "totalAmountDue": ""
}
"""


def strip_markdown_json(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text


def analyze_invoice_with_gemini(file_path: Path) -> dict | None:
    """
    Google equivalent of Form Recognizer + OpenAI:
    send the PDF bytes to Gemini and get structured JSON back.
    """
    try:
        pdf_bytes = file_path.read_bytes()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                INVOICE_PROMPT,
                Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            ],
            config=GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        raw = (response.text or "").strip()
        if not raw:
            logging.error("Empty response from Gemini for %s", file_path.name)
            return None
        return json.loads(strip_markdown_json(raw))
    except json.JSONDecodeError as e:
        logging.error("Invalid JSON from Gemini for %s: %s", file_path.name, e)
        return None
    except Exception as e:
        logging.error("Gemini invoice agent error for %s: %s", file_path.name, e)
        return None


def main(file_name: str = "") -> None:
    start = time.time()

    if file_name:
        files = [INPUT_PDF_DIR / file_name]
    else:
        files = sorted(INPUT_PDF_DIR.glob("*.pdf"))

    if not files:
        logging.warning(
            "No PDFs found in %s — drop invoice PDFs there and re-run.",
            INPUT_PDF_DIR,
        )
        return

    for path in files:
        if not path.exists():
            logging.error("Missing file: %s", path)
            continue

        logging.info("Processing %s with Google invoice agent (%s)", path.name, GEMINI_MODEL)
        parsed = analyze_invoice_with_gemini(path)
        if not parsed:
            logging.warning("Skipped: %s", path.name)
            continue

        out_path = OUTPUT_JSON_DIR / f"{path.stem}.json"
        out_path.write_text(json.dumps(parsed, indent=4), encoding="utf-8")
        logging.info("Saved: %s", out_path)
        print(json.dumps(parsed, indent=4))

    logging.info("All done in %.2f seconds", time.time() - start)


if __name__ == "__main__":
    # Leave empty to process all PDFs in input_pdfs/
    main("")
