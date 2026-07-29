"""
Google Gemini Invoice Extraction Agent

An agent that orchestrates invoice extraction via tools:
  - list_invoice_pdfs
  - extract_invoice
  - extract_all_invoices

Usage:
  python invoice_extraction_agent.py
  python invoice_extraction_agent.py "Extract only the Big Bang invoice"
  python invoice_extraction_agent.py --direct   # skip agent, process all PDFs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai.types import (
    AutomaticFunctionCallingConfig,
    GenerateContentConfig,
    HttpOptions,
    Part,
)

# ---------------------------
# Setup
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)
os.environ.pop("GOOGLE_API_KEY", None)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = os.getenv("GEMINI_INVOICE_MODEL", "gemini-2.5-flash")

INPUT_PDF_DIR = BASE_DIR / "data" / "input_pdfs"
OUTPUT_JSON_DIR = BASE_DIR / "data" / "output_files"
INPUT_PDF_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

client = genai.Client(
    api_key=GEMINI_API_KEY,
    # JSON mode + AFC tools need v1beta
    http_options=HttpOptions(api_version="v1beta"),
)

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

AGENT_SYSTEM = """
You are the Invoice Extraction Agent.

Your job is to process invoice PDFs using the provided tools.
Workflow:
1) Call list_invoice_pdfs to see available files.
2) Call extract_invoice for each relevant PDF (or extract_all_invoices).
3) Summarize what was saved (output paths, invoice numbers, totals).

Never invent file names. Always use tools for extraction.
"""


def strip_markdown_json(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text


def _extract_pdf_to_json(file_path: Path) -> dict:
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
        raise ValueError(f"Empty Gemini response for {file_path.name}")
    return json.loads(strip_markdown_json(raw))


# ---------------------------
# Agent tools (called by Gemini via AFC)
# ---------------------------
def list_invoice_pdfs() -> dict:
    """List PDF invoice filenames available in the input_pdfs folder."""
    files = sorted(p.name for p in INPUT_PDF_DIR.glob("*.pdf"))
    return {"count": len(files), "files": files, "input_dir": str(INPUT_PDF_DIR)}


def extract_invoice(file_name: str) -> dict:
    """
    Extract structured invoice JSON from one PDF in input_pdfs and save it
    to output_files/<stem>.json.

    Args:
        file_name: Exact PDF filename inside input_pdfs (including .pdf).
    """
    path = INPUT_PDF_DIR / file_name
    if not path.exists():
        return {"ok": False, "error": f"File not found: {file_name}"}

    logging.info("Tool extract_invoice: %s", file_name)
    parsed = _extract_pdf_to_json(path)
    out_path = OUTPUT_JSON_DIR / f"{path.stem}.json"
    out_path.write_text(json.dumps(parsed, indent=4), encoding="utf-8")
    return {
        "ok": True,
        "file": file_name,
        "output": str(out_path),
        "InvoiceNumber": parsed.get("InvoiceNumber", ""),
        "VendorName": parsed.get("VendorName", ""),
        "totalAmountDue": parsed.get("totalAmountDue", ""),
    }


def extract_all_invoices() -> dict:
    """Extract and save JSON for every PDF currently in input_pdfs."""
    files = sorted(INPUT_PDF_DIR.glob("*.pdf"))
    if not files:
        return {"ok": False, "error": "No PDFs found in input_pdfs", "results": []}

    results = []
    for path in files:
        results.append(extract_invoice(path.name))
    ok = sum(1 for r in results if r.get("ok"))
    return {"ok": ok == len(results), "processed": ok, "total": len(results), "results": results}


TOOLS = [list_invoice_pdfs, extract_invoice, extract_all_invoices]


def run_agent(user_goal: str) -> str:
    """Run the Gemini agent so it decides which tools to call."""
    logging.info("Starting invoice extraction agent (%s)", GEMINI_MODEL)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_goal,
        config=GenerateContentConfig(
            system_instruction=AGENT_SYSTEM,
            tools=TOOLS,
            temperature=0,
            automatic_function_calling=AutomaticFunctionCallingConfig(maximum_remote_calls=20),
        ),
    )
    return (response.text or "").strip()


def run_direct() -> None:
    """Bypass the agent loop and process all PDFs (batch mode)."""
    result = extract_all_invoices()
    print(json.dumps(result, indent=4))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gemini Invoice Extraction Agent")
    parser.add_argument(
        "goal",
        nargs="?",
        default="List available invoice PDFs and extract all of them. Summarize results.",
        help="Natural-language instruction for the agent",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Skip agent orchestration; extract all PDFs directly",
    )
    args = parser.parse_args(argv)

    try:
        if args.direct:
            run_direct()
            return 0

        summary = run_agent(args.goal)
        print(summary or "(agent finished with no text summary)")
        return 0
    except Exception as e:
        logging.error("Agent failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
