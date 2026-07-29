"""Turn invoice JSON files into searchable text chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def invoice_to_text(data: dict[str, Any], source: str) -> str:
    """Flatten structured invoice JSON into retrieval-friendly prose."""
    products = data.get("Products") or []
    product_lines = []
    for p in products:
        if not isinstance(p, dict):
            continue
        desc = p.get("PRODUCT DESCRIPTION") or ""
        gals = p.get("GROSS GALS")
        rate = p.get("Rate")
        amount = p.get("Amount")
        product_lines.append(
            f"- {desc}: gals={gals}, rate={rate}, amount={amount}"
        )

    products_block = "\n".join(product_lines) if product_lines else "- (none)"

    return f"""Invoice document
Source file: {source}
DocumentType: {data.get('DocumentType', '')}
VendorName: {data.get('VendorName', '')}
InvoiceNumber: {data.get('InvoiceNumber', '')}
InvoiceDate: {data.get('InvoiceDate', '')}
DeliveryDate: {data.get('DeliveryDate', '')}
Terms: {data.get('terms', '')}
DueDate: {data.get('dueDate', '')}
Customer: {data.get('customer', '')}
ShipTo: {data.get('shipTo', '')}
ShipToAddress: {data.get('shipToAddress', '')}, {data.get('shipToCity', '')}, {data.get('shipToState', '')} {data.get('shipToZip', '')}
PONumber: {data.get('PONumber', '')}
BOLNumber: {data.get('BOLNumber', '')}
ShiptoLocationNumber: {data.get('ShiptoLocationNumber', '')}
Products:
{products_block}
TotalAmountDue: {data.get('totalAmountDue', '')}
""".strip()


def load_invoice_documents(data_dir: Path) -> list[dict[str, Any]]:
    """
    Recursively load *.json invoice extracts under data_dir.
    Skips non-object JSON and empty files.
    """
    docs: list[dict[str, Any]] = []
    for path in sorted(data_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue

        rel = str(path.relative_to(data_dir))
        text = invoice_to_text(data, rel)
        docs.append(
            {
                "id": rel.replace("/", "__").replace(" ", "_"),
                "text": text,
                "metadata": {
                    "source": rel,
                    "vendor": str(data.get("VendorName") or ""),
                    "invoice_number": str(data.get("InvoiceNumber") or ""),
                    "invoice_date": str(data.get("InvoiceDate") or ""),
                    "ship_to": str(data.get("shipTo") or ""),
                    "total": str(data.get("totalAmountDue") or ""),
                },
            }
        )
    return docs
