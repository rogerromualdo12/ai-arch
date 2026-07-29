"""
Mansfield Invoice RAG Agent

Gemini agent with tools over the Chroma invoice index:
  - list_vendors
  - search_invoices
  - get_invoice
  - ask_rag

Usage:
  python -m rag.agent
  python -m rag.agent "What did Best Oil charge for diesel?"
  python -m rag.agent "List Holston invoices over $100"
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from google import genai
from google.genai.types import (
    AutomaticFunctionCallingConfig,
    GenerateContentConfig,
    HttpOptions,
)

from rag.config import CHAT_MODEL, GEMINI_API_KEY, TOP_K
from rag.retrieve import get_collection, list_vendor_stats, resolve_vendors, retrieve

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=HttpOptions(api_version="v1beta"),
)

AGENT_SYSTEM = """
You are the Mansfield LTL Invoice RAG Agent.

You help users analyze extracted fuel/invoice JSON using tools.
Guidelines:
1) Prefer tools over guessing. Never invent invoice numbers or totals.
2) For broad vendor questions, call list_vendors first.
3) For semantic lookup, call search_invoices (optionally with vendor filter).
4) For a specific invoice number, call get_invoice.
5) For a natural-language answer grounded in retrieved docs, call ask_rag.
6) When summarizing, cite InvoiceNumber + VendorName + source file.
7) Vendor aliases you can pass to tools: jp, awg, best oil, hg / holston.
"""


def list_vendors():
    """List vendors in the RAG index with document counts and aliases."""
    return list_vendor_stats()


def search_invoices(query: str, vendor: str = "", top_k: int = 5):
    """
    Semantic search over indexed invoices.

    Args:
        query: What to search for (product, city, PO, ship-to, etc.).
        vendor: Optional vendor filter/alias (jp, awg, best oil, hg, or exact name).
        top_k: Max results to return.
    """
    k = int(top_k) if top_k else TOP_K
    hits = retrieve(query, top_k=k, vendor=vendor or None)
    results = []
    for hit in hits:
        meta = hit["metadata"]
        results.append(
            {
                "source": meta.get("source"),
                "vendor": meta.get("vendor"),
                "invoice_number": meta.get("invoice_number"),
                "invoice_date": meta.get("invoice_date"),
                "ship_to": meta.get("ship_to"),
                "total": meta.get("total"),
                "distance": round(float(hit["distance"]), 4),
                "snippet": (hit["document"] or "")[:500],
            }
        )
    return {
        "query": query,
        "vendor_filter": vendor or None,
        "count": len(results),
        "results": results,
    }


def get_invoice(invoice_number: str, vendor: str = ""):
    """
    Fetch invoice document(s) by exact InvoiceNumber metadata.

    Args:
        invoice_number: Invoice number string, e.g. "6893".
        vendor: Optional vendor filter/alias to disambiguate.
    """
    collection = get_collection()
    invoice_number = str(invoice_number)

    # Fetch by invoice number, then optionally filter vendor in Python
    # (avoids complex Chroma where operators that can break across versions).
    got = collection.get(
        where={"invoice_number": invoice_number},
        include=["documents", "metadatas"],
    )
    vendors = resolve_vendors(vendor or None)
    wanted = {v.lower() for v in vendors} if vendors else None

    matches = []
    for doc, meta in zip(got.get("documents") or [], got.get("metadatas") or []):
        meta = meta or {}
        if wanted and str(meta.get("vendor", "")).lower() not in wanted:
            continue
        matches.append({"metadata": meta, "document": doc})

    return {
        "ok": bool(matches),
        "invoice_number": invoice_number,
        "count": len(matches),
        "matches": matches,
    }


def ask_rag(question: str, vendor: str = "", top_k: int = 5):
    """
    Retrieve relevant invoices and answer the question grounded in that context.

    Args:
        question: Natural language analytics question.
        vendor: Optional vendor filter/alias.
        top_k: How many chunks to retrieve.
    """
    k = int(top_k) if top_k else TOP_K
    hits = retrieve(question, top_k=k, vendor=vendor or None)
    if not hits:
        return {
            "ok": False,
            "answer": "No matching invoices found in the index for that query/filter.",
            "sources": [],
        }

    context_blocks = []
    sources = []
    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        sources.append(
            {
                "source": meta.get("source"),
                "vendor": meta.get("vendor"),
                "invoice_number": meta.get("invoice_number"),
                "total": meta.get("total"),
            }
        )
        context_blocks.append(
            f"[Source {i}: {meta.get('source')} | vendor={meta.get('vendor')} | "
            f"invoice={meta.get('invoice_number')} | total={meta.get('total')}]\n"
            f"{hit['document']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""Answer using ONLY the retrieved invoice context.
If insufficient, say so. Cite invoice numbers and vendors.

Question: {question}

Context:
{context}
"""
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config=GenerateContentConfig(
            temperature=0.2,
            automatic_function_calling=AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    return {
        "ok": True,
        "answer": (response.text or "").strip(),
        "sources": sources,
        "vendor_filter": vendor or None,
    }


TOOLS = [list_vendors, search_invoices, get_invoice, ask_rag]


def run_agent(user_goal: str) -> str:
    logging.info("Starting Mansfield RAG agent (%s)", CHAT_MODEL)
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=user_goal,
        config=GenerateContentConfig(
            system_instruction=AGENT_SYSTEM,
            tools=TOOLS,
            temperature=0,
            automatic_function_calling=AutomaticFunctionCallingConfig(
                maximum_remote_calls=15
            ),
        ),
    )
    return (response.text or "").strip()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Mansfield Invoice RAG Agent")
    parser.add_argument(
        "goal",
        nargs="?",
        default=(
            "List vendors in the index, then summarize how many invoices each has "
            "and give one sample total per vendor using search_invoices."
        ),
        help="Natural-language instruction for the agent",
    )
    parser.add_argument(
        "--json-tools-demo",
        action="store_true",
        help="Call list_vendors directly (no agent loop) and print JSON",
    )
    args = parser.parse_args(argv)

    try:
        if args.json_tools_demo:
            print(json.dumps(list_vendors(), indent=2))
            return 0
        summary = run_agent(args.goal)
        print(summary or "(agent finished with no text summary)")
        return 0
    except Exception as e:
        logging.error("Agent failed: %s", e)
        return 1


if __name__ == "__main__":
    if str(Path(__file__).resolve().parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(main())
