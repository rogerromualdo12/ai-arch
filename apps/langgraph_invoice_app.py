"""
LangGraph orchestration app — Mansfield invoice RAG.

Graph:
  classify_intent → (list_vendors | search | lookup_invoice | answer)
                 → synthesize → END

Usage:
  python -m apps.langgraph_invoice_app "How many Best Oil invoices are there?"
  python -m apps.langgraph_invoice_app "Find diesel deliveries for JP Fuels"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
from langgraph.graph import END, StateGraph

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env", override=True)
os.environ.pop("GOOGLE_API_KEY", None)

from rag.agent import ask_rag, get_invoice, list_vendors, search_invoices  # noqa: E402
from rag.config import CHAT_MODEL, GEMINI_API_KEY  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

gemini = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=HttpOptions(api_version="v1beta"),
)


class InvoiceState(TypedDict, total=False):
    question: str
    intent: Literal["vendors", "search", "invoice", "answer"]
    vendor: str
    invoice_number: str
    tool_result: dict
    answer: str


def _llm_json(prompt: str) -> dict:
    response = gemini.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config=GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )
    return json.loads((response.text or "{}").strip() or "{}")


def classify_intent(state: InvoiceState) -> InvoiceState:
    question = state["question"]
    data = _llm_json(
        f"""
Classify this Mansfield invoice question into one intent and optional filters.
Return JSON with keys:
  intent: one of vendors | search | invoice | answer
  vendor: alias or name if mentioned (jp, awg, best oil, hg) else ""
  invoice_number: if an invoice number is mentioned else ""

Question: {question}
"""
    )
    intent = data.get("intent") or "answer"
    if intent not in {"vendors", "search", "invoice", "answer"}:
        intent = "answer"
    return {
        **state,
        "intent": intent,
        "vendor": str(data.get("vendor") or ""),
        "invoice_number": str(data.get("invoice_number") or ""),
    }


def route_intent(state: InvoiceState) -> str:
    return state.get("intent") or "answer"


def node_list_vendors(state: InvoiceState) -> InvoiceState:
    return {**state, "tool_result": list_vendors()}


def node_search(state: InvoiceState) -> InvoiceState:
    result = search_invoices(
        query=state["question"],
        vendor=state.get("vendor") or "",
        top_k=5,
    )
    return {**state, "tool_result": result}


def node_invoice(state: InvoiceState) -> InvoiceState:
    number = state.get("invoice_number") or ""
    if not number:
        # fall back to search if classifier missed the number
        return node_search(state)
    result = get_invoice(number, vendor=state.get("vendor") or "")
    return {**state, "tool_result": result}


def node_answer(state: InvoiceState) -> InvoiceState:
    result = ask_rag(
        question=state["question"],
        vendor=state.get("vendor") or "",
        top_k=5,
    )
    return {**state, "tool_result": result}


def synthesize(state: InvoiceState) -> InvoiceState:
    tool_result = state.get("tool_result") or {}
    # ask_rag already returns a natural-language answer
    if isinstance(tool_result, dict) and tool_result.get("answer"):
        return {**state, "answer": tool_result["answer"]}

    response = gemini.models.generate_content(
        model=CHAT_MODEL,
        contents=(
            "Turn this tool result into a concise user-facing answer. "
            "Cite invoice numbers and vendors when present.\n\n"
            f"Question: {state['question']}\n\n"
            f"Tool result JSON:\n{json.dumps(tool_result, indent=2)[:8000]}"
        ),
        config=GenerateContentConfig(temperature=0.2),
    )
    return {**state, "answer": (response.text or "").strip()}


def build_graph():
    graph = StateGraph(InvoiceState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("list_vendors", node_list_vendors)
    graph.add_node("search", node_search)
    graph.add_node("invoice", node_invoice)
    graph.add_node("answer", node_answer)
    graph.add_node("synthesize", synthesize)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "vendors": "list_vendors",
            "search": "search",
            "invoice": "invoice",
            "answer": "answer",
        },
    )
    for node in ("list_vendors", "search", "invoice", "answer"):
        graph.add_edge(node, "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LangGraph Mansfield invoice app")
    parser.add_argument(
        "question",
        nargs="?",
        default="List vendors and counts in the invoice index.",
    )
    args = parser.parse_args(argv)

    app = build_graph()
    final = app.invoke({"question": args.question})
    print(final.get("answer") or json.dumps(final.get("tool_result"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
