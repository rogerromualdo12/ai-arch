"""
Microsoft Agent Framework orchestration — SequentialBuilder.

Two specialized agents run in sequence over Mansfield invoices:
  1) RetrieverAgent  — calls RAG search tools
  2) AnalystAgent    — writes the final cited answer

Uses Gemini via the OpenAI-compatible endpoint (same pattern as Semantic Kernel /
AutoGen-style multi-agent orchestration, on Microsoft Agent Framework).

Usage:
  python -m apps.maf_sequential_rag "Summarize Best Oil diesel invoices"
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from agent_framework.openai import OpenAIChatCompletionClient
from agent_framework.orchestrations import SequentialBuilder

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env", override=True)
os.environ.pop("GOOGLE_API_KEY", None)

from rag.agent import list_vendors, search_invoices, get_invoice  # noqa: E402
from rag.config import CHAT_MODEL, GEMINI_API_KEY  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


def build_gemini_client() -> OpenAIChatCompletionClient:
    # Chat Completions API (Gemini OpenAI-compat). OpenAIChatClient uses /responses,
    # which Gemini does not expose the same way.
    return OpenAIChatCompletionClient(
        model=CHAT_MODEL,
        api_key=GEMINI_API_KEY,
        base_url=GEMINI_OPENAI_BASE,
    )


def build_workflow():
    chat = build_gemini_client()

    retriever = chat.as_agent(
        name="RetrieverAgent",
        instructions=(
            "You retrieve Mansfield invoice evidence. "
            "ALWAYS use tools before answering. "
            "Prefer list_vendors for catalog questions, search_invoices for topical "
            "queries (pass vendor aliases jp/awg/best oil/hg when relevant), and "
            "get_invoice for exact invoice numbers. "
            "Return compact factual findings with invoice numbers, vendors, totals, sources."
        ),
        tools=[list_vendors, search_invoices, get_invoice],
    )

    analyst = chat.as_agent(
        name="AnalystAgent",
        instructions=(
            "You are a senior invoice analyst. Given retriever findings, write a clear "
            "final answer for the user. Cite InvoiceNumber + VendorName. "
            "If evidence is missing, say so. Do not invent numbers."
        ),
    )

    return SequentialBuilder(participants=[retriever, analyst]).build()


async def run_query(question: str) -> str:
    workflow = build_workflow()
    result = await workflow.run(question)

    # WorkflowRunResult helpers (MAF versions differ slightly)
    for attr in ("get_outputs", "get_messages", "get_final_response"):
        method = getattr(result, attr, None)
        if not callable(method):
            continue
        try:
            value = method()
            if inspect.isawaitable(value):
                value = await value
            if value is None:
                continue
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list) and value:
                last = value[-1]
                text = getattr(last, "text", None) or getattr(last, "content", None) or str(last)
                if text:
                    return str(text)
            text = getattr(value, "text", None)
            if text:
                return str(text)
        except Exception:
            continue

    texts: list[str] = []
    for event in getattr(result, "events", []) or []:
        data = getattr(event, "data", None)
        if data is None:
            continue
        text = getattr(data, "text", None)
        if text:
            texts.append(str(text))
            continue
        messages = getattr(data, "messages", None)
        if isinstance(messages, list):
            for m in messages:
                t = getattr(m, "text", None) or getattr(m, "content", None)
                if t:
                    texts.append(str(t))
    if texts:
        return texts[-1]
    return str(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MAF Sequential RAG orchestration")
    parser.add_argument(
        "question",
        nargs="?",
        default="List vendors with counts, then give one sample Best Oil total.",
    )
    args = parser.parse_args(argv)

    try:
        answer = asyncio.run(run_query(args.question))
        print(answer)
        return 0
    except Exception as e:
        logging.error("MAF orchestration failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
