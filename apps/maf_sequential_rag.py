"""
Microsoft Agent Framework orchestration — sequential Retriever → Analyst.

Two specialized agents over Mansfield invoices:
  1) RetrieverAgent  — calls RAG search tools
  2) AnalystAgent    — writes the final cited answer

Uses OpenAI GPT directly (api.openai.com) + OpenAI embeddings collection.

Usage:
  python -m apps.maf_sequential_rag "Summarize Best Oil diesel invoices"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from agent_framework.openai import OpenAIChatCompletionClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env", override=True)
os.environ.pop("GOOGLE_API_KEY", None)

from rag.agent import get_invoice, list_vendors, search_invoices  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def build_openai_client() -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(
        model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        api_key=os.environ["OPENAI_API_KEY"],
        # Default base_url → api.openai.com
    )

def _message_text(message) -> str:
    text = getattr(message, "text", None)
    if text and str(text).strip():
        return str(text).strip()
    parts = []
    for content in getattr(message, "contents", None) or []:
        # Text / FunctionResult shapes from MAF
        for attr in ("text", "result", "output", "value"):
            val = getattr(content, attr, None)
            if val is None:
                continue
            if isinstance(val, (dict, list)):
                parts.append(json.dumps(val, indent=2, default=str)[:4000])
            else:
                s = str(val).strip()
                if s:
                    parts.append(s)
    return "\n".join(parts).strip()


def _response_text(response) -> str:
    if response is None:
        return ""
    text = getattr(response, "text", None)
    if text and str(text).strip():
        return str(text).strip()
    parts = []
    for message in getattr(response, "messages", None) or []:
        t = _message_text(message)
        if t:
            parts.append(t)
    return "\n\n".join(parts).strip()


def build_agents():
    chat = build_openai_client()

    retriever = chat.as_agent(
        name="RetrieverAgent",
        instructions=(
            "You retrieve Mansfield invoice evidence. "
            "ALWAYS use tools before answering. "
            "Prefer list_vendors for catalog questions, search_invoices for topical "
            "queries (pass vendor aliases jp/awg/best oil/hg when relevant), and "
            "get_invoice for exact invoice numbers. "
            "After tools return, write a compact factual briefing with invoice numbers, "
            "vendors, totals, and source paths. Never leave the final message empty."
        ),
        tools=[list_vendors, search_invoices, get_invoice],
    )

    analyst = chat.as_agent(
        name="AnalystAgent",
        instructions=(
            "You are a senior invoice analyst. Given retriever findings, write a clear "
            "final answer for the user. Cite InvoiceNumber + VendorName. "
            "If evidence is missing, say so. Do not invent numbers. "
            "Always produce a non-empty final answer."
        ),
    )
    return retriever, analyst


async def run_query(question: str) -> str:
    """
    Explicit sequential orchestration (Retriever → Analyst).

    Avoids SequentialBuilder's opaque handoff, which with Gemini OpenAI-compat
    often yields an empty AnalystAgent response.
    """
    retriever, analyst = build_agents()

    logging.info("MAF step 1/2: RetrieverAgent")
    try:
        retriever_response = await retriever.run(question)
    except Exception as e:
        logging.error("RetrieverAgent failed: %s", e)
        # Deterministic fallback without another LLM hop
        from rag.agent import ask_rag

        logging.info("Falling back to rag.ask_rag")
        result = ask_rag(question)
        return result.get("answer") or str(result)

    findings = _response_text(retriever_response)
    if not findings:
        try:
            findings = json.dumps(retriever_response.to_dict(), indent=2, default=str)[:6000]
        except Exception:
            findings = "(retriever returned no textual findings)"

    logging.info("MAF step 2/2: AnalystAgent")
    analyst_prompt = (
        f"User question:\n{question}\n\n"
        f"Retriever findings (use only this evidence):\n{findings}\n\n"
        "Write the final user-facing answer now."
    )
    try:
        analyst_response = await analyst.run(analyst_prompt)
    except Exception as e:
        logging.warning("AnalystAgent failed (%s); returning retriever findings", e)
        return findings

    answer = _response_text(analyst_response)
    if answer:
        return answer

    logging.warning("AnalystAgent returned empty text; falling back to retriever findings")
    return findings


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
