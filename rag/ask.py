"""Ask questions over the Mansfield invoice RAG index."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions

from rag.config import CHAT_MODEL, GEMINI_API_KEY, TOP_K
from rag.retrieve import retrieve

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def answer(question: str, top_k: int = TOP_K, vendor: str | None = None) -> str:
    hits = retrieve(question, top_k=top_k, vendor=vendor)
    if not hits:
        return "No matching invoices found."

    context_blocks = []
    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        context_blocks.append(
            f"[Source {i}: {meta.get('source', '?')} | vendor={meta.get('vendor')} | "
            f"invoice={meta.get('invoice_number')} | total={meta.get('total')}]\n"
            f"{hit['document']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are an invoice analytics assistant for Mansfield LTL fuel invoices.

Answer the question using ONLY the retrieved invoice context below.
If the answer is not in the context, say you do not have enough information.
Cite invoice numbers / source files when possible.

Question: {question}

Retrieved context:
{context}
"""

    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=HttpOptions(api_version="v1beta"),
    )
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config=GenerateContentConfig(temperature=0.2),
    )
    return (response.text or "").strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the Mansfield invoice RAG")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--vendor",
        default="",
        help="Vendor filter/alias: jp, awg, best oil, hg",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print retrieved chunks before the answer",
    )
    args = parser.parse_args(argv)

    question = args.question
    if not question:
        question = input("Question: ").strip()
    if not question:
        logging.error("No question provided")
        return 1

    vendor = args.vendor or None
    try:
        if args.show_sources:
            hits = retrieve(question, top_k=args.top_k, vendor=vendor)
            print("=== Retrieved sources ===")
            for i, hit in enumerate(hits, 1):
                meta = hit["metadata"]
                print(
                    f"{i}. {meta.get('source')} "
                    f"(vendor={meta.get('vendor')}, invoice={meta.get('invoice_number')}, "
                    f"dist={hit['distance']:.4f})"
                )
            print()

        print(answer(question, top_k=args.top_k, vendor=vendor))
        return 0
    except Exception as e:
        logging.error("%s", e)
        return 1


if __name__ == "__main__":
    if str(Path(__file__).resolve().parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(main())
