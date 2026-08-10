"""Ask questions over the Mansfield invoice RAG index."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rag.agent import ask_rag
from rag.config import COLLECTION_NAME, EMBED_PROVIDER, TOP_K
from rag.retrieve import retrieve

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


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

    vendor = args.vendor or ""
    try:
        logging.info(
            "RAG ask provider=%s collection=%s", EMBED_PROVIDER, COLLECTION_NAME
        )
        if args.show_sources:
            hits = retrieve(question, top_k=args.top_k, vendor=vendor or None)
            print("=== Retrieved sources ===")
            for i, hit in enumerate(hits, 1):
                meta = hit["metadata"]
                print(
                    f"{i}. {meta.get('source')} "
                    f"(vendor={meta.get('vendor')}, invoice={meta.get('invoice_number')}, "
                    f"dist={hit['distance']:.4f})"
                )
            print()

        result = ask_rag(question, vendor=vendor, top_k=args.top_k)
        print(result.get("answer") or result)
        return 0 if result.get("ok") else 1
    except Exception as e:
        logging.error("%s", e)
        return 1


if __name__ == "__main__":
    if str(Path(__file__).resolve().parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(main())
