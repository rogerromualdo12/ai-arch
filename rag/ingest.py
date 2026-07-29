"""Ingest Mansfield invoice JSONs into a local Chroma vector store."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import chromadb
from google import genai
from google.genai import errors as genai_errors
from google.genai.types import EmbedContentConfig, HttpOptions

from rag.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    GEMINI_API_KEY,
    RAG_DATA_DIR,
)
from rag.documents import load_invoice_documents

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def embed_texts(client: genai.Client, texts: list[str]) -> list[list[float]]:
    """Embed texts with Gemini using a true batch call + retry on 429."""
    last_err: Exception | None = None
    for attempt in range(8):
        try:
            result = client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            return [list(e.values) for e in result.embeddings]
        except genai_errors.ClientError as e:
            last_err = e
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                wait = 55 + (15 * attempt)
                logging.warning(
                    "Rate limited; sleeping %ss (attempt %s)", wait, attempt + 1
                )
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Embedding failed after retries: {last_err}")


def ingest(reset: bool = True) -> int:
    if not RAG_DATA_DIR.exists():
        raise FileNotFoundError(f"RAG_DATA_DIR not found: {RAG_DATA_DIR}")

    docs = load_invoice_documents(RAG_DATA_DIR)
    if not docs:
        raise RuntimeError(f"No invoice JSON found under {RAG_DATA_DIR}")

    logging.info("Loaded %s invoice documents from %s", len(docs), RAG_DATA_DIR)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            chroma.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    collection = chroma.get_or_create_collection(name=COLLECTION_NAME)

    genai_client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=HttpOptions(api_version="v1beta"),
    )

    # Free tier ~100 embed requests/min; one batch request covers many docs.
    batch_size = 25
    total = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        texts = [d["text"] for d in batch]
        ids = [d["id"] for d in batch]
        metadatas = [d["metadata"] for d in batch]
        embeddings = embed_texts(genai_client, texts)
        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"Embedding count mismatch: got {len(embeddings)} for {len(batch)} docs"
            )
        collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        total += len(batch)
        logging.info("Indexed %s / %s", total, len(docs))
        time.sleep(1)

    logging.info(
        "Done. Collection=%s count=%s store=%s",
        COLLECTION_NAME,
        collection.count(),
        CHROMA_DIR,
    )
    return collection.count()


def main() -> int:
    reset = "--no-reset" not in sys.argv
    try:
        count = ingest(reset=reset)
        print(f"Ingested {count} documents into {CHROMA_DIR}")
        return 0
    except Exception as e:
        logging.error("%s", e)
        return 1


if __name__ == "__main__":
    if str(Path(__file__).resolve().parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(main())
