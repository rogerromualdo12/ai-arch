"""Ingest Mansfield invoice JSONs into a local Chroma vector store."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import chromadb

from rag.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    EMBED_PROVIDER,
    RAG_DATA_DIR,
)
from rag.documents import load_invoice_documents
from rag.embeddings import embed_texts

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def ingest(reset: bool = True) -> int:
    if not RAG_DATA_DIR.exists():
        raise FileNotFoundError(f"RAG_DATA_DIR not found: {RAG_DATA_DIR}")

    docs = load_invoice_documents(RAG_DATA_DIR)
    if not docs:
        raise RuntimeError(f"No invoice JSON found under {RAG_DATA_DIR}")

    logging.info(
        "Loaded %s invoice documents from %s (provider=%s model=%s collection=%s)",
        len(docs),
        RAG_DATA_DIR,
        EMBED_PROVIDER,
        EMBED_MODEL,
        COLLECTION_NAME,
    )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            chroma.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    collection = chroma.get_or_create_collection(name=COLLECTION_NAME)

    # Smaller batches to stay under free-tier / low TPM embedding limits.
    batch_size = 20 if EMBED_PROVIDER == "openai" else 25
    total = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        texts = [d["text"] for d in batch]
        ids = [d["id"] for d in batch]
        metadatas = [d["metadata"] for d in batch]
        embeddings = embed_texts(texts, task="document")
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
        time.sleep(1.5 if EMBED_PROVIDER == "openai" else 1)

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
        print(
            f"Ingested {count} documents into {CHROMA_DIR} "
            f"(collection={COLLECTION_NAME}, provider={EMBED_PROVIDER})"
        )
        return 0
    except Exception as e:
        logging.error("%s", e)
        return 1


if __name__ == "__main__":
    if str(Path(__file__).resolve().parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(main())
