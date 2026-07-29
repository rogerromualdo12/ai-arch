"""RAG config for Mansfield LTL invoice JSON corpus."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)
os.environ.pop("GOOGLE_API_KEY", None)

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
CHAT_MODEL = os.getenv("GEMINI_RAG_MODEL", "gemini-2.5-flash")

# Source of invoice JSON extracts (Mansfield LTL)
RAG_DATA_DIR = Path(
    os.getenv(
        "RAG_DATA_DIR",
        "/Users/rogerromualdojavier/Documents/Mansfield/LTL/Base prompt/output_files",
    )
).expanduser()

# Local Chroma persistence
CHROMA_DIR = Path(os.getenv("RAG_CHROMA_DIR", str(BASE_DIR / "rag_store")))
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "mansfield_invoices")

TOP_K = int(os.getenv("RAG_TOP_K", "5"))
