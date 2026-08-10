"""RAG config for Mansfield LTL invoice JSON corpus."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)
os.environ.pop("GOOGLE_API_KEY", None)

# Embedding / chat provider: "openai" | "gemini"
EMBED_PROVIDER = os.getenv("RAG_EMBED_PROVIDER", "openai").strip().lower()
CHAT_PROVIDER = os.getenv("RAG_CHAT_PROVIDER", EMBED_PROVIDER).strip().lower()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

GEMINI_CHAT_MODEL = os.getenv("GEMINI_RAG_MODEL", "gemini-2.5-flash")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

EMBED_MODEL = OPENAI_EMBED_MODEL if EMBED_PROVIDER == "openai" else GEMINI_EMBED_MODEL
CHAT_MODEL = OPENAI_CHAT_MODEL if CHAT_PROVIDER == "openai" else GEMINI_CHAT_MODEL

# Source of invoice JSON extracts (Mansfield LTL)
RAG_DATA_DIR = Path(
    os.getenv(
        "RAG_DATA_DIR",
        "/Users/rogerromualdojavier/Documents/Mansfield/LTL/Base prompt/output_files",
    )
).expanduser()

# Local Chroma persistence — keep Gemini + OpenAI collections side by side
CHROMA_DIR = Path(os.getenv("RAG_CHROMA_DIR", str(BASE_DIR / "rag_store")))
_default_collection = (
    "mansfield_invoices_openai"
    if EMBED_PROVIDER == "openai"
    else "mansfield_invoices"
)
COLLECTION_NAME = os.getenv("RAG_COLLECTION", _default_collection)

TOP_K = int(os.getenv("RAG_TOP_K", "5"))
