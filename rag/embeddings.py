"""Embedding helpers for Gemini and OpenAI providers."""

from __future__ import annotations

import logging
import time

from openai import OpenAI
from google import genai
from google.genai import errors as genai_errors
from google.genai.types import EmbedContentConfig, HttpOptions

from rag.config import (
    EMBED_MODEL,
    EMBED_PROVIDER,
    GEMINI_API_KEY,
    OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)


def embed_texts(texts: list[str], *, task: str = "document") -> list[list[float]]:
    """
    Embed a batch of texts.

    task: "document" for ingest, "query" for retrieval (Gemini task_type only).
    """
    if not texts:
        return []
    if EMBED_PROVIDER == "openai":
        return _embed_openai(texts)
    return _embed_gemini(texts, task=task)


def embed_query(text: str) -> list[float]:
    return embed_texts([text], task="query")[0]


def _embed_openai(texts: list[str]) -> list[list[float]]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required when RAG_EMBED_PROVIDER=openai")
    client = OpenAI(api_key=OPENAI_API_KEY)
    last_err: Exception | None = None
    for attempt in range(6):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            ordered = sorted(resp.data, key=lambda x: x.index)
            return [list(item.embedding) for item in ordered]
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "insufficient_quota" in msg or "credit" in msg:
                raise RuntimeError(
                    "OpenAI account has no credits for embeddings. "
                    "Add billing at https://platform.openai.com/settings/organization/billing/ "
                    f"(model={EMBED_MODEL}). Original error: {e}"
                ) from e
            if "rate" in msg or "429" in msg:
                wait = 20 * (attempt + 1)
                logger.warning("OpenAI embed rate limited; sleeping %ss", wait)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"OpenAI embedding failed after retries: {last_err}")


def _embed_gemini(texts: list[str], *, task: str) -> list[list[float]]:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is required when RAG_EMBED_PROVIDER=gemini")
    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=HttpOptions(api_version="v1beta"),
    )
    task_type = "RETRIEVAL_QUERY" if task == "query" else "RETRIEVAL_DOCUMENT"
    last_err: Exception | None = None
    for attempt in range(8):
        try:
            result = client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=EmbedContentConfig(task_type=task_type),
            )
            return [list(e.values) for e in result.embeddings]
        except genai_errors.ClientError as e:
            last_err = e
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                wait = 55 + (15 * attempt)
                logger.warning("Gemini embed rate limited; sleeping %ss", wait)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Gemini embedding failed after retries: {last_err}")
