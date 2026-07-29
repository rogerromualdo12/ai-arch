"""Shared retrieval helpers for Mansfield invoice RAG."""

from __future__ import annotations

from typing import Any

import chromadb
from google import genai
from google.genai.types import EmbedContentConfig, HttpOptions

from rag.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    GEMINI_API_KEY,
    TOP_K,
)

# Friendly aliases → exact VendorName values stored in metadata
VENDOR_ALIASES: dict[str, list[str]] = {
    "jp": ["JP Fuels"],
    "jp fuels": ["JP Fuels"],
    "jp fuel": ["JP Fuels"],
    "awg": ["AMERICAN WELDING & GAS INC", "AMERICAN WELDING & GAS"],
    "american welding": ["AMERICAN WELDING & GAS INC", "AMERICAN WELDING & GAS"],
    "best oil": ["Best Oil Inc"],
    "best oil inc": ["Best Oil Inc"],
    "hg": ["HOLSTON GASES", "HOLSTON GAS"],
    "holston": ["HOLSTON GASES", "HOLSTON GAS"],
    "holston gases": ["HOLSTON GASES", "HOLSTON GAS"],
}


def get_collection():
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma.get_or_create_collection(name=COLLECTION_NAME)
    if collection.count() == 0:
        raise RuntimeError(
            f"Empty collection '{COLLECTION_NAME}'. Run: python -m rag.ingest"
        )
    return collection


def resolve_vendors(vendor: str | None) -> list[str] | None:
    if not vendor or not vendor.strip():
        return None
    key = vendor.strip().lower()
    if key in VENDOR_ALIASES:
        return VENDOR_ALIASES[key]
    # Exact / case-insensitive match against known vendors in the index
    known = {v.lower(): v for vs in VENDOR_ALIASES.values() for v in vs}
    if key in known:
        return [known[key]]
    return [vendor.strip()]


def embed_query(text: str) -> list[float]:
    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=HttpOptions(api_version="v1beta"),
    )
    q_emb = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return list(q_emb.embeddings[0].values)


def retrieve(
    question: str,
    top_k: int = TOP_K,
    vendor: str | None = None,
) -> list[dict[str, Any]]:
    """Semantic search with optional vendor filter."""
    collection = get_collection()
    query_vec = embed_query(question)
    n = min(max(top_k, 1), collection.count())

    vendors = resolve_vendors(vendor)
    where = None
    if vendors:
        if len(vendors) == 1:
            where = {"vendor": vendors[0]}
        else:
            where = {"vendor": {"$in": vendors}}

    kwargs: dict[str, Any] = {
        "query_embeddings": [query_vec],
        "n_results": n,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    try:
        result = collection.query(**kwargs)
    except Exception:
        # Fallback if filter matches nothing / chroma where quirks
        if not where:
            raise
        result = collection.query(
            query_embeddings=[query_vec],
            n_results=min(n * 5, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        # Post-filter
        filtered = []
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        wanted = {v.lower() for v in vendors}
        for doc, meta, dist in zip(docs, metas, dists):
            meta = meta or {}
            if str(meta.get("vendor", "")).lower() in wanted:
                filtered.append({"document": doc, "metadata": meta, "distance": dist})
            if len(filtered) >= n:
                break
        return filtered

    hits: list[dict[str, Any]] = []
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append({"document": doc, "metadata": meta or {}, "distance": dist})
    return hits


def list_vendor_stats() -> dict[str, Any]:
    collection = get_collection()
    # Pull metadata in pages
    counts: dict[str, int] = {}
    offset = 0
    page = 200
    total = collection.count()
    while offset < total:
        batch = collection.get(
            include=["metadatas"],
            limit=min(page, total - offset),
            offset=offset,
        )
        for meta in batch.get("metadatas") or []:
            vendor = (meta or {}).get("vendor") or "(unknown)"
            counts[vendor] = counts.get(vendor, 0) + 1
        offset += page
    return {
        "total_documents": total,
        "vendors": [
            {"vendor": v, "count": c}
            for v, c in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        ],
        "aliases": {
            "jp / jp fuels": "JP Fuels",
            "awg": "AMERICAN WELDING & GAS INC",
            "best oil": "Best Oil Inc",
            "hg / holston": "HOLSTON GASES",
        },
    }
