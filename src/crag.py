"""
Post-retrieval: reranking, Corrective RAG, and diagram injection.
"""

from __future__ import annotations

import json
import re
from typing import Any

import requests
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from .config import (
    LLM_MODEL,
    API_KEY,
    API_BASE_URL,
    N_FILTER_DOCS,
    TOP_K,
    RERANK_API_ENDPOINT,
    RERANK_API_KEY,
    RERANK_MODEL,
    JINA_RERANK,
)

# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


def rerank_documents(
    query: str, docs: list[Document], top_k: int = TOP_K
) -> list[Document]:
    """Score all documents in a single LLM call for speed."""
    if not docs:
        return []

    if not N_FILTER_DOCS or len(docs) <= N_FILTER_DOCS:
        context_list = "\n".join(
            f"[{i + 1}] {doc.page_content[:500]}" for i, doc in enumerate(docs)
        )
    else:
        context_list = "\n".join(
            f"[{i + 1}] {doc.page_content[:500]}"
            for i, doc in enumerate(docs[:N_FILTER_DOCS])
        )

    prompt = (
        f"You are an H2 Economics expert. Given the query and the list of documents below, "
        f"identify the top {top_k} most relevant documents to answer the query.\n\n"
        f"Query: {query}\n\n"
        f"Documents:\n{context_list}\n\n"
        f"Respond with ONLY a comma-separated list of the document indices (e.g., 1, 3, 2). "
        f"Rank them from most to least relevant."
    )

    try:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=API_KEY,
            openai_api_base=API_BASE_URL,
            temperature=0,
        )
        resp = llm.invoke(prompt)
        # Extract indices
        indices = [int(n) for n in re.findall(r"\d+", resp.content)]

        reranked = []
        seen = set()
        for idx in indices:
            # Shift back to 0-indexed and bounds check
            real_idx = idx - 1
            if 0 <= real_idx < len(docs) and real_idx not in seen:
                reranked.append(docs[real_idx])
                seen.add(real_idx)

        # If LLM failed to provide enough indices, fill with original order
        if len(reranked) < top_k:
            for i, doc in enumerate(docs):
                if i not in seen:
                    reranked.append(doc)
                    seen.add(i)
                if len(reranked) >= top_k:
                    break

        return reranked[:top_k]
    except Exception:
        # Fallback to original order
        return docs[:top_k]


def rerank_documents_jina(
    query: str, docs: list[Document], top_k: int = TOP_K
) -> list[Document]:
    """Rerank documents using Jina Reranker API."""
    if not docs:
        return []

    if not RERANK_API_KEY:
        return docs[:top_k]

    documents = [doc.page_content for doc in docs]

    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": min(top_k, len(docs)),
        "return_documents": False,
    }

    headers = {
        "Authorization": f"Bearer {RERANK_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    try:
        response = requests.post(
            RERANK_API_ENDPOINT, json=payload, headers=headers, timeout=30
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        reranked = [docs[r["index"]] for r in results if 0 <= r["index"] < len(docs)]
        return reranked[:top_k]
    except Exception:
        return docs[:top_k]


def rerank(query: str, docs: list[Document], top_k: int = TOP_K) -> list[Document]:
    """Rerank documents using configured reranker (Jina or LLM)."""
    if JINA_RERANK:
        return rerank_documents_jina(query, docs, top_k=top_k)
    return rerank_documents(query, docs, top_k=top_k)


# ---------------------------------------------------------------------------
# DuckDuckGo diagram search
# ---------------------------------------------------------------------------

_DDG_URL = "https://api.duckduckgo.com/"


def search_diagram(topic: str) -> str | None:
    """Search DuckDuckGo for an economics diagram URL.

    Returns the best image/source URL found, or None.
    """
    search_query = (
        f"H2 Economics {topic} diagram site:wikimedia.org OR site:economicshelp.org"
    )
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.images(search_query, max_results=3))
            if results:
                return results[0].get("image") or results[0].get("url")
            # Fallback to text search for a link
            text_results = list(ddgs.text(search_query, max_results=3))
            if text_results:
                return text_results[0].get("href")
    except Exception:
        pass
    return None
