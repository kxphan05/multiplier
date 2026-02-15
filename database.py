"""
Database helpers for SQLite and Chroma.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import chromadb
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from config import (
    CHROMA_COLLECTION,
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    SQLITE_DB_PATH,
)

# ---------------------------------------------------------------------------
# Project root – all relative paths resolve from here
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------
def run_sql_query(query: str) -> list[dict[str, Any]]:
    """Execute a read-only SQL query and return rows as dicts."""
    db_path = _ROOT / SQLITE_DB_PATH
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(query)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Chroma helpers
# ---------------------------------------------------------------------------
_chroma_client: chromadb.ClientAPI | None = None
_embeddings: OllamaEmbeddings | None = None


def _get_chroma_collection() -> chromadb.Collection:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(_ROOT / CHROMA_DB_PATH))
    return _chroma_client.get_collection(CHROMA_COLLECTION)


def _get_embeddings() -> OllamaEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return _embeddings


def query_chroma(query: str, n_results: int = 6) -> list[Document]:
    """Embed *query* and retrieve the top-n Chroma documents."""
    col = _get_chroma_collection()
    emb = _get_embeddings()
    query_vec = emb.embed_query(query)
    results = col.query(query_embeddings=[query_vec], n_results=n_results)
    docs: list[Document] = []
    for i, doc_text in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i] if results["metadatas"] else {}
        distance = results["distances"][0][i] if results["distances"] else None
        meta["distance"] = distance
        docs.append(Document(page_content=doc_text, metadata=meta))
    return docs


def query_chroma_multi(queries: list[str], n_results: int = 4) -> list[Document]:
    """Run multiple queries against Chroma, deduplicate by content hash."""
    seen: set[int] = set()
    docs: list[Document] = []
    for q in queries:
        for doc in query_chroma(q, n_results=n_results):
            h = hash(doc.page_content)
            if h not in seen:
                seen.add(h)
                docs.append(doc)
    return docs
