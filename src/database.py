"""
Database helpers for SQLite and Chroma.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import chromadb
from langchain_core.documents import Document
from .config import CHROMA_COLLECTION, CHROMA_DB_PATH, SQLITE_DB_PATH
from chromadb.utils import embedding_functions

_ROOT = Path(__file__).resolve().parent.parent


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
_embedding_fn = None


def _get_chroma_collection() -> chromadb.Collection:
    global _chroma_client, _embedding_fn
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(_ROOT / CHROMA_DB_PATH))
    if _embedding_fn is None:
        _embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    return _chroma_client.get_collection(
        CHROMA_COLLECTION, embedding_function=_embedding_fn
    )


def query_chroma(query: str, n_results: int = 6) -> list[Document]:
    """Retrieve top-n documents from Chroma using its native embedding function."""
    col = _get_chroma_collection()
    # Chroma handles embedding automatically when query_texts is used
    results = col.query(query_texts=[query], n_results=n_results)

    docs: list[Document] = []
    if not results or not results["documents"]:
        return docs

    for i, doc_text in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i] if results["metadatas"] else {}
        distance = (
            results["distances"][0][i]
            if "distances" in results and results["distances"]
            else None
        )
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


def get_all_documents() -> list[Document]:
    """Fetch all rows from SQLite and convert to LangChain Documents."""
    rows = run_sql_query(
        "SELECT content, source, topic, year, doc_type FROM h2_econ_storage"
    )
    docs: list[Document] = []
    for row in rows:
        meta_str = (
            f"Source: {row.get('source', 'Unknown')}\n"
            f"Topic: {row.get('topic', 'Unknown')}\n"
            f"Year: {row.get('year', 'Unknown')}\n"
            f"Type: {row.get('doc_type', 'Unknown')}\n"
            f"Content: "
        )
        content = row.get("content", "")
        docs.append(
            Document(
                page_content=f"{meta_str}{content}",
                metadata={k: v for k, v in row.items() if k != "content"},
            )
        )
    return docs
