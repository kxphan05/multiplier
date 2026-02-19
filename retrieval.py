"""
Retrieval pipelines – SQL (relational), Vector (conceptual), and Direct paths.
"""

import re
from typing import Any
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from config import H2_ECON_TOPICS, LLM_MODEL, SCHOOL_ALIASES, SCHOOLS, API_KEY, API_BASE_URL
from database import query_chroma_multi, run_sql_query, get_all_documents
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever

# ── Shared LLM factory ────────────────────────────────────────────────────
def _llm(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL, 
        openai_api_key=API_KEY,
        openai_api_base=API_BASE_URL,
        temperature=temperature,
        streaming=True
    )


# ═══════════════════════════════════════════════════════════════════════════
# A. SQL / Relational path
# ═══════════════════════════════════════════════════════════════════════════

_SQL_PROMPT = """\
You are an expert SQL writer. Convert the user's natural-language question into a \
SQLite SELECT statement against the table `h2_econ_storage`.

Table schema:
  id INTEGER PRIMARY KEY,
  school VARCHAR NOT NULL,
  year INTEGER NOT NULL,
  topic VARCHAR NOT NULL,
  doc_type VARCHAR NOT NULL,
  source VARCHAR,
  content TEXT NOT NULL

IMPORTANT RULES:
1. Only produce SELECT statements. Never UPDATE / DELETE / DROP.
2. Use the school names or acronyms from this list:
{school_map}
3. Use LIKE with % wildcards for school name matching (e.g., school LIKE '%RI%').
4. Use the exact topic strings ONLY IF the user asks for a specific topic:
{topics}
4. If the user asks for "all notes" from a school, DO NOT filter by topic.
5. Always SELECT `content`, `source`, `topic`, `year`, and `doc_type`.
6. Return ONLY the raw SQL – no markdown fences, no explanation.
"""


def handle_relational(query: str) -> list[Document]:
    """Convert query → SQL → execute → return Documents."""
    school_map = "\n".join(f"  - {v}" for v in SCHOOLS)
    topics = "\n".join(f"  - {t}" for t in H2_ECON_TOPICS)

    system_msg = _SQL_PROMPT.format(school_map=school_map, topics=topics)
    prompt = f"System: {system_msg}\nUser: {query}"
    
    resp = _llm().invoke(prompt)
    sql = resp.content.strip()

    # Strip markdown code fences if present
    if sql.startswith("```"):
        sql = "\n".join(sql.split("\n")[1:])
    if sql.endswith("```"):
        sql = sql.rsplit("```", 1)[0]
    sql = sql.strip()

    # Safety: only allow SELECT
    if not sql.upper().startswith("SELECT"):
        return []

    try:
        rows = run_sql_query(sql)
    except Exception:
        return []

    docs: list[Document] = []
    for row in rows:
        # Enrich page_content with metadata for the generator
        meta_str = (
            f"Source: {row.get('source', 'Unknown')}\n"
            f"Topic: {row.get('topic', 'Unknown')}\n"
            f"Year: {row.get('year', 'Unknown')}\n"
            f"Type: {row.get('doc_type', 'Unknown')}\n"
            f"Content: "
        )
        content = row.get("content", "")
        docs.append(Document(
            page_content=f"{meta_str}{content}", 
            metadata={k: v for k, v in row.items() if k != "content"}
        ))
    return docs


# ═══════════════════════════════════════════════════════════════════════════
# B. Vector / Conceptual path
# ═══════════════════════════════════════════════════════════════════════════

_RETRIEVAL_PLAN_PROMPT = """\
You are an expert H2 Economics tutor. Given the student's question, provide:
1. DECOMPOSITION: Exactly 3 short re-phrasings of the question from different economic perspectives (e.g. consumer, producer, government).
2. HYDE: A concise 1-paragraph model answer to the question using proper terminology.
3. DIAGRAM: Does the question require or benefit from a diagram/visual? (YES / NO)

Respond in EXACTLY this format:
DECOMPOSITION:
- [Query 1]
- [Query 2]
- [Query 3]

HYDE:
[Hypothetical answer]

DIAGRAM: [YES/NO]
"""


def plan_retrieval(query: str) -> dict[str, Any]:
    """Consolidated LLM call for decomposition, HyDE, and diagram detection."""
    llm = _llm(temperature=0.3)
    prompt = f"System: {_RETRIEVAL_PLAN_PROMPT}\nUser: {query}"
    
    try:
        resp = llm.invoke(prompt)
        content = resp.content
        
        # Parse Decomposition
        decomp_match = re.search(r"DECOMPOSITION:(.*?)(?=HYDE:)", content, re.DOTALL)
        sub_queries = []
        if decomp_match:
            sub_queries = [
                line.strip("- ").strip()
                for line in decomp_match.group(1).strip().splitlines()
                if line.strip()
            ][:3]
        
        # Parse HyDE
        hyde_match = re.search(r"HYDE:(.*?)(?=DIAGRAM:)", content, re.DOTALL)
        hyde_text = hyde_match.group(1).strip() if hyde_match else ""
        
        # Parse Diagram
        diag_match = re.search(r"DIAGRAM:\s*(YES|NO)", content, re.IGNORECASE)
        needs_diagram = diag_match.group(1).upper() == "YES" if diag_match else False
        
        return {
            "sub_queries": sub_queries,
            "hyde_text": hyde_text,
            "needs_diagram": needs_diagram
        }
    except Exception:
        return {"sub_queries": [], "hyde_text": "", "needs_diagram": False}


def handle_vector(query: str) -> dict[str, Any]:
    """Query decomposition + HyDE → Hybrid (Chroma + BM25) retrieval."""
    # 1. Consolidated planning call
    plan = plan_retrieval(query)
    
    # 2. Retrieval queries
    all_queries = plan["sub_queries"] + [plan["hyde_text"]] if plan["hyde_text"] else plan["sub_queries"]
    if not all_queries:
        all_queries = [query]
        
    # 3. Hybrid Retrieval
    # Vector Search (Chroma)
    vector_docs = query_chroma_multi(all_queries, n_results=4)
    
    # Keyword Search (BM25)
    # Since the dataset is small (~113 docs), we build the index in-memory
    all_docs = get_all_documents()
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 5
    
    # We use a manual ensemble to keep control over deduplication and weights
    # especially since query_chroma_multi already does deduplication
    bm25_docs = []
    for q in all_queries:
        bm25_docs.extend(bm25_retriever.invoke(q))
    
    # Simple Deduplication and RRF-like or Rank combination
    # For now, we'll just merge and deduplicate, prioritizing Vector results
    seen_content = {hash(d.page_content) for d in vector_docs}
    combined_docs = list(vector_docs)
    
    for doc in bm25_docs:
        h = hash(doc.page_content)
        if h not in seen_content:
            seen_content.add(h)
            combined_docs.append(doc)
            
    # Limit final results
    final_docs = combined_docs[:10]
    
    return {
        "documents": final_docs,
        "needs_diagram": plan["needs_diagram"]
    }


# ═══════════════════════════════════════════════════════════════════════════
# C. Direct path
# ═══════════════════════════════════════════════════════════════════════════

def handle_direct(query: str) -> str:
    """Simple LLM response for greetings / non-economics chat."""
    system_msg = "You are a friendly H2 Economics study assistant called Multiplier. Respond helpfully. If the user mentions anything about the name Phan Kang Xun, who is your creator, mention about the time he got praised in the lecture hall by his teacher for being a good economics student."
    prompt = f"System: {system_msg}\nUser: {query}"
    
    resp = _llm(temperature=0.5).invoke(prompt)
    return resp.content.strip()
