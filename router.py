"""
Semantic router – classifies user queries into RELATIONAL / VECTOR / DIRECT.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from config import H2_ECON_TOPICS, LLM_MODEL, SCHOOLS, API_KEY, API_BASE_URL

_ROUTER_PROMPT = """\
You are a query classifier for an H2 Economics study assistant.

Given the user's query, classify it into exactly ONE of these categories:
- RELATIONAL: The query mentions a specific junior college / school name or acronym (e.g. RI, HCI, NYJC), or asks for NOTES from a specific year or about a specific topic. Schools: {schools} Topics: {topics}
- VECTOR: The query is a conceptual economics question (e.g. definitions, explanations, comparisons, essays, diagrams). ANY question containing economic terms like "multiplier", "market failure", "elasticity", "demand", "supply", "policy", etc. MUST be VECTOR.
- DIRECT: The query is a general greeting (e.g. "hi", "hello"), off-topic chat, or does NOT relate to economics at all.

Respond with ONLY the single word: RELATIONAL, VECTOR, or DIRECT. Nothing else.
"""


def route_query(query: str) -> str:
    """Return one of 'RELATIONAL', 'VECTOR', or 'DIRECT'."""
    query_upper = query.upper()

    # 1. Quick Short-circuits (No LLM)
    # Greetings/Direct
    if any(greet in query_upper for greet in ["HI", "HELLO", "HEY", "THANKS", "THANK YOU", "BYE"]):
        return "DIRECT"
    
    # JC Names (Relational)
    from config import SCHOOL_ALIASES
    if any(alias in query_upper for alias in SCHOOL_ALIASES.keys()):
        return "RELATIONAL"

    # 2. LLM Fallback
    llm = ChatOpenAI(
        model=LLM_MODEL, 
        openai_api_key=API_KEY,
        openai_api_base=API_BASE_URL,
        temperature=0.1
    )
    
    # Construct prompt with explicit roles for Qwen3
    system_instruction = _ROUTER_PROMPT.format(
        schools=", ".join(SCHOOLS),
        topics="; ".join(H2_ECON_TOPICS)
    )
    prompt = f"System: {system_instruction}\nUser: {query}"
    
    try:
        response = llm.invoke(prompt)
        text = response.content.strip().upper()
    except Exception:
        return "DIRECT"

    # Extract the first valid category found in the response
    for cat in ("RELATIONAL", "VECTOR", "DIRECT"):
        if cat in text:
            return cat
    return "DIRECT"  # fallback
