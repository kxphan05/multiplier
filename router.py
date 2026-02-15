"""
Semantic router – classifies user queries into RELATIONAL / VECTOR / DIRECT.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from config import H2_ECON_TOPICS, LLM_MODEL, SCHOOLS, API_KEY, API_BASE_URL

_ROUTER_PROMPT = """\
You are a query classifier for an H2 Economics study assistant.

Given the user's query, classify it into exactly ONE of these categories:

- VECTOR: This is the DEFAULT category for most questions. If the user is asking for an explanation, definition, comparison, essay, or diagram (e.g., "explain...", "what is...", "compare..."), it MUST be VECTOR. This includes specific economic concepts like "the impossible trinity", "multiplier effect", "market failure", etc.
- RELATIONAL: Only select this if the user is explicitly asking for files, notes, or papers from a library. This usually involves mentioning a school (e.g. RI, HCI, NYJC) AND asking for "notes", "papers", or "resources". Example: "Show me RI trade notes". 
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
    
    # Primary Conceptual Indicators (VECTOR)
    # If they ask to EXPLAIN or WHAT IS, it's almost always conceptual (VECTOR)
    if any(kw in query_upper for kw in ["EXPLAIN ", "WHAT IS ", "DEFINE ", "DESCRIBE "]):
        return "VECTOR"

    # 2. LLM Fallback (Let the model decide for complex cases)
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
    for cat in ("VECTOR", "RELATIONAL", "DIRECT"):
        if cat in text:
            return cat
    return "DIRECT"  # fallback
