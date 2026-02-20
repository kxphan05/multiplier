"""
Semantic router – classifies user queries into RELATIONAL / VECTOR / DIRECT.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from .config import H2_ECON_TOPICS, LLM_MODEL, SCHOOLS, API_KEY, API_BASE_URL

_ROUTER_PROMPT = """\
### Role
You are the Technical Router for an H2 Economics AI Tutor. Your job is to analyze the user's intent and route it to the correct retrieval path.

### Categories
- **VECTOR**: Use this for ANY query involving economic theory, concepts, definitions, or syllabus-specific analysis. If the user mentions "Multiplier", "MPS", "MAS", "Opportunity Cost", or "Elasticity", it MUST be VECTOR.
- **RELATIONAL**: Use ONLY if the user asks for specific administrative resources, file names, or school-specific collections (e.g., "RI 2024 Prelim Paper", "HCI notes on trade").
- **DIRECT**: Use ONLY for non-economic chatter, greetings ("hi", "who are you"), or meta-comments about the bot itself.

### Syllabus Keywords (Always VECTOR)
MPS, MPC, MPM, Multiplier, AD/AS, Market Failure, Externalities, Merit Goods, Exchange Rate Policy, Fiscal Policy, Comparative Advantage.

### Examples
- "Hi there!" -> DIRECT
- "Show me VJC notes for macro." -> RELATIONAL
- "How does a high MPS affect the multiplier?" -> VECTOR
- "Explain the MAS policy." -> VECTOR

### Response Format
Respond with ONLY the single word: VECTOR, RELATIONAL, or DIRECT.
"""


def route_query(query: str) -> str:
    """Return one of 'RELATIONAL', 'VECTOR', or 'DIRECT'."""
    query_upper = query.upper()

    # 1. Quick Short-circuits (No LLM)
    # Greetings/Direct
    if any(
        greet in query_upper
        for greet in ["HELLO", "HEY", "THANKS", "THANK YOU", "BYE"]
    ):
        return "DIRECT"

    # Primary Conceptual Indicators (VECTOR)
    # If they ask to EXPLAIN or WHAT IS, it's almost always conceptual (VECTOR)
    if any(
        kw in query_upper for kw in ["EXPLAIN ", "WHAT IS ", "DEFINE ", "DESCRIBE ", "HOW DOES ", "WHY DOES "]
    ):
        return "VECTOR"

    # 2. LLM Fallback (Let the model decide for complex cases)
    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=API_KEY,
        openai_api_base=API_BASE_URL,
        temperature=0.1,
    )

    # Construct prompt with explicit roles for Qwen3
    system_instruction = _ROUTER_PROMPT.format(
        schools=", ".join(SCHOOLS), topics="; ".join(H2_ECON_TOPICS)
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
