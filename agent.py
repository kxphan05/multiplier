"""
LangGraph agent – orchestrates routing, retrieval, reranking, CRAG, and
final answer generation with streaming support.
"""

from __future__ import annotations

import operator
import logging
from typing import Annotated, Any, Iterator, TypedDict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from config import LLM_MODEL, API_KEY, API_BASE_URL
from crag import rerank_documents, search_diagram
from retrieval import handle_direct, handle_relational, handle_vector
from router import route_query


# ---------------------------------------------------------------------------
# State schema – TypedDict so LangGraph merges keys properly
# ---------------------------------------------------------------------------

def _replace(a: Any, b: Any) -> Any:
    """Reducer: always take the newer value."""
    return b if b is not None else a


class AgentState(TypedDict, total=False):
    query: Annotated[str, _replace]
    route: Annotated[str, _replace]
    documents: Annotated[list[Document], _replace]
    reranked_docs: Annotated[list[Document], _replace]
    needs_diagram: Annotated[bool, _replace]
    diagram_url: Annotated[str | None, _replace]
    response: Annotated[str, _replace]


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def node_route(state: AgentState) -> dict[str, Any]:
    logger.info(f"Routing query: {state['query']}")
    route = route_query(state["query"])
    logger.info(f"Route selected: {route}")
    return {"route": route}


def node_relational(state: AgentState) -> dict[str, Any]:
    logger.info("Executing relational retrieval...")
    docs = handle_relational(state["query"])
    logger.info(f"Relational retrieval found {len(docs)} docs")
    return {"documents": docs}


def node_vector(state: AgentState) -> dict[str, Any]:
    logger.info("Executing vector retrieval...")
    result = handle_vector(state["query"])
    docs = result["documents"]
    logger.info(f"Vector retrieval found {len(docs)} docs")
    return {
        "documents": docs,
        "needs_diagram": result["needs_diagram"]
    }


def node_direct(state: AgentState) -> dict[str, Any]:
    logger.info("Handling direct query...")
    response = handle_direct(state["query"])
    return {"response": response}


def node_rerank(state: AgentState) -> dict[str, Any]:
    docs = state.get("documents", [])
    logger.info(f"Reranking {len(docs)} documents...")
    reranked = rerank_documents(state["query"], docs, top_k=5)
    logger.info(f"Reranking complete. Top {len(reranked)} retained.")
    return {"reranked_docs": reranked}


# node_crag removed - sufficiency and diagram logic moved to retrieval planning and generation


def node_generate(state: AgentState) -> dict[str, Any]:
    """Build context and generate the final answer."""
    route = state.get("route", "VECTOR")
    docs = state.get("reranked_docs", state.get("documents", []))
    
    # Use more context for RELATIONAL queries (likely lists)
    limit = 15 if route == "RELATIONAL" else 5
    context = "\n\n---\n\n".join(d.page_content for d in docs[:limit])
    
    diagram_url = state.get("diagram_url")
    diagram_instruction = ""
    if diagram_url:
        diagram_instruction = (
            f"\n\nA relevant diagram has been found. Include this image in your "
            f"answer using Markdown:\n![Economics Diagram]({diagram_url})"
        )

    if route == "RELATIONAL":
        system_msg = (
            "You are an H2 Economics library assistant. The user is asking for specific "
            "resources (notes, papers, etc.) from the database. Format your response "
            "clearly, using a table or a structured list. Include Source, Topic, and Year "
            "for each item. If no specific items are found, inform the user."
        )
    else:
        system_msg = (
            "You are an expert H2 Economics tutor. Using ONLY the context below, "
            "provide a comprehensive, well-structured answer to the student's question. "
            "Use proper economics terminology and structure your answer with headers "
            "where appropriate. If the context is insufficient, say so honestly."
        )

    user_msg = (
        f"Context:\n{context}\n{diagram_instruction}\n\n"
        f"Student question: {state['query']}"
    )
    prompt = f"System: {system_msg}\nUser: {user_msg}"

    llm = ChatOpenAI(
        model=LLM_MODEL, 
        openai_api_key=API_KEY,
        openai_api_base=API_BASE_URL,
        temperature=0.3
    )
    resp = llm.invoke(prompt)
    return {"response": resp.content}


# ---------------------------------------------------------------------------
# Routing condition
# ---------------------------------------------------------------------------

def route_condition(state: AgentState) -> str:
    return state.get("route", "DIRECT")


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph(include_generate: bool = True) -> StateGraph:
    g = StateGraph(AgentState)

    # Nodes
    g.add_node("route", node_route)
    g.add_node("relational", node_relational)
    g.add_node("vector", node_vector)
    g.add_node("direct", node_direct)
    g.add_node("rerank", node_rerank)
    
    if include_generate:
        g.add_node("generate", node_generate)

    # Edges
    g.set_entry_point("route")
    g.add_conditional_edges(
        "route",
        route_condition,
        {
            "RELATIONAL": "relational",
            "VECTOR": "vector",
            "DIRECT": "direct",
        },
    )

    # Relational / Vector → rerank
    g.add_edge("relational", "rerank")
    g.add_edge("vector", "rerank")
    
    if include_generate:
        g.add_edge("rerank", "generate")
        g.add_edge("generate", END)
    else:
        # If we skip generation (for streaming), end at rerank
        g.add_edge("rerank", END)

    # Direct → END
    g.add_edge("direct", END)

    return g


# Compiled graph singletons
_compiled_graph_full = None
_compiled_graph_retrieval_only = None


def get_graph(include_generate: bool = True):
    global _compiled_graph_full, _compiled_graph_retrieval_only
    
    if include_generate:
        if _compiled_graph_full is None:
            _compiled_graph_full = build_graph(include_generate=True).compile()
        return _compiled_graph_full
    else:
        if _compiled_graph_retrieval_only is None:
            _compiled_graph_retrieval_only = build_graph(include_generate=False).compile()
        return _compiled_graph_retrieval_only


# ---------------------------------------------------------------------------
# Separated entry-points for UI progress updates
# ---------------------------------------------------------------------------

def retrieve_context(query: str) -> dict[str, Any]:
    """Execute the retrieval graph (routing, retrieval, reranking, CRAG)."""
    graph = get_graph(include_generate=False)
    return graph.invoke({"query": query})


def generate_answer_stream(query: str, context_data: dict[str, Any]) -> Iterator[str]:
    """Generate the answer streaming, using context from retrieve_context."""
    logger.info("Starting answer generation stream...")
    
    # If direct route, yield the response as-is (already in context_data)
    if context_data.get("route") == "DIRECT":
        logger.info("Route is DIRECT, simulating streaming of pre-generated response.")
        response_text = context_data.get("response", "")
        if not response_text:
            response_text = "I'm sorry, I couldn't generate a response. Please try again."
        
        # Simulate streaming for consistent UI experience
        import time
        for word in response_text.split(" "):
            yield word + " "
            time.sleep(0.02)
        return

    # For retrieval routes, generate the answer with streaming
    route = context_data.get("route", "VECTOR")
    docs = context_data.get("reranked_docs", context_data.get("documents", []))
    logger.info(f"Generating answer with {len(docs)} documents in context (Route: {route}).")
    
    # Late-search for diagram if needed
    diagram_url = None
    if context_data.get("needs_diagram"):
        logger.info("Searching for diagram...")
        diagram_url = search_diagram(query)
        logger.info(f"Diagram found: {diagram_url}")

    # Use more context for RELATIONAL queries
    limit = 15 if route == "RELATIONAL" else 5
    context = "\n\n---\n\n".join(d.page_content for d in docs[:limit])

    diagram_instruction = ""
    if diagram_url:
        diagram_instruction = (
            f"\n\nA relevant diagram has been found. Include this image in your "
            f"answer using Markdown:\n![Economics Diagram]({diagram_url})"
        )

    if route == "RELATIONAL":
        system_msg = (
            "You are an H2 Economics library assistant. The user is asking for specific "
            "resources (notes, papers, etc.) from the database. Format your response "
            "clearly, using a table or a structured list. Include Source, Topic, and Year "
            "for each item. If no specific items are found, inform the user."
        )
    else:
        system_msg = (
            "You are an expert H2 Economics tutor. Using ONLY the context below, "
            "provide a comprehensive, well-structured answer to the student's question. "
            "Use proper economics terminology and structure your answer with headers "
            "where appropriate. If the context is insufficient, say so honestly. "
            "Use examples to illustrate your point as far as possible.\n\n"
            "LATEX FORMATTING RULES:\n"
            "1. ONLY use LaTeX for mathematical formulas, variables, and symbols (e.g., $PED$, $\\Delta$, $Q_d$).\n"
            "2. NEVER use LaTeX for normal words or conversational text (e.g., do NOT write $demand$ or $market$).\n"
            "3. ALWAYS wrap mathematical symbols like Delta in dollar signs: $\\Delta$. NEVER output raw backslash commands like \\Delta outside of dollar signs.\n"
            "4. Use single dollar signs for inline math ($x$) and double dollar signs for block math ($$x$$).\n"
            "5. Do NOT use brackets like \\( \\) or \\[ \\]."
            "6. When using the $ sign to express the concept of dollars, remember to use \$ to avoid LaTeX"
        )

    user_msg = (
        f"Context:\n{context}\n{diagram_instruction}\n\n"
        f"Student question: {query}"
    )
    prompt = f"System: {system_msg}\nUser: {user_msg}"

    logger.info("Initializing ChatOpenAI for generation...")
    try:
        llm = ChatOpenAI(
            model=LLM_MODEL, 
            openai_api_key=API_KEY,
            openai_api_base=API_BASE_URL,
            temperature=0.3, 
            streaming=True
        )

        chunk_count = 0
        for chunk in llm.stream(prompt):
            chunk_count += 1
            yield chunk.content
        logger.info(f"Generation complete. Yielded {chunk_count} chunks.")
    except Exception as e:
        logger.error(f"Error during generation streaming: {e}")
        yield f"\n\n**Error evaluating response:** {str(e)}"


def stream_response(query: str) -> Iterator[str]:
    """Legacy wrapper: Run retrieval then stream answer."""
    context_data = retrieve_context(query)
    yield from generate_answer_stream(query, context_data)


def run_query(query: str) -> str:
    """Non-streaming convenience function."""
    graph = get_graph(include_generate=True)
    result = graph.invoke({"query": query})
    return result.get("response", "")
