"""
Streamlit UI for the H2 Economics RAG Chatbot.
"""

import streamlit as st

from src.config import H2_ECON_TOPICS, LLM_MODEL, SCHOOLS
from src.agent import stream_response

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multiplier - Your H2 Economics Tutor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header style */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(15, 52, 96, 0.3);
    }
    .main-header h1 {
        color: #e2e8f0;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0.5rem 0 0 0;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
        font-size: 1rem;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #94a3b8;
        font-size: 0.85rem;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }

    /* Info boxes */
    .info-pill {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        margin: 0.15rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="main-header">
    <h1>📊 Multiplier - Your H2 Economics Tutor</h1>
    <p>Ask me anything about H2 Economics — from market failure to macroeconomic policies</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ System Info")
    st.markdown(f"**Model:** `{LLM_MODEL}`")
    st.markdown("**Mode:** API (PublicAI)")
    st.divider()

    st.markdown("### 🏫 Available Schools")
    for s in SCHOOLS:
        st.markdown(f"- {s}")
    st.divider()

    st.markdown("### 📚 Topics Covered")
    for t in H2_ECON_TOPICS:
        st.markdown(f"- {t}")
    st.divider()

    st.markdown("### 💡 Example Queries")
    examples = [
        "Show me NYJC 2024 notes on trade",
        "Explain market failure with examples",
        "Compare fiscal and monetary policy",
        "What is the multiplier effect?",
        "RI notes on macroeconomic policies",
    ]
    for ex in examples:
        st.markdown(f"- *{ex}*")

# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask an H2 Economics question..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate & stream assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # 1. Retrieval phase with status updates
        context_data = None
        with st.status("Thinking...", expanded=True) as status:
            st.write("🔍 Analyzing query...")
            # Import here to avoid circular imports if any, or just standard usage
            from src.agent import retrieve_context, generate_answer_stream

            context_data = retrieve_context(prompt)

            # Show what happened based on the route
            route = context_data.get("route", "DIRECT")
            if route == "RELATIONAL":
                st.write("🗄️ Querying database for school notes/data...")
            elif route == "VECTOR":
                st.write("📚 Searching conceptual knowledge base...")
                st.write("🧠 Reranking documents for relevance...")
                if context_data.get("needs_diagram"):
                    st.write("🖼️ Looking for relevant diagrams...")

            status.update(label="Response ready!", state="complete", expanded=False)

        # 2. Generation phase
        try:
            if context_data is None:
                st.error(
                    "❌ Error: Context data is missing. Retrieval failed silently."
                )
            else:
                full_response = st.write_stream(
                    generate_answer_stream(prompt, context_data)
                )
        except Exception as e:
            st.error(f"❌ Error during generation: {str(e)}")
            import traceback

            st.code(traceback.format_exc())

    st.session_state.messages.append({"role": "assistant", "content": full_response})
