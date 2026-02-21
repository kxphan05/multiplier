"""
Unit tests for the query routing and prompt generation paths.
Tests verify the prompts generated for DIRECT, VECTOR, and RELATIONAL routes.
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from src.router import route_query
from src.retrieval import handle_direct, handle_vector, handle_relational
from src.agent import AgentState, node_route, node_direct, node_relational, node_vector


class TestRouter:
    """Tests for the router module."""

    @patch("src.router.ChatOpenAI")
    def test_route_direct_greeting(self, mock_llm):
        """Test that greetings are routed to DIRECT."""
        mock_response = MagicMock()
        mock_response.content = "DIRECT"
        mock_llm.return_value.invoke.return_value = mock_response

        result = route_query("Hello there!")
        assert result == "DIRECT"

    @patch("src.router.ChatOpenAI")
    def test_route_vector_concept(self, mock_llm):
        """Test that conceptual questions are routed to VECTOR."""
        mock_response = MagicMock()
        mock_response.content = "VECTOR"
        mock_llm.return_value.invoke.return_value = mock_response

        result = route_query("Explain what is the multiplier effect?")
        assert result == "VECTOR"

    @patch("src.router.ChatOpenAI")
    def test_route_relational_school(self, mock_llm):
        """Test that school-specific queries are routed to RELATIONAL."""
        mock_response = MagicMock()
        mock_response.content = "RELATIONAL"
        mock_llm.return_value.invoke.return_value = mock_response

        result = route_query("Show me NJC notes on market failure")
        assert result == "RELATIONAL"


class TestDirectRoute:
    """Tests for the DIRECT route - simple LLM response."""

    @patch("src.retrieval.ChatOpenAI")
    def test_handle_direct_returns_string(self, mock_llm):
        """Test that handle_direct returns a string response."""
        mock_response = MagicMock()
        mock_response.content = "Hello! I'm your H2 Economics tutor."
        mock_llm.return_value.invoke.return_value = mock_response

        result = handle_direct("Hi there!")

        assert isinstance(result, str)
        assert len(result) > 0
        mock_llm.return_value.invoke.assert_called_once()


class TestVectorRoute:
    """Tests for the VECTOR route - conceptual knowledge retrieval."""

    @patch("src.retrieval.ChatOpenAI")
    @patch("src.retrieval.query_chroma_multi")
    @patch("src.retrieval.get_all_documents")
    @patch("src.retrieval.BM25Retriever")
    def test_handle_vector_returns_dict(
        self, mock_bm25, mock_get_docs, mock_chroma, mock_llm
    ):
        """Test that handle_vector returns documents and needs_diagram flag."""
        mock_chroma.return_value = []
        mock_get_docs.return_value = []
        mock_bm25_instance = MagicMock()
        mock_bm25_instance.invoke.return_value = []
        mock_bm25.from_documents.return_value = mock_bm25_instance

        mock_response = MagicMock()
        mock_response.content = """DECOMPOSITION:
- How does MPS affect the multiplier?
- What is the relationship between MPS and multiplier?
- Why is MPS important in economics?

HYDE:
The multiplier effect shows how an initial change in spending leads to a larger change in national income. The formula is Multiplier = 1 / MPS.

DIAGRAM: YES"""
        mock_llm.return_value.invoke.return_value = mock_response

        result = handle_vector("Explain the multiplier effect")

        assert isinstance(result, dict)
        assert "documents" in result
        assert "needs_diagram" in result
        assert result["needs_diagram"] is True


class TestRelationalRoute:
    """Tests for the RELATIONAL route - SQL-based retrieval."""

    @patch("src.retrieval.ChatOpenAI")
    @patch("src.retrieval.run_sql_query")
    def test_handle_relational_returns_docs(self, mock_sql, mock_llm):
        """Test that handle_relational returns Document objects."""
        mock_response = MagicMock()
        mock_response.content = "SELECT * FROM h2_econ_storage WHERE school LIKE '%RI%'"
        mock_llm.return_value.invoke.return_value = mock_response

        mock_sql.return_value = [
            {
                "source": "RI Notes",
                "topic": "Market Failure",
                "year": 2024,
                "doc_type": "Notes",
                "content": "Market failure occurs when...",
            }
        ]

        result = handle_relational("Show me RI notes on market failure")

        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], Document)


class TestAgentNodes:
    """Tests for LangGraph agent nodes."""

    @patch("src.agent.route_query")
    def test_node_route_returns_route(self, mock_route):
        """Test that node_route returns the route in state."""
        mock_route.return_value = "VECTOR"

        state = AgentState(query="What is elasticity?")
        result = node_route(state)

        assert "route" in result
        assert result["route"] == "VECTOR"

    @patch("src.agent.handle_direct")
    def test_node_direct_returns_response(self, mock_direct):
        """Test that node_direct returns response."""
        mock_direct.return_value = "Hello! I'm here to help."

        state = AgentState(query="Hi")
        result = node_direct(state)

        assert "response" in result

    @patch("src.agent.handle_vector")
    def test_node_vector_returns_docs_and_diagram_flag(self, mock_vector):
        """Test that node_vector returns documents and needs_diagram."""
        mock_vector.return_value = {
            "documents": [Document(page_content="Test doc", metadata={})],
            "needs_diagram": True,
        }

        state = AgentState(query="Explain demand")
        result = node_vector(state)

        assert "documents" in result
        assert "needs_diagram" in result

    @patch("src.agent.handle_relational")
    def test_node_relational_returns_docs(self, mock_relational):
        """Test that node_relational returns documents."""
        mock_relational.return_value = [Document(page_content="Test doc", metadata={})]

        state = AgentState(query="Show me NJC notes")
        result = node_relational(state)

        assert "documents" in result
        assert len(result["documents"]) == 1


class TestPromptGeneration:
    """Integration-style tests to verify prompt structures."""

    @patch("src.retrieval.ChatOpenAI")
    def test_direct_prompt_structure(self, mock_llm):
        """Verify that DIRECT route uses the correct system prompt."""
        mock_response = MagicMock()
        mock_response.content = "Hello!"
        mock_llm.return_value.invoke.return_value = mock_response

        handle_direct("Hello")

        call_args = mock_llm.return_value.invoke.call_args[0][0]
        assert "System:" in call_args
        assert "Multiplier" in call_args

    @patch("src.retrieval.ChatOpenAI")
    def test_relational_prompt_contains_sql_instructions(self, mock_llm):
        """Verify that RELATIONAL route includes SQL generation instructions."""
        mock_response = MagicMock()
        mock_response.content = "SELECT * FROM h2_econ_storage"
        mock_llm.return_value.invoke.return_value = mock_response

        with patch("src.retrieval.run_sql_query", return_value=[]):
            handle_relational("Show me RI notes")

        call_args = mock_llm.return_value.invoke.call_args[0][0]
        assert "SELECT" in call_args
        assert "h2_econ_storage" in call_args

    @patch("src.retrieval.ChatOpenAI")
    def test_vector_prompt_contains_decomposition(self, mock_llm):
        """Verify that VECTOR route includes retrieval planning."""
        mock_response = MagicMock()
        mock_response.content = """DECOMPOSITION:
- Query 1
- Query 2

HYDE:
Answer

DIAGRAM: NO"""
        mock_llm.return_value.invoke.return_value = mock_response

        with patch("src.retrieval.query_chroma_multi", return_value=[]):
            with patch("src.retrieval.get_all_documents", return_value=[]):
                with patch("src.retrieval.BM25Retriever") as mock_bm25:
                    mock_bm25_instance = MagicMock()
                    mock_bm25_instance.invoke.return_value = []
                    mock_bm25.from_documents.return_value = mock_bm25_instance

                    handle_vector("What is PED?")

        call_args = mock_llm.return_value.invoke.call_args[0][0]
        assert "DECOMPOSITION" in call_args
        assert "HYDE" in call_args
        assert "DIAGRAM" in call_args


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
