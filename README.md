# H2 Economics RAG Chatbot

A powerful, local CPU-only H2 Economics RAG (Retrieval-Augmented Generation) assistant built with LangGraph, Ollama, and Streamlit. This bot intelligently routes between school-specific notes (SQL) and conceptual economics retrieval (Chroma Vector Store), fetching real-time diagrams via DuckDuckGo.

## 🚀 Quick Start (Local)

### Prerequisites

1.  **Python 3.13** and **uv** installed.
2.  **Ollama** running locally with the following models:
    *   `gemma2:2b` (Main LLM)
    *   `dengcao/Qwen3-Reranker-0.6B:Q8_0` (Reranker)
    *   `all-minilm:l6-v2` (Embeddings)

### Installation

```bash
# Clone the repository and navigate to the directory
cd econs_data

# Sync dependencies and create virtual environment
uv sync
```

### Running the App

```bash
uv run streamlit run app.py
```

## 🏗️ Architecture

![H2 Economics RAG Architecture](./pipeline.png)

## 🐳 Dockerization

You can run the application using Docker to ensure a consistent environment.

```bash
# Build and run with Docker Compose
docker compose up --build
```

*Note: The Docker container is configured to connect to Ollama running on your host machine via `host.docker.internal`.*

## ☁️ Cloud Deployment

For deploying this app to the cloud (Streamlit Cloud, Railway, Hugging Face, or Vercel), please refer to the detailed **[CLOUDGUIDE.md](CLOUDGUIDE.md)**. 

> [!IMPORTANT]
> When moving to the cloud, it is highly recommended to switch from local Ollama to a Cloud LLM API (like Google Gemini or OpenAI) to maintain performance without requiring expensive GPU instances.

## 📚 Project Structure

*   `app.py`: Streamlit UI with streaming feedback.
*   `agent.py`: LangGraph orchestration of the RAG pipeline.
*   `router.py`: Semantic classification of user intent.
*   `retrieval.py`: Handling of SQL (Relational) and Vector (Chroma) retrieval paths.
*   `crag.py`: Corrective RAG logic, reranking, and diagram injection.
*   `db.py`: Helpers for SQLite and ChromaDB.
*   `config.py`: Centralised constants for schools, topics, and models.

## 💡 Example Queries to Test

*   **Conceptual**: *"Explain the concept of market failure with examples"*
*   **School Specific**: *"Show me NYJC 2024 notes on trade"*
*   **Comparison**: *"Compare fiscal and monetary policy"*
*   **Visuals**: *"Illustrate the welfare loss from a monopoly"* (Triggers diagram search)