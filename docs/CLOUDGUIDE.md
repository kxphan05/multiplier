# H2 Economics RAG Chatbot – Cloud Deployment Guide

This guide outlines recommended cloud hosting solutions and step-by-step instructions for deploying the H2 Economics RAG Chatbot.

## Hosting Recommendations

### 📊 Option 1: Streamlit Community Cloud (Easiest)
Best for: Quick, free deployment of the UI.
*   **Pros**: Free, extremely easy (connect GitHub repo), built-in streaming support.
*   **Cons**: No built-in GPU/CPU for local LLMs; requires switching to a Cloud API (OpenAI/Gemini).
*   **Best for**: Demonstrations or low-traffic usage with a cloud LLM.

### 🐳 Option 2: Railway or Render (Best for Docker)
Best for: Containerized full-stack apps (Streamlit + DBs).
*   **Pros**: Supports Docker directly, handles persistent storage (volumes) for your Chroma/SQLite DBs.
*   **Cons**: Paid (usage-based). Still likely needs a Cloud API for LLM unless you pay for heavy compute.
*   **Best for**: Production-like deployment of the current stack.

### 🤖 Option 3: Hugging Face Spaces (Best for ML)
Best for: Integrated machine learning prototypes.
*   **Pros**: Free tier available, built-in Streamlit support, can run smaller local models (if resource limits allow).
*   **Cons**: Resource limits on free tier.
*   **Best for**: Sharing the project with the AI/ML community.

---

## Deployment Steps (Recommended Path)

### Step 1: Transition to Cloud LLM (CRITICAL)
Ollama (local) is designed for local CPU/GPU use. For most cloud hosts, you should switch to a Cloud API like **Google Gemini (Free tier available)** or **OpenAI**.

1.  **Get an API Key**: e.g., from [Google AI Studio](https://aistudio.google.com/).
2.  **Update `config.py`**: Add `GOOGLE_API_KEY`.
3.  **Update `agent.py`**: Switch `ChatOllama` to `ChatGoogleGenerativeAI`.

### Step 2: Prepare for Vercel (Next.js Path)
If you specifically want to use **Vercel**, you should migrate from Streamlit to **Next.js**.
*   **Backend**: Use Vercel Serverless Functions (Python or TypeScript).
*   **Database**: Move SQLite to **Vercel Postgres** and Chroma to a hosted vector DB like **Pinecone** or **Supabase**.
*   **LLM**: Use the Vercel AI SDK with OpenAI or Gemini.

### Step 3: Deploy to Railway (Docker Path) – Recommended
1.  **Commit your code** (including `Dockerfile` and `docker-compose.yml`) to GitHub.
2.  **Log in to [Railway](https://railway.app/)**.
3.  **Create a new Project** → **Deploy from GitHub repo**.
4.  **Set Environment Variables**: 
    - `STREAMLIT_SERVER_PORT=8501`
    - `STREAMLIT_SERVER_ADDRESS=0.0.0.0`
    - (If using cloud LLM) `GOOGLE_API_KEY=your_key_here`
5.  **Wait for Build**: Railway will detect the `Dockerfile` and build it automatically.

---

## Summary Table

| Hosting | Type | Supports local LLM? | Best For |
| :--- | :--- | :--- | :--- |
| **Streamlit Cloud** | UI-Specific | No (API only) | Fast & Free UI |
| **Railway** | Containerized | Partially (Costs $) | Full Stack Docker |
| **Hugging Face** | ML Platform | Yes (Small models) | AI Prototypes |
| **Vercel** | Serverless | No (API only) | Scalable Web Apps |
