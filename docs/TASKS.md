Here is the updated TASKS.md for Google Antigravity. It incorporates your specific school and topic lists, the database schema, and the logic for local CPU-only execution with DuckDuckGo for diagram retrieval.
TASKS.md
1. Project Objective

Build a high-performance H2 Economics RAG agent running locally on CPU. The agent must intelligently route between a structured SQL library (for specific school notes) and a Chroma vector store (for conceptual retrieval), while fetching real-time diagrams via DuckDuckGo. For all LLMs, use Ollama/langchain ollama to host it locally.
2. Model & Tool Configuration (CPU Optimized)

    LLM Setup: Use Ollama with gemma3:4b or qwen3:1.7b for high-speed local inference.

    Search Tool: Instantiate DuckDuckGoSearchRun.

        Prompt Logic: If the response requires a visual, the agent must search using keywords like "H2 Economics [TOPIC] diagram".

    Databases:

        Vector: Chroma collection econ_notes at root.

        SQL: SQLite file h2_economics_library.db with table h2_econ_storage.

3. Semantic Router Node

    Logic: Classify incoming queries into three paths:

        RELATIONAL: If the query mentions a specific JC (e.g., "NYJC", "RI") or a specific year.

        VECTOR: For conceptual "Explain" or "How-to" questions.

        DIRECT: For general greetings or non-economics chat.

    School/Topic Matching: Use the provided SCHOOLS and H2_ECON_TOPICS lists to help the LLM identify entities.

4. Retrieval Pipelines
A. SQL Query Constructor (Relational Path)

    Prompt: Convert natural language to SQL for the h2_econ_storage table.

    Coercion: Ensure the LLM maps school acronyms (e.g., "SAJC") to the full string "St. Andrew's Junior College (SAJC)" from the SCHOOLS list.

    Output: Execute SELECT content, source FROM h2_econ_storage WHERE ...

B. Advanced Vector RAG (Vector Path)

    Query Decomposition: Generate 3 variations of the query to capture different economic angles (e.g., Consumer vs. Producer vs. Government perspective).

    HyDE: Generate a 1-paragraph "hypothetical" answer to use as a search query for Chroma.

5. Post-Retrieval & CRAG

    Ranking: Implement a simple similarity score filter. You can use ollama dengcao/Qwen3-Reranker-0.6B:Q8_0 to help with the ranking.

    Corrective RAG (CRAG):

        Check if retrieved chunks contain the required content.

        If the LLM determines a diagram is necessary to explain the answer (e.g., "Illustrate the welfare loss"), trigger the DuckDuckGoSearchRun tool.

    Diagram Injection: Extract the most relevant image/source link from DuckDuckGo and embed it in the Markdown response.

6. Real-time Streaming

    Implementation: Enable streaming=True on the LangChain LLM object.

    UI Integration: Ensure Antigravity's output block handles Server-Sent Events (SSE) to display the "thinking" process and the response as it is generated.

Implementation Notes for Antigravity

    SQL Handling: Use LangChain's SQLDatabaseChain for the Relational path, but customize the prompt to strictly use the H2_ECON_TOPICS categories when filtering the topic column.

    DuckDuckGo Diagram Search: Instruct the agent to append "site:wikimedia.org OR site:economicshelp.org" to diagram searches to get high-quality academic visuals.

    CPU Performance: Ensure Ollama is configured with num_thread equal to the physical cores of the CPU to prevent lag during parallel query decomposition.