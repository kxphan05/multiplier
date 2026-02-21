# Multiplier - H2 Economics RAG Chatbot

Run the Streamlit app:
```bash
streamlit run app/app.py
```

Or using the virtual environment:
```bash
.venv/bin/streamlit run app/app.py
```

Run evaluation:
```bash
.venv/bin/python -m eval.rag_evaluation
```

From pkx (Not AI): The motivation behind this project was to test whether a RAG pipeline is really necessary to improve responses to students that are asking questions related to H2 Economics. As everyone knows, we could very easily use a Large and Established LLM to give more-than-satisfactory answers to our H2 Econs questions. I am trying to see if this parametric knowledge can be improved with a RAG pipeline.

At first, I wanted to generate as complex a pipeline as possible to impress people. But as people gradually learn in tech, what is complex and 'perfect' often doesn't beat what is 'good enough'. Responses would take 6 seconds on average, which really isn't ideal in terms of UX, especially if they can easily switch to Gemini or ChatGPT to give them a quicker answer. Hence, I created an evaluation pipeline to see
  1. Whether the context pulled by RAG is relevant in the first place.
  2. Whether I can remove any part of this convoluted pipeline (especially in the vector route) to improve latency without massively deteriorating the quality of the response.

I tried addressing the first point by implementing RAGAS, which gives telemetry into what is being documents are being pulled during retrieval. It also judges context based on Context Precision (whether the context is relavant to the ground truth) and Context Recall (whether all of the ideas in the ground truth are retrieved).

Unfortunately, since I do not have access to a strong, free tier embedding model, I couldn't judge AnswerRelevancy.

I decided to do away with the metrics of Faithfulness and Correctness since answers in economics are more of a social science -- two equally valid answers can come to different conclusions.

For the second point, I ultimately decided to use 3 LLM judges to grade the responses on - clarity, correctness, depth of analysis and relevance to singapore. They take on 3 different identities as a Student, Teacher and Examiner.