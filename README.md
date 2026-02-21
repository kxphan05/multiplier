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

# Ablation Study Results Summary
| Configuration | Description | Avg Total Latency (s) | Avg Rerank Time (s) | Avg Weighted Score | Context Precision | Context Recall |
|---------------|-------------|----------------------|---------------------|-------------------|-------------------|----------------|
| multi-hyde-5doc | RAG, no reranking, top 5 docs | 0.4 | N/A | ~7.1 | 0.49 | 0.17 |
| multi-hyde-5doc-JINArerank | RAG, JINA cross-encoder reranking | ~1.1 | ~0.6 | ~8.8 | 0.56 | 0.14 |
| multi-hyde-5doc-rerank-crag | RAG, LLM reranking | ~1.9 | ~1.5 | ~8.6 | 0.51 | 0.12 |
| multi-hyde-10>5doc-rerank | RAG, rerank from 10 docs to 5 | ~1.9 | ~1.6 | ~8.8 | 0.48 | 0.18 |
| parametric | No RAG (model's parametric knowledge) | ~2.3 | N/A | ~8.9 | 0.0 | 0.0 |


## Key Observations:
### Latency:
- JINA reranking is fastest (~0.6s rerank time)
- LLM reranking is slower (~1.5-1.6s)
- JINA reranking provides ~2-3x speedup over LLM reranking
### Quality (Weighted Score):
- All RAG configurations with reranking score similarly (~8.6-8.8)
- Parametric (no RAG) performs surprisingly well (~8.9)
- No-rerank baseline has lowest quality (~7.1) due to one critical failure (weighted_score=0 for COVID vaccine question)
### Context Metrics:
- JINA reranking achieves highest context precision (0.56)
- Context recall is low across all RAG configurations (0.12-0.18)
- Parametric has 0 context metrics since no retrieval occurs

From pkx (Not AI): The motivation behind this project was to test whether a RAG pipeline is really necessary to improve responses to students that are asking questions related to H2 Economics. As everyone knows, we could very easily use a Large and Established LLM to give more-than-satisfactory answers to our H2 Econs questions. I am trying to see if this parametric knowledge can be improved with a RAG pipeline.

At first, I wanted to generate as complex a pipeline as possible to impress people. But as people gradually learn in tech, what is complex and 'perfect' often doesn't beat what is 'good enough'. Responses would take 6 seconds on average, which really isn't ideal in terms of UX, especially if they can easily switch to Gemini or ChatGPT to give them a quicker answer. Hence, I created an evaluation pipeline to see
  1. Whether the context pulled by RAG is relevant in the first place.
  2. Whether I can remove any part of this convoluted pipeline (especially in the vector route) to improve latency without massively deteriorating the quality of the response.

I tried addressing the first point by implementing RAGAS, which gives telemetry into what is being documents are being pulled during retrieval. It also judges context based on Context Precision (whether the context is relavant to the ground truth) and Context Recall (whether all of the ideas in the ground truth are retrieved). I also improved the pipeline by adding BM25, so specific documents that have economic keywords (like elasticity, multiplier) can be pulled from the haystack.

Unfortunately, since I do not have access to a strong, free tier embedding model, I couldn't judge AnswerRelevancy.

I decided to do away with the metrics of Faithfulness and Correctness since answers in economics are more of a social science -- two equally valid answers can come to different conclusions.

For the second point, I ultimately decided to use 3 LLM judges to grade the responses on - clarity, correctness, depth of analysis and relevance to singapore. They take on 3 different identities as a Student, Teacher and Examiner (all with temperature 0.15 for more predictable responses).

Some problems that I faced with the LLM judges were that there were unnecessarily penalizing answers, specifically with this prompt `"Is the demand for Singapore's electronics exports likely to be price elastic or inelastic?"`. The ground truth for this answer was `Highly price elastic (PED > 1) due to the availability of many international substitutes (e.g., from Taiwan or Korea). Thus, a slight increase in the exchange rate can lead to a more than proportionate fall in quantity demanded, worsening export revenue. However, Singaporean electronics are often price inelastic because they occupy high-tech, specialized niches with few direct substitutes, making global buyers less sensitive to price changes. Furthermore, as an export-oriented economy focused on high-value manufacturing, Singapore’s reputation for precision and reliability creates a "brand" loyalty that outweighs minor price fluctuations.` Obviously this is a very open question as one may argue for price inelasticity due to a better quality of Singaporean electronics or specialized goods (which was in the response). However, the Teacher and Examiner was penalizing this answer! Hence I had to do some prompt engineering to improve the response.

In my original implementation, I tried using a normal LLM to rerank, which might not return the best results. Having tried the new JINA reranker that uses cross-encoding to determine semantic similarity, it actually performs a bit better than the LLM (probably because it is able to pick up more singaporean examples).
For examples, for the prompt `Why is the MAS's exchange rate policy more effective than interest rate policy for Singapore?`, the LLM pulled actually pulled a chapter summary as it's number one prompt `a. Why Singapore’s Central Bank Monetary Authority of Singapore (MAS) \nchooses to use a monetary policy centred on exchange rate,  \nb. How we manage this exchange rate in different economic situations \nand  \nc. The impact of the exchange rate on our macroeconomic objectives.  \n \n2.3.1 Why the choice of exchange rate as the monetary policy tool  in \nSingapore`, presumably because of it's semantic similarity. In contrast, JINA pulled the context `domestic markets like Singapore, where volume of trade is close to 4 times \nof the country’s GDP,  Singapore’s Monetary Authority of Singapore (MAS)  \nchooses exchange rate-centred monetary policy instead. \n \nInterest Rate Centred Monetary Policy \nInterest rate can be known as the price of money. From lenders’ \nperspective, it represents their cost of borrowing. From savers’ perspective,`, which provides much better Singaporean context. 

