import json
import logging
import os
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from openai import OpenAI, AsyncOpenAI
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import (
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.crag import rerank_documents
from src.database import query_chroma


class OpenRouterLLM:
    def __init__(self, api_key: str, model: str = "allenai/Molmo2-8B"):
        self.client = OpenAI(base_url="https://api.publicai.co/v1", api_key=api_key)
        self.model = model
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://api.publicai.co/v1",
            temperature=0,
        )

    def chat(self, messages: List[Dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, max_tokens=2000
        )
        return response.choices[0].message.content


def load_eval_data(csv_path: str) -> pd.DataFrame:
    """Load evaluation data from CSV"""
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} evaluation queries")
    return df


def run_rag_query(query: str) -> Dict[str, Any]:
    """Execute the full RAG pipeline and return results"""
    from src.agent import retrieve_context, get_graph

    # Get retrieval context
    context_data = retrieve_context(query)

    # Get documents
    docs = context_data.get("reranked_docs", context_data.get("documents", []))
    contexts = [doc.page_content for doc in docs]

    # Generate answer
    graph = get_graph(include_generate=True)
    result = graph.invoke({"query": query})
    answer = result.get("response", "")

    return {
        "answer": answer,
        "contexts": contexts,
        "documents": docs,
        "route": context_data.get("route", "UNKNOWN"),
    }


def evaluate_with_ragas(
    eval_csv: str, openrouter_api_key: str, output_path: str = "evaluation_results.csv"
) -> pd.DataFrame:
    """Run RAGAS evaluation on the RAG pipeline"""

    # Load eval data
    eval_df = load_eval_data(eval_csv)

    evaluator_wrapper = llm_factory('allenai/Molmo2-8B', client=AsyncOpenAI(base_url="https://api.publicai.co/v1", api_key=openrouter_api_key), max_tokens=2048, temperature=0)

    # Run RAG pipeline for each query
    answers = []
    contexts_list = []

    logger.info("Running RAG pipeline on evaluation queries...")

    for idx, row in eval_df.iterrows():
        query = str(row["query"])
        try:
            result = run_rag_query(query)
            answers.append(result["answer"])
            contexts_list.append(result["contexts"])
            logger.info(f"Completed evaluation {int(idx) + 1}/{len(eval_df)}")
        except Exception as e:
            logger.error(f"Error on query {idx}: {e}")
            answers.append("")
            contexts_list.append([])

    # Create RAGAS dataset using HuggingFace datasets library
    from datasets import Dataset as HFDataset

    ragas_data = HFDataset.from_dict(
        {
            "question": eval_df["query"].tolist(),
            "answer": answers,
            "ground_truth": eval_df["ground_truth"].tolist(),
            "contexts": contexts_list,
        }
    )

    # Run evaluation
    logger.info("Running RAGAS evaluation...")
    metrics = [Faithfulness(), AnswerCorrectness(), ContextPrecision(), ContextRecall()]

    results = evaluate(dataset=ragas_data, metrics=metrics, llm=evaluator_wrapper)

    # Convert to DataFrame and save
    results_df = results.to_pandas()
    results_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

    return results_df


def run_llm_judge_evaluation(
    eval_csv: str, openrouter_api_key: str, output_path: str = "llm_judge_results.csv"
) -> pd.DataFrame:
    """Run LLM judge evaluation using OpenRouter"""

    eval_df = load_eval_data(eval_csv)
    openrouter_client = OpenAI(
        base_url="https://api.publicai.co/v1", api_key=openrouter_api_key
    )

    results = []

    logger.info("Running LLM judge evaluation...")

    for idx, row in eval_df.iterrows():
        query = str(row["query"])
        try:
            rag_result = run_rag_query(query)

            response = openrouter_client.chat.completions.create(
                model="allenai/Molmo2-8B",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert economics tutor evaluating RAG answers. Rate from 1-10.",
                    },
                    {
                        "role": "user",
                        "content": f"""Query: {row["query"]}

Ground Truth: {row["ground_truth"]}

Generated Answer: {rag_result["answer"]}

Rate the answer on:
1. Faithfulness (does it stick to the context?)
2. Correctness (is it factually accurate?)
3. Relevance (does it answer the question?)

Do not rate everything too highly. Remember, the answer must be evaluated critically based on the provided contexts and the ground truth. If the answer is mostly correct but has some minor inaccuracies, it should not receive a perfect score.

Format as JSON with keys: faithfulness, correctness, relevance, feedback""",
                    },
                ],
                temperature=0,
                max_tokens=500,
            )

            content = response.choices[0].message.content
            try:
                scores = json.loads(content)
            except:
                scores = {
                    "faithfulness": 0,
                    "correctness": 0,
                    "relevance": 0,
                    "feedback": content,
                }

            results.append(
                {
                    "query": row["query"],
                    "ground_truth": row["ground_truth"],
                    "generated_answer": rag_result["answer"],
                    **scores,
                }
            )

            logger.info(f"Completed {int(idx) + 1}/{len(eval_df)}")

        except Exception as e:
            logger.error(f"Error on query {idx}: {e}")
            results.append({"query": row["query"], "error": str(e)})

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    logger.info(f"LLM judge results saved to {output_path}")

    return results_df


def run_unified_evaluation(
    eval_csv: str,
    openrouter_api_key: str,
    ragas_output: str = "ragas_results.csv",
    judge_output: str = "llm_judge_results.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run RAG pipeline once, then execute both RAGAS and LLM judge evaluation.

    Returns:
        Tuple of (ragas_results_df, judge_results_df)
    """
    eval_df = load_eval_data(eval_csv)

    logger.info("Running RAG pipeline for all queries...")

    answers: List[str] = []
    contexts_list: List[List[str]] = []

    for idx, row in eval_df.iterrows():
        query = str(row["query"])
        try:
            result = run_rag_query(query)
            answers.append(result["answer"])
            contexts_list.append(result["contexts"])
            logger.info(f"RAG pipeline: {int(idx) + 1}/{len(eval_df)}")
        except Exception as e:
            logger.error(f"Error on query {idx}: {e}")
            answers.append("")
            contexts_list.append([])

    logger.info("Running RAGAS evaluation...")
    evaluator_wrapper = llm_factory('allenai/Molmo2-8B', client=AsyncOpenAI(base_url="https://api.publicai.co/v1", api_key=openrouter_api_key), max_tokens=2048, temperature=0)

    from datasets import Dataset as HFDataset

    ragas_data = HFDataset.from_dict(
        {
            "question": eval_df["query"].tolist(),
            "answer": answers,
            "ground_truth": eval_df["ground_truth"].tolist(),
            "contexts": contexts_list,
        }
    )

    metrics = [Faithfulness(), AnswerCorrectness(), ContextPrecision(), ContextRecall()]
    ragas_results = evaluate(dataset=ragas_data, metrics=metrics, llm=evaluator_wrapper)
    ragas_df = ragas_results.to_pandas()
    ragas_df.to_csv(ragas_output, index=False)
    logger.info(f"RAGAS results saved to {ragas_output}")

    logger.info("Running LLM judge evaluation...")
    openrouter_client = OpenAI(
        base_url="https://api.publicai.co/v1", api_key=openrouter_api_key
    )

    judge_results: List[Dict] = []

    for idx, row in eval_df.iterrows():
        try:
            response = openrouter_client.chat.completions.create(
                model="allenai/Molmo2-8B",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert economics tutor evaluating RAG answers. Rate from 1-10.",
                    },
                    {
                        "role": "user",
                        "content": f"""Query: {row["query"]}

Ground Truth: {row["ground_truth"]}

Generated Answer: {answers[idx]}

Rate the answer on:
1. Faithfulness (does it stick to the context?)
2. Correctness it fact (isually accurate?)
3. Relevance (does it answer the question?)

Format as JSON with keys: faithfulness, correctness, relevance, feedback""",
                    },
                ],
                temperature=0,
                max_tokens=500,
            )

            content = response.choices[0].message.content
            try:
                scores = json.loads(content)
            except:
                scores = {
                    "faithfulness": 0,
                    "correctness": 0,
                    "relevance": 0,
                    "feedback": content,
                }

            judge_results.append(
                {
                    "query": row["query"],
                    "ground_truth": row["ground_truth"],
                    "generated_answer": answers[idx],
                    **scores,
                }
            )

            logger.info(f"LLM judge: {int(idx) + 1}/{len(eval_df)}")

        except Exception as e:
            logger.error(f"Error on query {idx}: {e}")
            judge_results.append({"query": row["query"], "error": str(e)})

    judge_df = pd.DataFrame(judge_results)
    judge_df.to_csv(judge_output, index=False)
    logger.info(f"LLM judge results saved to {judge_output}")

    return ragas_df, judge_df


def generate_summary_report(results_df: pd.DataFrame) -> Dict:
    """Generate summary statistics from evaluation results"""

    summary = {}

    numeric_cols = results_df.select_dtypes(include=["float64", "int64"]).columns

    for col in numeric_cols:
        summary[f"avg_{col}"] = results_df[col].mean()
        summary[f"std_{col}"] = results_df[col].std()
        summary[f"min_{col}"] = results_df[col].min()
        summary[f"max_{col}"] = results_df[col].max()

    return summary


LatencyResult = Dict[str, float]


def measure_embedding_time(query: str) -> float:
    """Measure time to embed a query using Chroma's default embedding function."""
    from chromadb.utils import embedding_functions

    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    start = time.perf_counter()
    embedding_fn([query])
    return time.perf_counter() - start


def measure_retrieval_time(query: str, n_results: int = 6) -> float:
    """Measure time to retrieve documents from Chroma."""
    start = time.perf_counter()
    query_chroma(query, n_results=n_results)
    return time.perf_counter() - start


def measure_rerank_time(query: str, docs: List[Document], top_k: int = 5) -> float:
    """Measure time for LLM reranking."""
    start = time.perf_counter()
    rerank_documents(query, docs, top_k=top_k)
    return time.perf_counter() - start


def run_latency_evaluation(
    eval_csv: str, n_results: int = 6, top_k: int = 5
) -> pd.DataFrame:
    """Run latency evaluation on the RAG pipeline."""
    eval_df = load_eval_data(eval_csv)
    results: List[LatencyResult] = []

    logger.info("Running latency evaluation...")

    for idx, row in eval_df.iterrows():
        query = str(row["query"])
        logger.info(f"[{int(idx) + 1}/{len(eval_df)}] {query[:50]}...")

        result: LatencyResult = {}

        t_embed = measure_embedding_time(query)
        result["embedding_time_s"] = t_embed

        t_retrieve = measure_retrieval_time(query, n_results)
        result["retrieval_time_s"] = t_retrieve

        docs = query_chroma(query, n_results=n_results)

        if docs:
            t_rerank = measure_rerank_time(query, docs, top_k)
            result["rerank_time_s"] = t_rerank
        else:
            result["rerank_time_s"] = 0.0

        result["total_time_s"] = (
            result["embedding_time_s"]
            + result["retrieval_time_s"]
            + result["rerank_time_s"]
        )

        results.append(result)
        logger.info(
            f"   Embed: {t_embed:.3f}s | Retrieve: {t_retrieve:.3f}s | "
            f"Rerank: {result['rerank_time_s']:.3f}s | Total: {result['total_time_s']:.3f}s"
        )

    results_df = pd.DataFrame(results)
    return results_df


def print_latency_report(results_df: pd.DataFrame) -> None:
    """Print a formatted latency report."""
    print("\n" + "=" * 70)
    print("LATENCY EVALUATION REPORT")
    print("=" * 70)
    print(f"\nTotal queries evaluated: {len(results_df)}")
    print("\nAggregate Statistics (in seconds):")
    print("-" * 70)

    metric_names = {
        "embedding_time_s": "Embedding",
        "retrieval_time_s": "Retrieval (Chroma)",
        "rerank_time_s": "Reranking (LLM)",
        "total_time_s": "Total E2E",
    }

    for metric, name in metric_names.items():
        if metric in results_df.columns:
            values = results_df[metric]
            sorted_values = sorted(values)
            n = len(sorted_values)
            p95 = sorted_values[int(n * 0.95)] if n >= 20 else sorted_values[-1]
            p99 = sorted_values[int(n * 0.99)] if n >= 100 else sorted_values[-1]

            print(f"\n{name}:")
            print(f"  Mean:   {values.mean():.3f}s")
            print(f"  Median: {values.median():.3f}s")
            print(f"  Min:    {values.min():.3f}s")
            print(f"  Max:    {values.max():.3f}s")
            print(f"  P95:    {p95:.3f}s")
            print(f"  P99:    {p99:.3f}s")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    import sys

    PUBLICAI_API_KEY = os.getenv("PUBLICAI_API_KEY")
    if not PUBLICAI_API_KEY:
        print("Error: Please set PUBLICAI_API_KEY environment variable")
        sys.exit(1)

    # Run latency evaluation
    print("=" * 50)
    print("Running Latency Evaluation...")
    print("=" * 50)

    latency_results = run_latency_evaluation(
        eval_csv="eval/eval.csv",
    )
    latency_results.to_csv("latency_results.csv", index=False)
    print_latency_report(latency_results)

    # Run unified RAGAS + LLM judge evaluation (single RAG pipeline pass)
    print("\n" + "=" * 50)
    print("Running Unified RAGAS + LLM Judge Evaluation...")
    print("=" * 50)

    ragas_results, judge_results = run_unified_evaluation(
        eval_csv="eval/eval.csv",
        openrouter_api_key=PUBLICAI_API_KEY,
    )

    print("\nRAGAS Results:")
    print(ragas_results)

    print("\nLLM Judge Results:")
    print(judge_results)

    # Summary
    print("\n" + "=" * 50)
    print("Summary Report")
    print("=" * 50)

    ragas_summary = generate_summary_report(ragas_results)
    judge_summary = generate_summary_report(judge_results)

    print("\nRAGAS Metrics:")
    for metric, value in ragas_summary.items():
        print(f"  {metric}: {value:.3f}" if value else f"  {metric}: N/A")

    print("\nLLM Judge Metrics:")
    for metric, value in judge_summary.items():
        if metric != "query":
            print(
                f"  {metric}: {value:.3f}"
                if isinstance(value, float)
                else f"  {metric}: {value}"
            )
