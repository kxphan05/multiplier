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
    ContextPrecision,
    ContextRecall,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.crag import rerank
from src.database import query_chroma


class JudgePersona:
    STUDENT = {
        "name": "STUDENT",
        "system_prompt": "You are an H2 Economics student evaluating answers for clarity and understandability.",
        "focus": "Understandability",
        "rubric": """
### YOUR FOCUS: UNDERSTANDABILITY (1-10)
You are a student who needs to learn from this answer. Evaluate:
1. Is the language clear and accessible? (avoiding unnecessary jargon OR explaining it when used)
2. Are concepts broken down logically? (Step-by-step explanations)
3. Would a fellow student understand this without additional help?

Scoring Guide:
- 9-10: Crystal clear, excellent for learning
- 7-8: Clear with minor gaps
- 5-6: Understandable but needs improvement
- 3-4: Confusing, hard to follow
- 1-2: Incomprehensible or misleading
""",
        "output_key": "understandability",
    }

    TEACHER = {
        "name": "TEACHER",
        "system_prompt": "You are an experienced H2 Economics teacher evaluating answers for theoretical correctness.",
        "focus": "Correctness & Logic",
        "rubric": """
### YOUR FOCUS: CORRECTNESS & LOGIC (1-10)
You are a teacher checking for theoretical accuracy. Evaluate:
1. Are the causal chains complete? (Step A → Step B → Step C clearly linked)
2. Is the economic reasoning theoretically sound?
3. Are diagrams/concepts correctly applied?
4. Any factual errors or misconceptions?
5. Depth of analysis (does it just state the obvious or does it show deeper understanding?)

### CRITICAL INSTRUCTION:
Economic theory often allows for multiple valid perspectives. If the Generated Answer provides a logical, well-reasoned economic argument that differs from the Ground Truth but is theoretically sound within the H2 Economics syllabus (e.g., arguing inelastic vs elastic based on different valid assumptions), do NOT penalize the 'Correctness' score harshly.
Use the Ground Truth as a list of mandatory concepts. The student's answer does not need to match the phrasing or length of the Ground Truth, but it MUST encompass the core economic logic contained within it. If the student provides additional valid analysis not in or even contradicting the Ground Truth, do not penalize them; instead, reward them for 'Evaluation' (EV) marks.

IT IS COMPLETELY FINE IF THE ANSWER'S CONCLUSION DIFFERS FROM THE GROUND TRUTH AS LONG AS THE ECONOMIC REASONING IS SOUND AND THE CORE CONCEPTS ARE COVERED. The Ground Truth is not a rigid answer key but a reference for key concepts and logic that should be present in a correct answer.

Scoring Guide:
- 9-10: Flawless economic reasoning
- 7-8: Minor theoretical gaps
- 5-6: Some errors but core logic intact
- 3-4: Significant errors in reasoning
- 1-2: Fundamentally wrong economic logic

CRITICAL: If you give a score below 5, the answer will be flagged as a CRITICAL FAILURE.
""",
        "output_key": "correctness",
    }

    EXAMINER = {
        "name": "EXAMINER",
        "system_prompt": "You are a Cambridge A-Level Economics examiner evaluating answers for Singapore context and evaluation.",
        "focus": "Singapore Relevance & Evaluation",
        "rubric": """
### YOUR FOCUS: SINGAPORE RELEVANCE & EVALUATION (1-10)
You are an examiner checking for Singapore-specific application. Evaluate:
1. Does it mention Singapore's specific constraints?
2. Are Singapore-specific examples used?
3. Is there balanced evaluation? (weighing pros/cons, short-run vs long-run)

### CRITICAL INSTRUCTION:
It is important that you do not penalize negatively (ie give a low score just because it doesn't give a Singaporean example). It is more than sufficient that the answer uses features of Singapore or uses Singapore as a reference to explain the economic concepts.
For example, if the answer explains the concept of price elasticity of demand and uses the example of Singapore's public transport fares to illustrate it, that is sufficient to demonstrate Singapore relevance. The answer does not need to be 100% focused on Singapore or use multiple Singapore examples to get a high score. However, if the answer shows no evidence of understanding Singapore's unique economic context at all, then it should receive a low score.

IT IS COMPLETELY FINE IF THE ANSWER'S CONCLUSION DIFFERS FROM THE GROUND TRUTH AS LONG AS THE ECONOMIC REASONING IS SOUND AND THE CORE CONCEPTS ARE COVERED. The Ground Truth is not a rigid answer key but a reference for key concepts and logic that should be present in a correct answer.

Scoring Guide:
- 9-10: Excellent Singapore context with balanced evaluation
- 7-8: Good Singapore references, minor gaps in evaluation
- 5-6: Some Singapore context but generic treatment
- 3-4: Mostly generic, lacks Singapore specificity
- 1-2: No Singapore context or factually incorrect (e.g., suggesting interest rate policy)

CRITICAL: If you give a score below 4, the answer will be flagged as a CRITICAL FAILURE.
""",
        "output_key": "singapore_relevance",
    }

    ALL = [STUDENT, TEACHER, EXAMINER]


def parse_json_with_retry(
    content: str | None, max_retries: int = 1, persona_name: str = ""
) -> Dict:
    """Parse JSON from LLM response with retry logic."""
    if content is None:
        logger.warning(f"[{persona_name}] Content is None")
        return {}

    def try_extract_json(text: str) -> str | None:
        strategies = [
            lambda t: t[
                t.find("```json") + 7 : t.find("```", t.find("```json") + 7)
            ].strip()
            if "```json" in t
            else None,
            lambda t: t[t.find("```") + 3 : t.find("```", t.find("```") + 3)].strip()
            if t.count("```") >= 2
            else None,
            lambda t: _extract_balanced_json(t),
        ]
        for strategy in strategies:
            try:
                result = strategy(text)
                if result:
                    json.loads(result)
                    return result
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return None

    def _extract_balanced_json(text: str) -> str | None:
        start = text.find("{")
        if start == -1:
            return None
        brace_count = 0
        for i, char in enumerate(text[start:], start):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    return text[start : i + 1]
        return None

    for attempt in range(max_retries + 1):
        json_str = try_extract_json(content)
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"[{persona_name}] Attempt {attempt + 1}: JSON decode error: {e}"
                )
                logger.debug(
                    f"[{persona_name}] Attempted to parse: {json_str[:200]}..."
                )
        else:
            logger.warning(
                f"[{persona_name}] Attempt {attempt + 1}: Could not extract JSON from response"
            )

    logger.error(
        f"[{persona_name}] All parsing attempts failed. Raw content (first 500 chars): {content[:500]}"
    )
    return {}


def evaluate_with_persona(
    client: OpenAI,
    persona: Dict,
    query: str,
    ground_truth: str,
    generated_answer: str,
    temperature: float = 0.15,
) -> Dict:
    """Evaluate an answer using a specific persona."""
    user_prompt = f"""Query: {query}

Ground Truth: {ground_truth}

Generated Answer: {generated_answer}

{persona["rubric"]}

### OUTPUT FORMAT:
Provide your reasoning first, then return ONLY a JSON object:
{{
  "reasoning": "Your step-by-step analysis...",
  "{persona["output_key"]}": <int 1-10>,
  "feedback": "Specific improvement suggestions..."
}}

Remember: You are the {persona["name"]} focusing on {persona["focus"]}.
"""

    try:
        response = client.chat.completions.create(
            model="allenai/Molmo2-8B",
            messages=[
                {"role": "system", "content": persona["system_prompt"]},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=600,
        )
        content = response.choices[0].message.content
        parsed = parse_json_with_retry(
            content, max_retries=1, persona_name=persona["name"]
        )

        if not parsed:
            logger.warning(
                f"[{persona['name']}] Parsing returned empty dict. Raw response:\n{content}"
            )

        score_key = persona["output_key"]
        score = parsed.get(score_key, 0)

        if score == 0 and score_key in [
            "correctness",
            "understandability",
            "singapore_relevance",
        ]:
            for alt_key in [f"{score_key}_score", "score", "rating"]:
                if alt_key in parsed and parsed[alt_key] > 0:
                    score = parsed[alt_key]
                    logger.info(
                        f"[{persona['name']}] Used fallback key '{alt_key}' = {score}"
                    )
                    break

        return {
            "score": score,
            "reasoning": parsed.get("reasoning", parsed.get("analysis", "N/A")),
            "feedback": parsed.get("feedback", parsed.get("suggestions", "N/A")),
            "raw_response": content[:500] if content else "N/A",
        }
    except Exception as e:
        logger.error(f"Error evaluating with {persona['name']}: {e}")
        return {
            "score": 0,
            "reasoning": f"Error: {str(e)}",
            "feedback": "Evaluation failed",
            "raw_response": str(e),
        }


def apply_gatekeeper_logic(teacher_score: int, examiner_score: int) -> tuple[bool, str]:
    """Apply gatekeeper logic to determine if answer is a critical failure."""
    if teacher_score < 5:
        return True, f"CRITICAL_FAILURE: Teacher correctness ({teacher_score}) < 5"
    if examiner_score < 4:
        return (
            True,
            f"CRITICAL_FAILURE: Examiner Singapore relevance ({examiner_score}) < 4",
        )
    return False, "PASSED"


def compute_weighted_score(
    understandability: float,
    correctness: float,
    singapore_relevance: float,
    is_critical_failure: bool = False,
) -> float:
    """Compute weighted average score. Weights: Teacher (40%), Examiner (35%), Student (25%)."""
    if is_critical_failure:
        return 0.0

    weighted = (
        correctness * 0.40 + singapore_relevance * 0.35 + understandability * 0.25
    )
    return round(weighted, 2)


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
    if docs == []:
        docs = context_data.get("documents", [])
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

    evaluator_wrapper = llm_factory(
        "allenai/Molmo2-8B",
        client=AsyncOpenAI(
            base_url="https://api.publicai.co/v1", api_key=openrouter_api_key
        ),
        max_tokens=2048,
        temperature=0,
    )

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
    metrics = [ContextPrecision(), ContextRecall()]

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
    evaluator_wrapper = llm_factory(
        "allenai/Molmo2-8B",
        client=AsyncOpenAI(
            base_url="https://api.publicai.co/v1", api_key=openrouter_api_key
        ),
        max_tokens=2048,
        temperature=0,
    )

    from datasets import Dataset as HFDataset

    ragas_data = HFDataset.from_dict(
        {
            "question": eval_df["query"].tolist(),
            "answer": answers,
            "ground_truth": eval_df["ground_truth"].tolist(),
            "contexts": contexts_list,
        }
    )

    metrics = [ContextPrecision(), ContextRecall()]
    ragas_results = evaluate(dataset=ragas_data, metrics=metrics, llm=evaluator_wrapper)
    ragas_df = ragas_results.to_pandas()
    ragas_df.to_csv(ragas_output, index=False)
    logger.info(f"RAGAS results saved to {ragas_output}")

    logger.info(
        "Running Multi-Role Agent Ensemble evaluation (Student, Teacher, Examiner)..."
    )
    logger.info("Using low temperature (0.15) for strict rubric adherence.")
    openrouter_client = OpenAI(
        base_url="https://api.publicai.co/v1", api_key=openrouter_api_key
    )

    judge_results: List[Dict] = []

    for idx, row in eval_df.iterrows():
        try:
            query = str(row["query"])
            ground_truth = str(row["ground_truth"])
            generated_answer = answers[idx]

            persona_results = {}
            for persona in JudgePersona.ALL:
                logger.info(f"  Evaluating with {persona['name']}...")
                result = evaluate_with_persona(
                    client=openrouter_client,
                    persona=persona,
                    query=query,
                    ground_truth=ground_truth,
                    generated_answer=generated_answer,
                    temperature=0.15,
                )
                persona_results[persona["name"]] = result

            student_score = persona_results["STUDENT"]["score"]
            teacher_score = persona_results["TEACHER"]["score"]
            examiner_score = persona_results["EXAMINER"]["score"]

            is_critical_failure, failure_reason = apply_gatekeeper_logic(
                teacher_score=teacher_score,
                examiner_score=examiner_score,
            )

            weighted_score = compute_weighted_score(
                understandability=student_score,
                correctness=teacher_score,
                singapore_relevance=examiner_score,
                is_critical_failure=is_critical_failure,
            )

            judge_results.append(
                {
                    "query": query,
                    "ground_truth": ground_truth,
                    "generated_answer": generated_answer,
                    "understandability": student_score,
                    "correctness": teacher_score,
                    "singapore_relevance": examiner_score,
                    "weighted_score": weighted_score,
                    "is_critical_failure": is_critical_failure,
                    "failure_reason": failure_reason if is_critical_failure else "N/A",
                    "student_reasoning": persona_results["STUDENT"]["reasoning"],
                    "student_feedback": persona_results["STUDENT"]["feedback"],
                    "teacher_reasoning": persona_results["TEACHER"]["reasoning"],
                    "teacher_feedback": persona_results["TEACHER"]["feedback"],
                    "examiner_reasoning": persona_results["EXAMINER"]["reasoning"],
                    "examiner_feedback": persona_results["EXAMINER"]["feedback"],
                }
            )

            status = "CRITICAL_FAILURE" if is_critical_failure else "PASSED"
            logger.info(
                f"Ensemble judge: {int(idx) + 1}/{len(eval_df)} | "
                f"U={student_score} C={teacher_score} SG={examiner_score} | "
                f"Weighted={weighted_score} | {status}"
            )

        except Exception as e:
            logger.error(f"Error on query {idx}: {e}")
            judge_results.append(
                {
                    "query": row["query"],
                    "error": str(e),
                    "is_critical_failure": True,
                }
            )

    judge_df = pd.DataFrame(judge_results)
    judge_df.to_csv(judge_output, index=False)
    logger.info(f"Multi-Role Agent Ensemble results saved to {judge_output}")

    critical_count = sum(
        1 for r in judge_results if r.get("is_critical_failure", False)
    )
    logger.info(
        f"Summary: {critical_count}/{len(judge_results)} answers flagged as CRITICAL_FAILURE"
    )

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
    """Measure time for reranking."""
    start = time.perf_counter()
    rerank(query, docs, top_k=top_k)
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
