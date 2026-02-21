"""
Configuration constants for the H2 Economics RAG Chatbot.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Model & infrastructure
# ---------------------------------------------------------------------------
LLM_MODEL = "allenai/Molmo2-8B"
API_BASE_URL = "https://api.publicai.co/v1"
API_KEY = os.getenv("PUBLICAI_API_KEY")
USE_RAG = (
    True  # If False, bypass retrieval entirely and use model's parametric knowledge
)
RERANK = True  # Whether to perform reranking of retrieved documents
JINA_RERANK = True  # If True, use Jina reranker; if False, use LLM-based reranking
RERANK_API_ENDPOINT = "https://api.jina.ai/v1/rerank"
RERANK_API_KEY = os.getenv("JINA_API_KEY")
RERANK_MODEL = "jina-reranker-v2-base-multilingual"

EMBEDDING_MODEL = "default"
TOP_K = 5  # Number of documents to return to the model
N_FILTER_DOCS = None  # Number of documents to consider for reranking

SQLITE_DB_PATH = "h2_economics_library.db"
CHROMA_DB_PATH = "chroma_db_h2econs"
CHROMA_COLLECTION = "econ_notes"

# ---------------------------------------------------------------------------
# Schools
# ---------------------------------------------------------------------------
SCHOOLS: list[str] = [
    "St. Andrew's Junior College (SAJC)",
    "Hwa Chong Institution (HCI)",
    "Anglo-Chinese Junior College (ACJC)",
    "Eunoia Junior College (EJC)",
    "Nanyang Junior College (NYJC)",
    "Raffles Institution (RI)",
]

# Quick lookup: acronym / short‐form → canonical full name
SCHOOL_ALIASES: dict[str, str] = {
    "SAJC": "St. Andrew's Junior College (SAJC)",
    "SA": "St. Andrew's Junior College (SAJC)",
    "HCI": "Hwa Chong Institution (HCI)",
    "HC": "Hwa Chong Institution (HCI)",
    "HWA CHONG": "Hwa Chong Institution (HCI)",
    "ACJC": "Anglo-Chinese Junior College (ACJC)",
    "AC": "Anglo-Chinese Junior College (ACJC)",
    "EJC": "Eunoia Junior College (EJC)",
    "EUNOIA": "Eunoia Junior College (EJC)",
    "NYJC": "Nanyang Junior College (NYJC)",
    "NY": "Nanyang Junior College (NYJC)",
    "NANYANG": "Nanyang Junior College (NYJC)",
    "RI": "Raffles Institution (RI)",
    "RAFFLES": "Raffles Institution (RI)",
}

# ---------------------------------------------------------------------------
# H2 Economics topic categories (match DB values exactly)
# ---------------------------------------------------------------------------
H2_ECON_TOPICS: list[str] = [
    "The Central Economic Problem (Scarcity, PPC, Decision-making)",
    "Markets: Demand, Supply, and Elasticities",
    "Firms: Cost, Revenue, and Profit Maximisation",
    "Firms: Market Structures and Strategies",
    "Market Failure and Government Intervention",
    "Intro to Macroeconomics (AD/AS and Circular Flow)",
    "Macroeconomic Performance and Standard of Living",
    "Macroeconomic Policies (Fiscal, Monetary, Supply-side)",
    "The International Economy (Trade, BOP, Exchange Rates, Globalisation)",
]

# Evaluation
NUM_JUDGES = 3
JUDGE_TEMPERATURE = 0.9
