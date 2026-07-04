"""Runtime configuration for LendLens.

Zero-key by default: with no API keys the app runs fully offline
(lexical retrieval + rule-based extraction + cached answers).
Keys unlock LLM extraction/chat and dense embeddings.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = ROOT / "lendlens.db"

# Document categories a loan file is sorted into
DOC_CATEGORIES = [
    "identity",
    "income",
    "employment",
    "banking",
    "obligations",
    "vehicle",
    "declarations",
    "policy",
]

# Underwriting thresholds (mirrors data/policy.md — keep in sync)
POLICY = {
    "foir_max": 0.55,          # (existing EMIs + proposed EMI) / net income
    "bounce_max_12m": 2,       # inward return limit in last 12 months
    "income_mismatch_max": 0.25,  # |stated − bank-derived| / stated
    "min_tenure_months": 12,   # employment tenure floor
    "ltv_max": {"0-3": 0.90, "4-7": 0.80, "8+": 0.70},  # by vehicle age (years)
    "min_monthly_income": 20000,
}

PROPOSED_RATE = 0.145  # annual, used to estimate proposed EMI
PROPOSED_TENURE_MONTHS = 48


# Claude model for RAG synthesis (override with ANTHROPIC_MODEL in .env)
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")


def anthropic_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY") or None


def openai_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY") or None


def live_mode() -> bool:
    """True when an LLM is available for generation/extraction."""
    return anthropic_key() is not None


def dense_available() -> bool:
    """True when dense embeddings can be computed."""
    return openai_key() is not None
