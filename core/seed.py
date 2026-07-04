"""Seed the store with the bundled demo portfolio + policy corpus.

Gold segment references in the demo data are written as text substrings
("gold_contains") and resolved to segment IDs at ingest time, so chunking
changes never silently break the eval set.
"""

import json

from core.config import DATA_DIR
from core.pipeline import run_pipeline
from core.store import (
    init_db,
    list_applicants,
    reset_db,
    save_answers,
    save_applicant,
    save_eval_questions,
    save_segments,
    segments_for,
)


def _resolve(substrings: list[str], segments: list[dict]) -> list[str]:
    ids = []
    for s in segments:
        if any(sub in s["text"] for sub in substrings):
            ids.append(s["id"])
    return ids


def seed(force: bool = False) -> None:
    init_db()
    if list_applicants() and not force:
        return
    reset_db()

    # policy corpus (applicant_id = NULL)
    policy_text = (DATA_DIR / "policy.md").read_text(encoding="utf-8")
    policy_segments = run_pipeline(policy_text, id_prefix="pol")
    for s in policy_segments:
        s["category"] = "policy"
    save_segments(None, policy_segments)

    portfolio = json.loads((DATA_DIR / "portfolio.json").read_text(encoding="utf-8"))
    for ap in portfolio["applicants"]:
        save_applicant(
            ap["name"],
            meta={"stated": ap["stated"], "gold": ap["gold"]},
            is_demo=True,
            applicant_id=ap["id"],
        )
        segments = run_pipeline(ap["documents"], id_prefix=ap["prefix"])
        save_segments(ap["id"], segments)

        searchable = segments + policy_segments
        save_answers(
            ap["id"],
            [
                {
                    "question": c["question"],
                    "answer": c["answer"],
                    "sources": _resolve(c.get("source_contains", []), searchable),
                }
                for c in ap.get("cached_answers", [])
            ],
        )
        save_eval_questions(
            ap["id"],
            [
                {
                    "question": q["question"],
                    "gold": _resolve(q.get("gold_contains", []), searchable),
                    "category": q.get("category"),
                }
                for q in ap.get("eval_questions", [])
            ],
        )


def policy_corpus() -> list[dict]:
    return segments_for(None)


if __name__ == "__main__":
    seed(force=True)
    print(f"Seeded {len(list_applicants())} applicants, {len(policy_corpus())} policy segments.")
