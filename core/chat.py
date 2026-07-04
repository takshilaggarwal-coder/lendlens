"""Grounded Q&A over a loan file + policy corpus.

Every answer carries citations (segment IDs + snippets). Two paths:

  live mode  — Claude answers from retrieved segments only, instructed to
               cite [seg_id] inline and admit gaps
  demo mode  — nearest cached answer when confident, else an extractive
               reply built from the top retrieved segments

Both paths run behind the fairness guardrail.
"""

from core import guardrail
from core.config import live_mode
from core.retrieval import BM25, HybridRetriever
from core.store import answers_for


def _mask_pii(text: str) -> str:
    """Mask long digit runs (accounts/phones) in displayed snippets."""
    import re

    return re.sub(r"\b(\d{4,6})(\d{4})\b", lambda m: "•" * len(m.group(1)) + m.group(2), text)


def _citations(segs: list[dict]) -> list[dict]:
    return [
        {
            "id": s["id"],
            "category": s.get("category"),
            "snippet": _mask_pii(s["text"][:260] + ("…" if len(s["text"]) > 260 else "")),
            "score": s.get("score"),
            "matched_by": s.get("matched_by", []),
        }
        for s in segs
    ]


def answer(question: str, file_segments: list[dict], policy_segments: list[dict],
           applicant_id: str | None = None) -> dict:
    """Answer a question grounded in the applicant file + policy corpus."""
    verdict = guardrail.check(question)
    if verdict and verdict["level"] == "block":
        return {
            "text": guardrail.block_message(verdict["groups"]),
            "citations": [],
            "mode": "guardrail",
        }

    retriever = HybridRetriever(file_segments + policy_segments)
    hits = retriever.retrieve(question, k=5)

    note = ""
    if verdict and verdict["level"] == "note":
        note = (
            "_Note: I've ignored the protected-attribute part of the question; "
            "answering on financial signals only._\n\n"
        )

    if live_mode():
        return {"text": note + _answer_llm(question, hits), "citations": _citations(hits), "mode": "live"}

    cached = _nearest_cached(question, applicant_id)
    if cached:
        return {
            "text": note + cached["answer"],
            "citations": _citations([s for s in file_segments + policy_segments if s["id"] in cached["sources"]]) or _citations(hits[:3]),
            "mode": "cached",
        }

    return {"text": note + _extractive(question, hits), "citations": _citations(hits[:3]), "mode": "extractive"}


# ------------------------------------------------------------- demo answers

_CACHE_CONFIDENCE = 6.0  # BM25 score floor for reusing a cached answer


def _nearest_cached(question: str, applicant_id: str | None) -> dict | None:
    if not applicant_id:
        return None
    cached = answers_for(applicant_id)
    if not cached:
        return None
    index = BM25([c["question"] for c in cached])
    scores = index.scores(question)
    best = max(range(len(scores)), key=lambda i: scores[i])
    return cached[best] if scores[best] >= _CACHE_CONFIDENCE else None


def _extractive(question: str, hits: list[dict]) -> str:
    if not hits:
        return "Nothing in this file matches that question. Try asking about income, EMIs, banking conduct, the vehicle, or policy thresholds."
    lines = ["Here's what the file says (closest passages first):", ""]
    for s in hits[:3]:
        label = s.get("category", "file")
        lines.append(f"- **[{s['id']}]** ({label}) “{_mask_pii(s['text'][:220])}…”")
    lines.append("")
    lines.append("_Offline demo mode: showing evidence extractively. Add an `ANTHROPIC_API_KEY` for synthesized answers over these same citations._")
    return "\n".join(lines)


# -------------------------------------------------------------- live answer

def _answer_llm(question: str, hits: list[dict]) -> str:
    import anthropic

    evidence = "\n\n".join(f"[{s['id']}] ({s.get('category')}) {s['text']}" for s in hits)
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=700,
        system=(
            "You are an underwriting copilot for used-car loans. Answer ONLY from "
            "the evidence segments provided. Cite segment ids inline like [seg_012] "
            "after each claim. If the evidence doesn't contain the answer, say so "
            "plainly — never invent figures. Keep answers under 150 words, precise, "
            "numbers formatted with ₹ and commas."
        ),
        messages=[{"role": "user", "content": f"Evidence:\n{evidence}\n\nQuestion: {question}"}],
    )
    return msg.content[0].text
