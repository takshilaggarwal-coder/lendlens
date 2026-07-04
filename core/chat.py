"""Grounded Q&A over a loan file + policy corpus — RAG branch (rag-anthropic).

Every question runs a retrieval-augmented generation loop:

  1. the fairness guardrail screens the question (hard filter, pre-retrieval)
  2. the hybrid retriever pulls the top-k evidence segments (BM25 + RRF,
     dense embeddings joining the fusion when an OpenAI key is present)
  3. Claude synthesizes the answer from those segments ONLY — it never sees
     the raw file — citing [seg_id] inline and admitting gaps

Requires ANTHROPIC_API_KEY (copy .env.example to .env). Without a key the
module degrades to an extractive fallback so nothing crashes, but this
branch is meant to run live; use `main` for the zero-key offline demo.
"""

from core import guardrail
from core.config import ANTHROPIC_MODEL, live_mode
from core.retrieval import HybridRetriever

_TOP_K = 8        # evidence segments handed to the model
_MAX_TOKENS = 700  # answer budget

_SYSTEM = (
    "You are an underwriting copilot for used-car loans at a regulated NBFC. "
    "Answer ONLY from the numbered evidence segments provided — they are the "
    "sole source of truth. Cite segment ids inline like [seg_012] after every "
    "claim. If the evidence does not contain the answer, say exactly what is "
    "missing — never estimate or invent figures. Format numbers with ₹ and "
    "Indian comma grouping. Protected attributes (religion, caste, gender, "
    "marital status, community) must never influence any statement about "
    "creditworthiness. Keep answers under 150 words."
)


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
    hits = retriever.retrieve(question, k=_TOP_K)

    note = ""
    if verdict and verdict["level"] == "note":
        note = (
            "_Note: I've ignored the protected-attribute part of the question; "
            "answering on financial signals only._\n\n"
        )

    if not live_mode():
        return {
            "text": note + _extractive(question, hits),
            "citations": _citations(hits[:3]),
            "mode": "extractive",
        }

    try:
        return {"text": note + _answer_llm(question, hits), "citations": _citations(hits), "mode": "live"}
    except Exception as exc:  # degrade gracefully — never crash the UI on an API hiccup
        fallback = (
            f"_Live answer unavailable ({exc.__class__.__name__}) — showing retrieved evidence instead._\n\n"
            + _extractive(question, hits)
        )
        return {"text": note + fallback, "citations": _citations(hits[:3]), "mode": "extractive"}


# ------------------------------------------------------------ RAG synthesis

def _answer_llm(question: str, hits: list[dict]) -> str:
    import anthropic

    evidence = "\n\n".join(f"[{s['id']}] ({s.get('category', 'file')}) {s['text']}" for s in hits)
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=_MAX_TOKENS,
        system=_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Evidence segments:\n\n{evidence}\n\nUnderwriter's question: {question}",
        }],
    )
    return msg.content[0].text


# ------------------------------------------------------- extractive fallback

def _extractive(question: str, hits: list[dict]) -> str:
    if not hits:
        return "Nothing in this file matches that question. Try asking about income, EMIs, banking conduct, the vehicle, or policy thresholds."
    lines = ["Here's what the file says (closest passages first):", ""]
    for s in hits[:3]:
        label = s.get("category", "file")
        lines.append(f"- **[{s['id']}]** ({label}) “{_mask_pii(s['text'][:220])}…”")
    lines.append("")
    lines.append("_This branch runs Q&A as live RAG — set `ANTHROPIC_API_KEY` in `.env` for synthesized answers over these same citations._")
    return "\n".join(lines)
