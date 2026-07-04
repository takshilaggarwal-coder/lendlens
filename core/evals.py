"""Evaluation harness — runs entirely offline.

Three layers, mirroring how the system can fail:

  1. extraction — do derived signals match hand-labelled gold values?
  2. retrieval  — do the right segments come back for known questions?
                  (hit@3, hit@5, MRR)
  3. decision   — does the rule engine reproduce the gold verdict?
"""

from core.retrieval import HybridRetriever
from core.signals import build_profile, recommend, ring_signals, run_rules
from core.store import eval_questions_for, list_applicants, segments_for

_FIELD_TOLERANCE = 0.02  # 2% numeric tolerance


def extraction_report(applicant: dict, profile: dict) -> list[dict]:
    gold = applicant.get("meta", {}).get("gold", {})
    rows = []
    for field, expected in gold.get("fields", {}).items():
        got = profile.get(field)
        ok = _match(expected, got)
        rows.append({"field": field, "expected": expected, "extracted": got, "ok": ok})
    return rows


def _match(expected, got) -> bool:
    if got is None:
        return False
    try:
        e, g = float(expected), float(got)
        if e == 0:
            return g == 0
        return abs(e - g) / abs(e) <= _FIELD_TOLERANCE
    except (TypeError, ValueError):
        return str(expected).strip().lower() == str(got).strip().lower()


def retrieval_report(applicant_id: str, segments: list[dict], policy_segments: list[dict]) -> dict:
    questions = eval_questions_for(applicant_id)
    if not questions:
        return {"questions": [], "hit3": None, "hit5": None, "mrr": None}
    retriever = HybridRetriever(segments + policy_segments)
    rows, rr_sum, h3, h5 = [], 0.0, 0, 0
    for q in questions:
        hits = retriever.retrieve(q["question"], k=5)
        ids = [h["id"] for h in hits]
        rank = next((i + 1 for i, sid in enumerate(ids) if sid in q["gold"]), None)
        rr_sum += (1 / rank) if rank else 0.0
        h3 += 1 if rank and rank <= 3 else 0
        h5 += 1 if rank else 0
        rows.append({"question": q["question"], "gold": q["gold"], "top5": ids, "rank": rank})
    n = len(questions)
    return {"questions": rows, "hit3": h3 / n, "hit5": h5 / n, "mrr": rr_sum / n}


def portfolio_report() -> dict:
    """Run all three layers across every applicant in the store."""
    applicants = list_applicants()
    policy_segments = segments_for(None)
    segs_by_id = {ap["id"]: segments_for(ap["id"]) for ap in applicants}
    rings = ring_signals(applicants, segs_by_id)

    per_applicant, ext_total, ext_ok = [], 0, 0
    ret_metrics = []
    decisions_ok, decisions_total = 0, 0

    for ap in applicants:
        segs = segs_by_id[ap["id"]]
        profile = build_profile(ap, segs)
        flags = run_rules(profile)
        rec = recommend(flags, rings, ap["id"])

        ext = extraction_report(ap, profile)
        ext_total += len(ext)
        ext_ok += sum(1 for r in ext if r["ok"])

        ret = retrieval_report(ap["id"], segs, policy_segments)
        if ret["mrr"] is not None:
            ret_metrics.append(ret)

        gold_verdict = ap.get("meta", {}).get("gold", {}).get("verdict")
        verdict_ok = None
        if gold_verdict:
            decisions_total += 1
            verdict_ok = rec["verdict"] == gold_verdict
            decisions_ok += 1 if verdict_ok else 0

        per_applicant.append({
            "applicant": ap["name"],
            "applicant_id": ap["id"],
            "extraction": ext,
            "retrieval": ret,
            "flags": flags,
            "recommendation": rec,
            "gold_verdict": gold_verdict,
            "verdict_ok": verdict_ok,
        })

    def _avg(key):
        vals = [m[key] for m in ret_metrics if m[key] is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "per_applicant": per_applicant,
        "rings": rings,
        "summary": {
            "extraction_accuracy": ext_ok / ext_total if ext_total else None,
            "hit3": _avg("hit3"),
            "hit5": _avg("hit5"),
            "mrr": _avg("mrr"),
            "decision_agreement": decisions_ok / decisions_total if decisions_total else None,
        },
    }
