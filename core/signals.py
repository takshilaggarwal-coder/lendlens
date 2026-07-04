"""Signal extraction + underwriting rules.

Turns categorized segments into a structured applicant profile, then runs
transparent policy rules. Every derived number and every red flag carries
the segment IDs it was computed from — evidence-linked decisioning, not a
black box.
"""

import re
import statistics

from core.config import POLICY, PROPOSED_RATE, PROPOSED_TENURE_MONTHS
from core.pipeline import parse_transactions

_SALARY_HINTS = ("salary", "sal ", "payroll", "wages")
_EMI_HINTS = ("emi", "ach d", "loan installment", "nach")
_BOUNCE_HINTS = ("return", "bounce", "insufficient", "reversal chg")


# ------------------------------------------------------------- txn analysis

def analyse_statement(segments: list[dict]) -> dict:
    """Aggregate bank behaviour from all statement segments."""
    txns, src = [], []
    for s in segments:
        if s.get("category") != "banking":
            continue
        parsed = parse_transactions(s["text"])
        if parsed:
            txns.extend(parsed)
            src.append(s["id"])

    salary_credits = [t for t in txns if t["amount"] > 0 and any(h in t["desc"].lower() for h in _SALARY_HINTS)]
    emi_debits = [t for t in txns if t["amount"] < 0 and any(h in t["desc"].lower() for h in _EMI_HINTS)]
    bounces = [t for t in txns if any(h in t["desc"].lower() for h in _BOUNCE_HINTS)]
    balances = [t["balance"] for t in txns]

    monthly_emis = {}
    for t in emi_debits:
        key = t["desc"].lower()[:18]
        monthly_emis[key] = max(monthly_emis.get(key, 0), -t["amount"])

    return {
        "txn_count": len(txns),
        "salary_credits": [t["amount"] for t in salary_credits],
        "bank_income": statistics.mean([t["amount"] for t in salary_credits]) if salary_credits else 0.0,
        "existing_emi_total": sum(monthly_emis.values()),
        "emi_lines": len(monthly_emis),
        "bounce_count": len(bounces),
        "avg_balance": statistics.mean(balances) if balances else 0.0,
        "min_balance": min(balances) if balances else 0.0,
        "sources": src,
    }


# ------------------------------------------------------------ profile build

def _first_match(segments, category, pattern):
    rx = re.compile(pattern, re.I)
    for s in segments:
        if s.get("category") != category:
            continue
        m = rx.search(s["text"])
        if m:
            return _to_num(m.group(1)), s["id"]
    # fall back to any category
    for s in segments:
        m = rx.search(s["text"])
        if m:
            return _to_num(m.group(1)), s["id"]
    return None, None


def _to_num(s):
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return None


_TENURE_RE = re.compile(
    r"(?:for|since|past|tenure of)\s+(?:the\s+past\s+)?(\d+)\s+(months?|years?)", re.I
)


def _tenure(segments) -> tuple[int | None, str | None]:
    """Employment tenure in months, with the unit read from the text."""
    for pool in (
        [s for s in segments if s.get("category") == "employment"],
        segments,
    ):
        for s in pool:
            m = _TENURE_RE.search(s["text"])
            if m:
                n, unit = int(m.group(1)), m.group(2).lower()
                return (n * 12 if unit.startswith("year") else n), s["id"]
    return None, None


def build_profile(applicant: dict, segments: list[dict]) -> dict:
    """Merge stated fields (from the application narrative) with bank-derived truth."""
    stated = applicant.get("meta", {}).get("stated", {})
    bank = analyse_statement(segments)

    stated_income, inc_src = _first_match(
        segments, "income", r"(?:take-home|in-hand|net(?:\s+monthly)?)\s+(?:salary|income|pay)\s+(?:of\s+)?(?:₹|Rs\.?\s?)([\d,]+)"
    )
    if stated_income is None:
        stated_income = stated.get("monthly_income")

    tenure_months, tenure_src = stated.get("tenure_months"), None
    if tenure_months is None:
        tenure_months, tenure_src = _tenure(segments)

    loan_amount = stated.get("loan_amount")
    vehicle_price = stated.get("vehicle_price")
    vehicle_age = stated.get("vehicle_age_years", 4)

    proposed_emi = _emi(loan_amount) if loan_amount else 0.0
    net_income = bank["bank_income"] or (stated_income or 0)
    foir = (bank["existing_emi_total"] + proposed_emi) / net_income if net_income else None
    ltv = loan_amount / vehicle_price if loan_amount and vehicle_price else None

    return {
        "stated_income": stated_income,
        "stated_income_src": inc_src,
        "bank_income": round(bank["bank_income"], 0),
        "bank_sources": bank["sources"],
        "salary_credits": bank["salary_credits"],
        "existing_emi_total": round(bank["existing_emi_total"], 0),
        "emi_lines": bank["emi_lines"],
        "bounce_count": bank["bounce_count"],
        "avg_balance": round(bank["avg_balance"], 0),
        "min_balance": round(bank["min_balance"], 0),
        "txn_count": bank["txn_count"],
        "tenure_months": tenure_months,
        "tenure_src": tenure_src,
        "loan_amount": loan_amount,
        "vehicle_price": vehicle_price,
        "vehicle_age_years": vehicle_age,
        "proposed_emi": round(proposed_emi, 0),
        "foir": round(foir, 3) if foir is not None else None,
        "ltv": round(ltv, 3) if ltv is not None else None,
    }


def _emi(principal: float, rate=PROPOSED_RATE, months=PROPOSED_TENURE_MONTHS) -> float:
    r = rate / 12
    return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)


# ----------------------------------------------------------------- rules

def run_rules(profile: dict) -> list[dict]:
    """Transparent policy checks. Each flag: id, severity, finding, evidence."""
    flags = []

    def flag(fid, severity, finding, evidence=None, policy_ref=None):
        flags.append({
            "id": fid, "severity": severity, "finding": finding,
            "evidence": evidence or [], "policy_ref": policy_ref,
        })

    si, bi = profile.get("stated_income"), profile.get("bank_income")
    if si and bi:
        drift = abs(si - bi) / si
        if drift > POLICY["income_mismatch_max"]:
            flag(
                "income_mismatch", "high",
                f"Stated take-home ₹{si:,.0f} vs bank-derived ₹{bi:,.0f} ({drift:.0%} apart).",
                [profile.get("stated_income_src")] + profile.get("bank_sources", []),
                "POL-4.2 income verification",
            )

    foir = profile.get("foir")
    if foir is not None and foir > POLICY["foir_max"]:
        flag(
            "foir_breach", "high",
            f"FOIR {foir:.0%} exceeds the {POLICY['foir_max']:.0%} ceiling "
            f"(existing EMIs ₹{profile['existing_emi_total']:,.0f} + proposed ₹{profile['proposed_emi']:,.0f}).",
            profile.get("bank_sources", []),
            "POL-5.1 repayment capacity",
        )

    if profile.get("bounce_count", 0) > POLICY["bounce_max_12m"]:
        flag(
            "bounce_history", "high",
            f"{profile['bounce_count']} inward returns/bounces in the statement window (limit {POLICY['bounce_max_12m']}).",
            profile.get("bank_sources", []),
            "POL-5.3 banking conduct",
        )

    credits = profile.get("salary_credits", [])
    if len(credits) >= 3 and all(c % 1000 == 0 for c in credits):
        flag(
            "round_credits", "medium",
            "Every salary credit is a perfectly round figure — payroll credits rarely are. "
            "Pattern consistent with staged/self-funded deposits.",
            profile.get("bank_sources", []),
            "POL-7.1 fraud indicators",
        )

    tm = profile.get("tenure_months")
    if tm is not None and tm < POLICY["min_tenure_months"]:
        flag(
            "thin_tenure", "medium",
            f"Employment tenure {tm} months is below the {POLICY['min_tenure_months']}-month floor.",
            [profile.get("tenure_src")],
            "POL-3.2 employment stability",
        )

    ltv, age = profile.get("ltv"), profile.get("vehicle_age_years", 4)
    band = "0-3" if age <= 3 else ("4-7" if age <= 7 else "8+")
    if ltv is not None and ltv > POLICY["ltv_max"][band]:
        flag(
            "ltv_breach", "medium",
            f"LTV {ltv:.0%} exceeds {POLICY['ltv_max'][band]:.0%} cap for a {age}-year-old vehicle.",
            [],
            "POL-6.2 loan-to-value grid",
        )

    if bi and bi < POLICY["min_monthly_income"]:
        flag(
            "income_floor", "high",
            f"Bank-derived income ₹{bi:,.0f} is below the ₹{POLICY['min_monthly_income']:,} program floor.",
            profile.get("bank_sources", []),
            "POL-4.1 minimum income",
        )

    return [f for f in flags if f]


# ---------------------------------------------------- cross-file ring check

def ring_signals(all_applicants: list[dict], segments_by_id: dict[str, list[dict]]) -> list[dict]:
    """Shared-attribute detection across the portfolio.

    Fraud rings reuse infrastructure: a phone number, an employer, a
    reference. Exact-match linkage across files is cheap and catches the
    clumsy majority.
    """
    seen: dict[tuple, list[tuple[str, str]]] = {}
    for ap in all_applicants:
        for seg in segments_by_id.get(ap["id"], []):
            ents = seg.get("entities", {})
            for phone in ents.get("phones", []):
                seen.setdefault(("phone", phone), []).append((ap["id"], seg["id"]))
            for org in ents.get("orgs", []):
                seen.setdefault(("employer", org.lower()), []).append((ap["id"], seg["id"]))
            for acct in ents.get("accounts", []):
                seen.setdefault(("account", acct), []).append((ap["id"], seg["id"]))

    hits = []
    for (kind, value), refs in seen.items():
        applicants = sorted({a for a, _ in refs})
        if len(applicants) > 1:
            display = value if kind != "phone" else f"…{value[-4:]}"
            hits.append({
                "kind": kind,
                "value": display,
                "applicants": applicants,
                "evidence": sorted({s for _, s in refs}),
            })
    return hits


# ------------------------------------------------------------ recommendation

def recommend(flags: list[dict], ring_hits: list[dict], applicant_id: str) -> dict:
    involved_in_ring = any(applicant_id in h["applicants"] for h in ring_hits)
    highs = [f for f in flags if f["severity"] == "high"]
    mediums = [f for f in flags if f["severity"] == "medium"]

    if involved_in_ring:
        verdict, reason = "decline_refer_fraud", "Cross-file linkage detected — route to fraud review before any credit decision."
    elif len(highs) >= 2:
        verdict, reason = "decline", "Multiple high-severity policy breaches."
    elif len(highs) == 1:
        verdict, reason = "refer", "One high-severity breach — needs credit-manager review with the cited evidence."
    elif mediums:
        verdict, reason = "approve_conditions", "Within policy except minor exceptions; approve with conditions (e.g. lower LTV or co-applicant)."
    else:
        verdict, reason = "approve", "Clean file — within all policy thresholds."

    return {"verdict": verdict, "reason": reason, "ring": involved_in_ring}
