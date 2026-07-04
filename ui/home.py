"""Portfolio overview."""

import streamlit as st

from core.signals import build_profile, recommend, ring_signals, run_rules
from core.store import list_applicants, segments_for
from ui.common import badge, inr


def render():
    st.title("🔍 LendLens")
    st.markdown(
        "**Loan-file intelligence for used-car lending.** Messy application bundles in — "
        "evidence-linked signals, policy flags, fraud linkage, and grounded Q&A out."
    )

    applicants = list_applicants()
    segs_by_id = {a["id"]: segments_for(a["id"]) for a in applicants}
    rings = ring_signals(applicants, segs_by_id)

    results = []
    total_flags = 0
    for a in applicants:
        profile = build_profile(a, segs_by_id[a["id"]])
        flags = run_rules(profile)
        rec = recommend(flags, rings, a["id"])
        total_flags += len(flags)
        results.append((a, profile, flags, rec))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files in queue", len(applicants))
    c2.metric("Segments indexed", sum(len(s) for s in segs_by_id.values()))
    c3.metric("Policy flags raised", total_flags)
    c4.metric("Cross-file linkages", len(rings), delta="⚠️ ring risk" if rings else None, delta_color="inverse")

    if rings:
        st.info(
            f"🕸️ **Ring Watch:** {len(rings)} cross-file linkage(s) surfaced across the portfolio — "
            "review the **Ring Watch** page before decisioning any linked file."
        )

    st.divider()

    for a, profile, flags, rec in results:
        with st.container(border=True):
            left, mid, right = st.columns([2.2, 2.2, 1.6])
            with left:
                st.subheader(a["name"])
                st.markdown(badge(rec["verdict"]), unsafe_allow_html=True)
                st.caption(rec["reason"])
            with mid:
                foir = f"{profile['foir']:.0%}" if profile.get("foir") is not None else "—"
                ltv = f"{profile['ltv']:.0%}" if profile.get("ltv") is not None else "—"
                st.caption(
                    f"Loan {inr(profile.get('loan_amount'))} · "
                    f"bank income {inr(profile.get('bank_income'))} · FOIR {foir}"
                )
                st.caption(f"Bounces: {profile.get('bounce_count', 0)} · LTV: {ltv}")
            with right:
                st.metric("Flags", len(flags))

    st.divider()
    st.markdown(
        "##### How it works\n"
        "1. **Ingest** — one messy text bundle per applicant (form narrative, employer letter, bank statement)\n"
        "2. **Extract** — segments are categorized; amounts, orgs, phones, and transactions parsed into a profile\n"
        "3. **Decide** — transparent policy rules raise flags, each pinned to the exact evidence segment\n"
        "4. **Link** — shared phones/employers/accounts across files surface fraud rings rules can't see\n"
        "5. **Verify** — an offline eval harness scores extraction, retrieval, and decision agreement"
    )
