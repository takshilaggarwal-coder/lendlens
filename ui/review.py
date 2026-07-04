"""File Review — profile, flags with evidence, recommendation."""

import streamlit as st

from core.signals import build_profile, recommend, ring_signals, run_rules
from core.store import list_applicants, segments_for
from ui.common import SEVERITY_ICON, applicant_picker, badge, inr


def render():
    st.title("📄 File Review")
    applicant = applicant_picker("review_picker")
    if not applicant:
        return

    segments = segments_for(applicant["id"])
    profile = build_profile(applicant, segments)

    applicants = list_applicants()
    segs_by_id = {a["id"]: segments_for(a["id"]) for a in applicants}
    rings = ring_signals(applicants, segs_by_id)
    flags = run_rules(profile)
    rec = recommend(flags, rings, applicant["id"])

    st.markdown(badge(rec["verdict"]), unsafe_allow_html=True)
    st.caption(rec["reason"])

    # ---------------- profile card ----------------
    st.subheader("Extracted profile")
    r1 = st.columns(4)
    r1[0].metric("Stated income", inr(profile.get("stated_income")))
    delta = None
    si, bi = profile.get("stated_income"), profile.get("bank_income")
    if si and bi:
        delta = f"{(bi - si) / si:+.1%} vs stated"
    r1[1].metric("Bank-derived income", inr(bi), delta=delta, delta_color="off")
    r1[2].metric("Existing EMIs", inr(profile.get("existing_emi_total")))
    r1[3].metric("Proposed EMI", inr(profile.get("proposed_emi")))

    r2 = st.columns(4)
    r2[0].metric("FOIR", f"{profile['foir']:.1%}" if profile.get("foir") is not None else "—")
    r2[1].metric("LTV", f"{profile['ltv']:.1%}" if profile.get("ltv") is not None else "—")
    r2[2].metric("Bounces", profile.get("bounce_count", 0))
    r2[3].metric("Tenure (months)", profile.get("tenure_months") or "—")

    r3 = st.columns(4)
    r3[0].metric("Avg balance", inr(profile.get("avg_balance")))
    r3[1].metric("Min balance", inr(profile.get("min_balance")))
    r3[2].metric("Transactions parsed", profile.get("txn_count", 0))
    r3[3].metric("Segments", len(segments))

    # ---------------- flags ----------------
    st.subheader(f"Policy flags ({len(flags)})")
    if not flags:
        st.success("No policy flags — file is within all thresholds.")
    seg_index = {s["id"]: s for s in segments}
    for f in flags:
        icon = SEVERITY_ICON.get(f["severity"], "🟡")
        with st.container(border=True):
            st.markdown(f"{icon} **{f['id']}** · {f['finding']}")
            if f.get("policy_ref"):
                st.caption(f"Policy: {f['policy_ref']}")
            evidence = [e for e in f.get("evidence", []) if e]
            if evidence:
                with st.expander(f"Evidence ({len(evidence)})"):
                    for eid in evidence:
                        seg = seg_index.get(eid)
                        if seg:
                            st.markdown(f"**`{eid}`** · _{seg.get('category')}_\n\n> {seg['text'][:400]}")

    if rec["ring"]:
        st.warning(
            "🕸️ This file is part of a **cross-file linkage** — see Ring Watch. "
            "Fraud review precedes any credit decision, whatever the rule results say."
        )

    # ---------------- raw segments ----------------
    with st.expander(f"Browse all {len(segments)} extracted segments"):
        for s in segments:
            ents = s.get("entities", {})
            chips = " · ".join(
                filter(None, [
                    f"orgs: {', '.join(ents['orgs'])}" if ents.get("orgs") else None,
                    f"phones: {len(ents.get('phones', []))}" if ents.get("phones") else None,
                    f"txns: {ents.get('txn_count', 0)}" if ents.get("txn_count") else None,
                ])
            )
            st.markdown(f"**`{s['id']}`** · _{s.get('category')}_ {('· ' + chips) if chips else ''}")
            st.text(s["text"][:500])
