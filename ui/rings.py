"""Ring Watch — cross-file linkage detection."""

import streamlit as st

from core.signals import ring_signals
from core.store import get_applicant, list_applicants, segments_for


def render():
    st.title("🕸️ Ring Watch")
    st.caption(
        "Fraud rings reuse infrastructure: phone numbers, employers, accounts. "
        "Rules score files one at a time — linkage looks **across** the portfolio. "
        "Any shared attribute between unrelated applications routes every linked "
        "file to the fraud desk (POL-7.1)."
    )

    applicants = list_applicants()
    segs_by_id = {a["id"]: segments_for(a["id"]) for a in applicants}
    hits = ring_signals(applicants, segs_by_id)

    if not hits:
        st.success("No shared attributes across the current portfolio.")
        return

    st.warning(
        f"🕸️ **{len(hits)} shared attribute(s) found across applications** — "
        "all linked files are routed to the fraud desk (POL-7.1)."
    )

    for h in hits:
        with st.container(border=True):
            names = " ↔ ".join(get_applicant(a)["name"] for a in h["applicants"])
            st.markdown(f"**Shared {h['kind']}:** `{h['value']}`")
            st.markdown(f"Links: **{names}**")
            seg_lookup = {
                s["id"]: (get_applicant(aid)["name"], s)
                for aid in h["applicants"]
                for s in segs_by_id[aid]
            }
            with st.expander(f"Evidence ({len(h['evidence'])} segments)"):
                for sid in h["evidence"]:
                    if sid in seg_lookup:
                        owner, seg = seg_lookup[sid]
                        st.markdown(f"**`{sid}`** — {owner}\n\n> {seg['text'][:300]}…")

    st.divider()
    st.markdown(
        "**Why this matters:** each linked file can look individually plausible — "
        "one is even approve-with-conditions on pure rules. The linkage is only "
        "visible at portfolio level, which is exactly where rings operate."
    )
