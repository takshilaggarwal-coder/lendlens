"""Shared UI helpers."""

import streamlit as st

from core.store import list_applicants

VERDICT_STYLE = {
    "approve": ("Approve", "#0f7b3e", "#e6f6ec"),
    "approve_conditions": ("Approve with conditions", "#0b6e6e", "#e2f4f4"),
    "refer": ("Refer to credit manager", "#a15c00", "#fdf0dd"),
    "decline": ("Decline", "#b3261e", "#fdeaea"),
    "decline_refer_fraud": ("Fraud desk — do not decision", "#7a1fa2", "#f5e8fb"),
}

SEVERITY_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}


def badge(verdict: str) -> str:
    label, fg, bg = VERDICT_STYLE.get(verdict, (verdict, "#444", "#eee"))
    return f'<span class="ll-badge" style="color:{fg};background:{bg};">{label}</span>'


def applicant_picker(key: str = "applicant") -> dict | None:
    applicants = list_applicants()
    if not applicants:
        st.info("No applications yet — add one on the **New Application** page.")
        return None
    if "selected_applicant" not in st.session_state:
        st.session_state.selected_applicant = applicants[0]["id"]
    ids = [a["id"] for a in applicants]
    default = st.session_state.selected_applicant
    idx = ids.index(default) if default in ids else 0
    choice = st.selectbox(
        "Application",
        applicants,
        index=idx,
        format_func=lambda a: a["name"],
        key=key,
    )
    st.session_state.selected_applicant = choice["id"]
    return choice


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander(f"📎 Evidence — {len(citations)} segment(s)"):
        for c in citations:
            match = f" · matched by {', '.join(c['matched_by'])}" if c.get("matched_by") else ""
            st.markdown(
                f"**`{c['id']}`** · _{c.get('category', 'file')}_{match}\n\n> {c['snippet']}"
            )


def inr(x) -> str:
    if x is None:
        return "—"
    return f"₹{x:,.0f}"
