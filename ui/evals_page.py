"""Evals — extraction accuracy, retrieval quality, decision agreement."""

import pandas as pd
import plotly.express as px
import streamlit as st

from core.evals import portfolio_report


def render():
    st.title("📊 Evals")
    st.caption(
        "The harness runs three layers offline: field extraction vs hand-labelled gold, "
        "retrieval hit@k / MRR on known questions, and rule-engine verdicts vs expected outcomes. "
        "If you can't measure the pipeline, you can't ship it."
    )

    if st.button("Run eval harness", type="primary"):
        st.session_state["eval_report"] = portfolio_report()

    report = st.session_state.get("eval_report")
    if not report:
        st.info("Press **Run eval harness** to score the portfolio.")
        return

    s = report["summary"]
    c = st.columns(5)
    c[0].metric("Extraction accuracy", _pct(s["extraction_accuracy"]))
    c[1].metric("Retrieval hit@3", _pct(s["hit3"]))
    c[2].metric("Retrieval hit@5", _pct(s["hit5"]))
    c[3].metric("MRR", f"{s['mrr']:.2f}" if s["mrr"] is not None else "—")
    c[4].metric("Decision agreement", _pct(s["decision_agreement"]))

    st.divider()

    # extraction detail
    st.subheader("Field extraction vs gold labels")
    rows = []
    for ap in report["per_applicant"]:
        for r in ap["extraction"]:
            rows.append({
                "applicant": ap["applicant"],
                "field": r["field"],
                "expected": r["expected"],
                "extracted": r["extracted"],
                "ok": "✅" if r["ok"] else "❌",
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # per-applicant accuracy chart
    acc = []
    for ap in report["per_applicant"]:
        ext = ap["extraction"]
        if ext:
            acc.append({
                "applicant": ap["applicant"],
                "accuracy": sum(1 for r in ext if r["ok"]) / len(ext),
            })
    if acc:
        fig = px.bar(
            pd.DataFrame(acc), x="applicant", y="accuracy", range_y=[0, 1],
            title="Extraction accuracy by applicant", text_auto=".0%",
        )
        fig.update_layout(height=340, margin=dict(t=48, b=8))
        st.plotly_chart(fig, use_container_width=True)

    # retrieval detail
    st.subheader("Retrieval — does the right evidence come back?")
    for ap in report["per_applicant"]:
        ret = ap["retrieval"]
        if not ret["questions"]:
            continue
        with st.expander(
            f"{ap['applicant']} — hit@3 {_pct(ret['hit3'])}, MRR {ret['mrr']:.2f}"
        ):
            for q in ret["questions"]:
                rank = f"rank {q['rank']}" if q["rank"] else "missed"
                icon = "✅" if q["rank"] and q["rank"] <= 3 else ("🟠" if q["rank"] else "❌")
                st.markdown(f"{icon} _{q['question']}_ → {rank}")

    # decisions
    st.subheader("Decision agreement")
    dec = [
        {
            "applicant": ap["applicant"],
            "expected": ap["gold_verdict"] or "—",
            "engine": ap["recommendation"]["verdict"],
            "match": "✅" if ap["verdict_ok"] else ("❌" if ap["verdict_ok"] is False else "—"),
            "flags": len(ap["flags"]),
        }
        for ap in report["per_applicant"]
    ]
    st.dataframe(pd.DataFrame(dec), use_container_width=True, hide_index=True)

    if report["rings"]:
        st.caption(f"Ring Watch active: {len(report['rings'])} cross-file linkage(s) factored into verdicts.")


def _pct(x) -> str:
    return f"{x:.0%}" if x is not None else "—"
