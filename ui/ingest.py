"""New Application — ingest a fresh loan-file bundle."""

import streamlit as st

from core.pipeline import run_pipeline
from core.store import save_applicant, save_segments

_TEMPLATE = """Loan application — <Name>, age <..>, resident of <...>. Contact number <10 digits>.
Seeking ₹<amount> to purchase a <year> <make model> quoted at ₹<price> by <dealer>.

Applicant states a take-home salary of ₹<amount> per month as <designation>.

<Name> has worked with <Employer Pvt Ltd> for <N> months as <designation>.

Declared obligations: <...>

Bank — Account statement, A/c No. XX<digits>, <months>
05-Mar-2026 | NEFT CR | SALARY <EMPLOYER> MAR | +45,000 | bal 61,200
07-Mar-2026 | ACH D | <LENDER> EMI 1234 | -5,200 | bal 56,000
"""


def render():
    st.title("➕ New Application")
    st.caption(
        "Paste one text bundle per applicant — application narrative, employer letter, "
        "and pipe-delimited statement lines. The pipeline segments, classifies, extracts, "
        "and indexes it; the file then appears everywhere else in the app."
    )

    name = st.text_input("Applicant name")
    c = st.columns(3)
    loan_amount = c[0].number_input("Loan amount (₹)", min_value=0, value=400000, step=10000)
    vehicle_price = c[1].number_input("Vehicle price (₹)", min_value=0, value=550000, step=10000)
    vehicle_age = c[2].number_input("Vehicle age (years)", min_value=0, max_value=20, value=4)

    uploaded = st.file_uploader("…or upload a .txt bundle", type=["txt"])
    text = st.text_area("Document bundle", height=320, placeholder=_TEMPLATE)
    if uploaded is not None:
        text = uploaded.read().decode("utf-8", errors="replace")
        st.info(f"Using uploaded file ({len(text)} chars).")

    if st.button("Run extraction pipeline", type="primary", disabled=not (name and text.strip())):
        bar = st.progress(0.0, text="Starting…")

        def tick(label, frac):
            bar.progress(frac, text=label)

        aid = save_applicant(
            name,
            meta={"stated": {
                "loan_amount": loan_amount or None,
                "vehicle_price": vehicle_price or None,
                "vehicle_age_years": vehicle_age,
            }},
        )
        segments = run_pipeline(text, id_prefix=aid[:4], progress=tick)
        save_segments(aid, segments)
        st.session_state.selected_applicant = aid
        st.success(
            f"Ingested **{len(segments)} segments** for {name}. "
            "Open **File Review** for the profile and flags, or **Ask the File** to query it."
        )
        cats = {}
        for s in segments:
            cats[s["category"]] = cats.get(s["category"], 0) + 1
        st.caption(" · ".join(f"{k}: {v}" for k, v in sorted(cats.items())))
