"""LendLens — loan-file intelligence for used-car lending.

Run:  streamlit run app.py
"""

import streamlit as st

from core.seed import seed
from ui import ask, evals_page, home, ingest, review, rings

st.set_page_config(
    page_title="LendLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; }
      [data-testid="stMetricValue"] { font-size: 1.6rem; }
      .ll-badge {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600; letter-spacing: .02em;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def _bootstrap():
    seed()
    return True


_bootstrap()

pages = st.navigation(
    [
        st.Page(home.render, title="Portfolio", icon="🏠", url_path="portfolio", default=True),
        st.Page(review.render, title="File Review", icon="📄", url_path="review"),
        st.Page(ask.render, title="Ask the File", icon="💬", url_path="ask"),
        st.Page(rings.render, title="Ring Watch", icon="🕸️", url_path="rings"),
        st.Page(evals_page.render, title="Evals", icon="📊", url_path="evals"),
        st.Page(ingest.render, title="New Application", icon="➕", url_path="new"),
    ]
)

with st.sidebar:
    st.markdown("### LendLens")
    st.caption(
        "Turns messy loan files into evidence-linked underwriting signals. "
        "Hybrid retrieval (BM25 + dense + RRF), transparent policy rules, "
        "cross-file fraud linkage, and an offline eval harness."
    )
    from core.config import live_mode

    mode = "🟢 Live (LLM)" if live_mode() else "🟡 Offline demo"
    st.caption(f"Mode: {mode} — set `ANTHROPIC_API_KEY` for live answers.")

pages.run()
