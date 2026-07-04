"""Ask the File — grounded Q&A with citations."""

import streamlit as st

from core.chat import answer
from core.config import ANTHROPIC_MODEL, live_mode
from core.seed import policy_corpus
from core.store import segments_for
from ui.common import applicant_picker, render_citations

_SUGGESTIONS = [
    "What is the applicant's real monthly income?",
    "What FOIR does this application carry?",
    "Any bounces in banking conduct?",
    "What fraud indicators does this file show?",
    "What does policy say about LTV for older vehicles?",
]


def render():
    st.title("💬 Ask the File")
    applicant = applicant_picker("ask_picker")
    if not applicant:
        return
    st.caption(
        "Answers are grounded in this applicant's documents + the lending policy corpus. "
        "Every reply cites the segments it used. Protected attributes are hard-filtered "
        "out of decisioning by the fairness guardrail."
    )
    if live_mode():
        st.info(
            f"🔑 **Live RAG mode** ({ANTHROPIC_MODEL}) — Claude synthesizes each answer "
            "from the retrieved evidence segments only, citing [seg_id] inline."
        )
    else:
        st.warning(
            "**No `ANTHROPIC_API_KEY` found.** This branch runs Q&A as retrieval-augmented "
            "generation — copy `.env.example` to `.env` and add your key. Until then, "
            "answers fall back to extractive evidence."
        )

    key = f"chat_{applicant['id']}"
    if key not in st.session_state:
        st.session_state[key] = []

    cols = st.columns(len(_SUGGESTIONS[:3]))
    clicked = None
    for c, s in zip(cols, _SUGGESTIONS[:3]):
        if c.button(s, key=f"sugg_{applicant['id']}_{s[:18]}", use_container_width=True):
            clicked = s

    for msg in st.session_state[key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])
            if msg["role"] == "assistant":
                render_citations(msg.get("citations", []))
                if msg.get("mode"):
                    st.caption(f"mode: {msg['mode']}")

    question = st.chat_input("Ask about income, obligations, conduct, vehicle, or policy…") or clicked
    if not question:
        return

    st.session_state[key].append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and synthesizing…" if live_mode() else "Retrieving evidence…"):
            result = answer(
                question,
                segments_for(applicant["id"]),
                policy_corpus(),
                applicant_id=applicant["id"],
            )
        st.markdown(result["text"])
        render_citations(result["citations"])
        st.caption(f"mode: {result['mode']}")

    st.session_state[key].append(
        {"role": "assistant", "text": result["text"], "citations": result["citations"], "mode": result["mode"]}
    )
