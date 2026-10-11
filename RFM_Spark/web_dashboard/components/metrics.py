"""Metric cards and deterministic commentary."""

from __future__ import annotations

import streamlit as st


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, delta) in zip(columns, items):
        with column:
            st.metric(label, value, delta)


def note(text: str, kind: str = "info") -> None:
    if kind == "warning":
        st.warning(text)
    elif kind == "success":
        st.success(text)
    else:
        st.info(text)


def observation(text: str) -> None:
    st.markdown(f"<div class='observation'><b>Nhận xét</b><br>{text}</div>", unsafe_allow_html=True)
