"""Shared academic dashboard styling."""

from __future__ import annotations

import streamlit as st


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { max-width: 1500px; padding-top: 1.5rem; }
        h1, h2, h3 { color: #16324f; }
        .subtitle { color: #55728b; font-size: 1.05rem; margin-top: -0.65rem; }
        .observation { background: #eef6ff; border-left: 4px solid #2563eb; padding: .75rem 1rem; margin: .75rem 0 1rem; border-radius: .25rem; }
        .small-muted { color: #64748b; font-size: .88rem; }
        .pipeline { display:flex; flex-wrap:wrap; gap:.45rem; align-items:center; margin: 1rem 0 1.2rem; }
        .pipeline span { background:#e8f1f8; color:#16324f; padding:.45rem .7rem; border-radius:.35rem; font-weight:600; }
        .pipeline b { color:#64748b; }
        section[data-testid="stSidebar"] { background: #f7fafc; }
        </style>
        """,
        unsafe_allow_html=True,
    )
