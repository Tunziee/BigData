from __future__ import annotations

from pathlib import Path

import streamlit as st


APP_ROOT = Path(__file__).resolve().parents[1]


def load_css() -> None:
    css_path = APP_ROOT / "styles" / "theme.css"
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def sidebar_brand() -> None:
    with st.sidebar:
        st.markdown("## RFM_Spark")
        st.caption("Nhóm 11 · Fashion Customer Segmentation")
        st.divider()
        st.caption("Locked research pipeline outputs")
        st.caption("Không retrain · Không rerun Spark")


def page_header(title: str, subtitle: str, source: str = "Locked project artifacts", cohort: str | None = None) -> None:
    st.title(title)
    st.markdown(f"<div class='page-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def section_title(title: str, description: str | None = None) -> None:
    st.markdown(f"<h2 class='section-title'>{title}</h2>", unsafe_allow_html=True)
    if description:
        st.caption(description)


def source_note(text: str) -> None:
    # Kept as a compatibility helper for pages that still call it, but source
    # provenance is intentionally hidden from the four-page presentation UI.
    return None


def caveat(text: str, level: str = "info") -> None:
    if level == "warning":
        st.warning(text)
    elif level == "success":
        st.success(text)
    else:
        st.info(text)


def empty_state(title: str, detail: str) -> None:
    st.markdown(f"<div class='empty-state'><strong>{title}</strong><br>{detail}</div>", unsafe_allow_html=True)


def metric_grid(items: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        with col:
            st.metric(label, value, delta)


def format_number(value: float | int | None, decimals: int = 0) -> str:
    if value is None:
        return "—"
    if decimals:
        return f"{float(value):,.{decimals}f}"
    return f"{int(value):,}"
