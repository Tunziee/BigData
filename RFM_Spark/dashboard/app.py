from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from components.ui import load_css, sidebar_brand


st.set_page_config(
    page_title="Fashion Customer Segmentation",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()
sidebar_brand()

page_root = Path(__file__).parent / "pages"
pages = [
    st.Page(str(page_root / "model_comparison.py"), title="Model Comparison", icon=":material/compare_arrows:"),
    st.Page(str(page_root / "customer_segments.py"), title="Customer Segments", icon=":material/groups:"),
    st.Page(str(page_root / "customer_explorer.py"), title="Customer Explorer", icon=":material/person_search:"),
    st.Page(str(page_root / "spark_performance.py"), title="Spark Performance", icon=":material/speed:"),
]
navigation = st.navigation(pages, position="sidebar")
navigation.run()
