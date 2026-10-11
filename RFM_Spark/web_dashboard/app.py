from __future__ import annotations

import streamlit as st

from components.loaders import load_overview
from components.styles import apply_styles


st.set_page_config(
    page_title="Phân tích khách hàng bán lẻ thời trang",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_styles()

st.title("Phân tích và phân cụm khách hàng bán lẻ thời trang trên nền tảng Apache Spark")
st.markdown('<div class="subtitle">RFM/eRFM · K-Means · Spark SQL · FP-Growth</div>', unsafe_allow_html=True)
st.markdown("### Lớp trực quan hóa bổ sung")
st.write("Chọn một trang trong thanh bên để khám phá các kết quả đã khóa của dự án. Dashboard chỉ đọc JSON, CSV và Parquet nhẹ; không khởi động Spark và không nạp dữ liệu giao dịch 31,8M dòng.")

data = load_overview()
st.info("Mở trang **Tổng quan** để xem toàn bộ KPI và pipeline phân tích.")
st.caption("Nhóm 11 · Apache Spark + K-Means + FP-Growth · Kết quả được đọc từ locked experimental artifacts.")
