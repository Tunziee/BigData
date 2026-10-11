from __future__ import annotations

import streamlit as st

from components.charts import polish
from components.loaders import load_overview
from components.metrics import metric_row, note, observation


st.set_page_config(page_title="Tổng quan", layout="wide")
data = load_overview()
st.title("Tổng quan dự án")
st.caption("Các KPI dưới đây được đọc từ Stage 27, Stage 16 và Stage 28 V2.")

metric_row([
    ("Dòng giao dịch sản phẩm", f"{data['transaction_rows']:,}", None),
    ("Khách hàng", f"{data['customers']:,}", None),
    ("Articles", f"{data['articles']:,}", None),
    ("Cụm M2 cuối", str(data["clusters"]), None),
    ("FP-Growth rules", f"{data['rules']:,}", None),
])

st.markdown("### Pipeline phân tích")
st.markdown(
    '<div class="pipeline"><span>Raw Data</span><b>→</b><span>Cleaning</span><b>→</b><span>Parquet</span><b>→</b><span>RFM/eRFM</span><b>→</b><span>B0/M1/M2</span><b>→</b><span>Evaluation</span><b>→</b><span>Spark SQL</span><b>→</b><span>FP-Growth</span><b>→</b><span>Visualization</span></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.2, 1])
with left:
    st.markdown("### Mục tiêu")
    st.write("Xây dựng quy trình phân tích khách hàng bán lẻ có thể tái lập: chuẩn hóa dữ liệu, tạo RFM/eRFM, so sánh B0/M1/M2, diễn giải phân khúc M2 và bổ sung lớp Spark SQL/FP-Growth để hỗ trợ quyết định.")
    observation("Dashboard này chỉ trình bày và so sánh các kết quả đã khóa; không tạo thí nghiệm mới và không yêu cầu Spark khi khởi động.")
with right:
    st.markdown("### Phạm vi dữ liệu")
    st.write(f"Giai đoạn quan sát: **{data['start_date']} → {data['end_date']}**. Tổng Monetary theo artifact SQL: **{data['monetary_sum']:,.4f}**.")
    note("Kiểm tra dữ liệu sạch: PASS" if data["validation_pass"] else "Kiểm tra dữ liệu sạch: cần xem lại", "success" if data["validation_pass"] else "warning")

st.caption("Nhóm 11 · Apache Spark + K-Means + FP-Growth · Data and model results loaded from locked experimental artifacts.")
