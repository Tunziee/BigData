from __future__ import annotations

import streamlit as st

from components.charts import donut, heatmap, radar
from components.ui import caveat, metric_grid, page_header, section_title, source_note
from services.artifacts import load_segments


page_header(
    "Customer segments",
    "Khám phá phân bố và hồ sơ giao dịch theo mô hình đã khóa.",
    "Enhanced K-Means Cluster Profiling · Primary Three-Model Quality Comparison · Spark SQL Analytical Layer",
)

model_id = st.selectbox("Mô hình phân khúc", ["M2", "M1", "B0"], index=0, format_func=lambda x: {"M2": "M2 · Enhanced K-Means", "M1": "M1 · Main K-Means", "B0": "B0 · Basic K-Means"}[x])
segments = load_segments(model_id)
is_full = model_id == "M2"
feature_cols = ["Recency", "Frequency", "Monetary", "AvgBasketSize", "ProductVariety"]

metric_grid([
    ("Model", model_id, "Enhanced eRFM5D" if model_id == "M2" else "RFM3D baseline/main"),
    ("Số cụm", str(len(segments)), None),
    ("Khách hàng", f"{int(segments['CustomerCount'].sum() if 'CustomerCount' in segments else segments['customer_count'].sum()):,}", "full data" if is_full else "cohort 1M"),
    ("Phạm vi profile", "1.362.281" if is_full else "1.000.000", "customers"),
])

if not is_full:
    caveat("B0/M1 profile trong dashboard là profile của cohort so sánh 1M; không suy luận đó là phân bố full 1.362.281 khách hàng.")

if model_id == "M2":
    distribution = segments.rename(columns={"segment_name_en": "Segment", "customer_count": "Customers", "customer_ratio": "Ratio"}).copy()
    distribution["Ratio"] = distribution["Ratio"] * 100
    med = distribution.rename(columns={"median_recency": "Recency", "median_frequency": "Frequency", "median_monetary": "Monetary", "median_avg_basket_size": "AvgBasketSize", "median_product_variety": "ProductVariety"})
else:
    distribution = segments.rename(columns={"SegmentNameEN": "Segment", "CustomerCount": "Customers", "RatioPct": "Ratio"}).copy()
    med = distribution.copy()

left, right = st.columns([.9, 1.1])
with left:
    st.plotly_chart(donut(distribution, "Segment", "Customers", f"Phân bố {model_id}"), use_container_width=True)
with right:
    heat = med[["SegmentId", *feature_cols]].melt(id_vars="SegmentId", var_name="Feature", value_name="Median")
    heat["SegmentId"] = heat["SegmentId"].astype(str)
    st.plotly_chart(heatmap(heat, "Feature", "SegmentId", "Median", "Raw median profile"), use_container_width=True)

section_title("Radar profile", "Chuẩn hóa min–max chỉ phục vụ trực quan hóa; bảng bên dưới giữ raw median.")
selected_ids = st.multiselect("Chọn tối đa 3 cụm để đối chiếu", distribution["SegmentId"].astype(int).tolist(), default=distribution["SegmentId"].astype(int).tolist()[:2], max_selections=3)
if selected_ids:
    selected = med[med["SegmentId"].isin(selected_ids)].copy()
    for col in feature_cols:
        lo, hi = float(med[col].min()), float(med[col].max())
        selected[col] = 0.5 if hi == lo else (selected[col] - lo) / (hi - lo)
    selected["Segment"] = selected["SegmentId"].map(distribution.set_index("SegmentId")["Segment"])
    st.plotly_chart(radar(selected, feature_cols, "Segment", "So sánh profile chuẩn hóa"), use_container_width=True)
else:
    st.info("Chọn ít nhất một cụm để xem radar.")

section_title("Segment profile table")
table = med[["SegmentId", "Segment", "Customers", "Ratio", *feature_cols]].copy()
table.columns = ["Cluster", "Segment", "Customers", "Ratio %", *feature_cols]
st.dataframe(table.style.format({"Ratio %": "{:.2f}", "Recency": "{:.0f}", "Frequency": "{:.0f}", "Monetary": "{:.4f}", "AvgBasketSize": "{:.2f}", "ProductVariety": "{:.0f}"}), hide_index=True, use_container_width=True)

section_title("Segment detail")
chosen = st.selectbox("Chọn cụm", distribution["SegmentId"].astype(int).tolist(), format_func=lambda x: f"Cụm {x} · {distribution.loc[distribution['SegmentId'].eq(x), 'Segment'].iloc[0]}")
detail = med[med["SegmentId"].eq(chosen)].iloc[0]
metric_grid([
    ("Customers", f"{int(detail['Customers']):,}", None),
    ("Ratio", f"{float(detail['Ratio']):.2f}%", None),
    ("Recency median", f"{float(detail['Recency']):.0f} ngày", None),
    ("Frequency median", f"{float(detail['Frequency']):.0f} ngày mua", None),
])
st.write(f"**{detail['Segment']}**: Recency median {detail['Recency']:.0f}, Frequency median {detail['Frequency']:.0f}, Monetary median {detail['Monetary']:.4f}, AvgBasketSize median {detail['AvgBasketSize']:.2f}, ProductVariety median {detail['ProductVariety']:.0f}.")
st.caption("Diễn giải chỉ mô tả hành vi giao dịch quan sát được; không suy diễn nhân khẩu học, xác suất rời bỏ hoặc hành vi tương lai.")
source_note("AvgBasketSize là proxy số dòng sản phẩm trung bình trên mỗi ngày mua phân biệt, không phải kích thước invoice basket thật.")
