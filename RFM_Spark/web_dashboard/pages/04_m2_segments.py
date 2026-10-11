from __future__ import annotations

import streamlit as st

from components.charts import donut, heatmap, radar
from components.loaders import load_overview, load_segments
from components.metrics import metric_row, observation


st.set_page_config(page_title="Phân khúc M2", layout="wide")
st.title("Phân khúc khách hàng M2")
segments = load_segments()
overview = load_overview()
metric_row([("K", str(overview["clusters"]), None), ("Khách hàng", f"{overview['customers']:,}", None), ("Feature space", "eRFM5D", "log1p + StandardScaler")])

left, right = st.columns(2)
with left:
    st.plotly_chart(donut(segments, "segment_name_en", "customer_count", "Phân phối khách hàng theo cụm"), use_container_width=True)
with right:
    heat = segments.melt(id_vars=["Cluster", "segment_name_vi"], value_vars=["median_recency", "median_frequency", "median_monetary", "median_avg_basket_size", "median_product_variety"], var_name="Feature", value_name="Median")
    st.plotly_chart(heatmap(heat, "Feature", "Cluster", "Median", "Heatmap hồ sơ trung vị M2"), use_container_width=True)

st.markdown("### Hồ sơ cụm")
table = segments[["Cluster", "segment_name_vi", "customer_count", "customer_ratio", "median_recency", "median_frequency", "median_monetary", "median_avg_basket_size", "median_product_variety"]].copy()
table.columns = ["Cụm", "Nhãn", "Khách hàng", "Tỷ lệ", "Median Recency", "Median Frequency", "Median Monetary", "Median AvgBasketSize", "Median ProductVariety"]
st.dataframe(table.style.format({"Tỷ lệ": "{:.2%}", "Median Recency": "{:.0f}", "Median Frequency": "{:.0f}", "Median Monetary": "{:.4f}", "Median AvgBasketSize": "{:.2f}", "Median ProductVariety": "{:.0f}"}), hide_index=True, use_container_width=True)

st.markdown("### Drill-down cụm")
choice = st.selectbox("Chọn cụm", segments["Cluster"].astype(int).tolist(), format_func=lambda value: f"Cụm {value} — {segments.loc[segments['Cluster'].eq(value), 'segment_name_vi'].iloc[0]}")
selected = segments[segments["Cluster"] == choice].iloc[0]
metric_row([("Khách hàng", f"{int(selected['customer_count']):,}", None), ("Tỷ lệ", f"{selected['customer_ratio']:.2%}", None), ("Median Recency", f"{selected['median_recency']:.0f} ngày", None), ("Median Frequency", f"{selected['median_frequency']:.0f} ngày mua", None)])
left, right = st.columns([1, 1.2])
with left:
    radar_row = {"Recency": selected["median_recency"], "Frequency": selected["median_frequency"], "Monetary": selected["median_monetary"], "AvgBasketSize": selected["median_avg_basket_size"], "ProductVariety": selected["median_product_variety"]}
    st.plotly_chart(radar(radar_row, list(radar_row)), use_container_width=True)
with right:
    st.write(f"**{selected['segment_name_vi']}**")
    st.write(f"Median Monetary: **{selected['median_monetary']:.4f}** · Median AvgBasketSize: **{selected['median_avg_basket_size']:.2f}** · Median ProductVariety: **{selected['median_product_variety']:.0f}**.")
    st.write("Diễn giải chỉ mô tả hành vi giao dịch quan sát được từ các median R/F/M/eRFM; không suy diễn nhân khẩu học hoặc tính cách.")

observation("Các cụm M2 được đọc từ membership cuối và đối soát với Stage 27 SQL; tổng số khách hàng theo năm cụm phải bằng 1.362.281.")
