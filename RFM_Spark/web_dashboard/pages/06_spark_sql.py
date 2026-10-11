from __future__ import annotations

import plotly.express as px
import streamlit as st

from components.charts import line, polish
from components.loaders import load_sql
from components.metrics import observation


st.set_page_config(page_title="Spark SQL", layout="wide")
st.title("Spark SQL — phân tích kinh doanh")
monthly, products, ages, segments = load_sql()
tabs = st.tabs(["MONTHLY TREND", "PRODUCT GROUP", "AGE BAND", "SEGMENTS"])

with tabs[0]:
    st.plotly_chart(line(monthly, "month_label", "monetary_sum", "Monthly Monetary trend", {"month_label": "Tháng", "monetary_sum": "Monetary"}), use_container_width=True)
    peak = monthly.loc[monthly["monetary_sum"].idxmax()]
    observation(f"Tháng có Monetary cao nhất trong artifact là **{peak['month_label']}** với {peak['monetary_sum']:,.2f}; kết quả này được đọc từ Stage 27 Q2.")

with tabs[1]:
    top = products.sort_values("monetary_sum", ascending=False).head(10)
    st.plotly_chart(polish(px.bar(top.sort_values("monetary_sum"), x="monetary_sum", y="product_group", orientation="h", title="Top 10 product groups theo Monetary", color_discrete_sequence=["#2563eb"])), use_container_width=True)
    st.dataframe(top, hide_index=True, use_container_width=True)
    observation(f"Product group đứng đầu theo Monetary là **{top.iloc[0]['product_group']}**.")

with tabs[2]:
    st.plotly_chart(polish(px.bar(ages, x="age_band", y="monetary_sum", title="Monetary theo age band", color_discrete_sequence=["#0f766e"])), use_container_width=True)
    st.dataframe(ages, hide_index=True, use_container_width=True)
    top_age = ages.loc[ages["monetary_sum"].idxmax(), "age_band"]
    observation(f"Age band có tổng Monetary cao nhất là **{top_age}**; đây là mô tả tổng hợp theo artifact, không phải suy luận nhân khẩu học ngoài dữ liệu.")

with tabs[3]:
    st.plotly_chart(polish(px.bar(segments, x="segment_name_vi", y="customer_count", title="M2 segment analytics trong Spark SQL", color="segment_name_vi", color_discrete_sequence=["#2563eb", "#0f766e", "#f59e0b", "#dc2626", "#7c3aed"])), use_container_width=True)
    st.dataframe(segments[["Cluster", "segment_name_vi", "customer_count", "customer_ratio", "median_recency", "median_frequency", "median_monetary"]], hide_index=True, use_container_width=True)
    observation("Spark SQL tái hiện membership M2 và cung cấp các phân tích theo thời gian, product group, age band và segment.")
