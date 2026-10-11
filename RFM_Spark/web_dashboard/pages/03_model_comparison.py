from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import polish
from components.loaders import load_model_comparison, load_quality_series
from components.metrics import metric_row, observation


st.set_page_config(page_title="So sánh mô hình", layout="wide")
st.title("So sánh B0 / M1 / M2")
st.caption("Stage 22 authority · các màu/đánh giá chỉ có ý nghĩa trong cùng evaluation space.")
comparison = load_model_comparison()
quality, sse, _ = load_quality_series()

selected = st.selectbox("Chọn mô hình để xem cấu hình", ["B0", "M1", "M2"], index=2)
row = comparison[comparison["ModelID"].astype(str) == selected].iloc[0]
metric_row([("Feature space", str(row["TrainingFeatureSpace"]), None), ("K", str(int(row["K"])), None), ("Initialization", str(row["InitMode"]), None), ("Role", str(row["ModelRole"]), None)])

display = comparison[["ModelID", "ModelName", "TrainingFeatureSpace", "K", "InitMode", "Silhouette_RFM3D", "Silhouette_ERFM5D", "SSEPerCustomer_RFM3D", "SSEPerCustomer_ERFM5D", "NormalizedEntropy", "ClusterSizeCV"]].copy()
display.columns = ["Model", "Tên", "Feature space", "K", "Init", "Silhouette RFM3D", "Silhouette eRFM5D", "SSE/customer RFM3D", "SSE/customer eRFM5D", "Entropy", "Cluster CV"]
st.dataframe(display.style.format({c: "{:.6f}" for c in display.columns[5:]}), hide_index=True, use_container_width=True)

left, right = st.columns(2)
with left:
    long = comparison.melt(id_vars="ModelID", value_vars=["Silhouette_RFM3D", "Silhouette_ERFM5D"], var_name="Space", value_name="Silhouette")
    long["Space"] = long["Space"].str.replace("Silhouette_", "", regex=False).str.replace("ERFM5D", "eRFM5D")
    st.plotly_chart(polish(px.bar(long, x="ModelID", y="Silhouette", color="Space", barmode="group", title="Silhouette theo evaluation space", color_discrete_sequence=["#2563eb", "#f59e0b"])), use_container_width=True)
with right:
    st.plotly_chart(polish(px.bar(sse, x="model_id", y="sse_per_customer", color="space", barmode="group", title="SSE/customer trong common space", labels={"model_id": "Model", "sse_per_customer": "SSE/customer", "space": "Space"}, color_discrete_sequence=["#0f766e", "#dc2626"])), use_container_width=True)

metric_long = quality.melt(id_vars="ModelID", value_vars=["Entropy", "Cluster CV", "Share range"], var_name="Metric", value_name="Value")
left, right = st.columns(2)
with left:
    st.plotly_chart(polish(px.bar(metric_long[metric_long["Metric"].isin(["Entropy", "Cluster CV"])], x="ModelID", y="Value", color="Metric", barmode="group", title="Entropy và Cluster CV", color_discrete_sequence=["#7c3aed", "#475569"])), use_container_width=True)
with right:
    st.plotly_chart(polish(px.bar(metric_long[metric_long["Metric"] == "Share range"], x="ModelID", y="Value", title="Khoảng min/max cluster share (điểm phần trăm)", color_discrete_sequence=["#2563eb"])), use_container_width=True)

observation("M2 có Silhouette eRFM5D cao hơn M1 từ 0.356573 lên 0.481807 (+35.12%) và SSE/customer giảm từ 2.257108 xuống 1.347449 (−40.30%). M2 cải thiện rõ rệt cho objective eRFM5D nhưng không vượt trội tuyệt đối trong mọi không gian đặc trưng.")
st.warning("Không gắn nhãn universal winner: M1 vẫn có Silhouette RFM3D cao hơn M2, vì vậy mọi kết luận phải giữ đúng evaluation space.")
