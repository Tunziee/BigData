from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import space_bar
from components.ui import caveat, metric_grid, page_header, section_title, source_note
from services.artifacts import load_common_sse, load_entropy_balance, load_model_quality
from styles.plotly_theme import apply_theme


page_header(
    "So sánh chất lượng mô hình",
    "Đánh giá B0, M1 và M2 trên cohort so sánh cố định 1.000.000 khách hàng.",
    "Primary Three-Model Quality Comparison · comparison matrix",
    "cohort 1M · common-space evaluation",
)

quality = load_model_quality()
sse = load_common_sse()
entropy, cv, share = load_entropy_balance()
quality = quality.merge(entropy, on="ModelID").merge(cv, on="ModelID").merge(share, on="ModelID")

space = st.radio("Không gian đánh giá", ["RFM3D", "eRFM5D"], horizontal=True, key="quality_space")
sil_col = "Silhouette_RFM3D" if space == "RFM3D" else "Silhouette_ERFM5D"
sse_col = "SSEPerCustomer_RFM3D" if space == "RFM3D" else "SSEPerCustomer_ERFM5D"
best_row = quality.loc[quality[sil_col].idxmax()]
m1 = quality.loc[quality["ModelID"].eq("M1")].iloc[0]
m2 = quality.loc[quality["ModelID"].eq("M2")].iloc[0]
improvement = (float(m2["Silhouette_ERFM5D"]) / float(m1["Silhouette_ERFM5D"]) - 1) * 100
reduction = (1 - float(m2["SSEPerCustomer_ERFM5D"]) / float(m1["SSEPerCustomer_ERFM5D"])) * 100

metric_grid([
    (f"Silhouette {space}", f"{float(best_row[sil_col]):.6f}", f"{best_row['ModelID']} cao nhất trong space này"),
    (f"SSE/customer {space}", f"{float(quality.loc[quality[sse_col].idxmin(), sse_col]):.6f}", f"{quality.loc[quality[sse_col].idxmin(), 'ModelID']} thấp nhất"),
    ("Cụm / features", "5 / 5" if space == "eRFM5D" else "4 / 3", "M2 / M1-B0"),
    ("M1 → M2 eRFM5D", f"+{improvement:.2f}%", f"SSE/customer −{reduction:.2f}%"),
])

if space == "eRFM5D":
    caveat("M2 cải thiện chất lượng phân hoạch trong không gian eRFM5D so với M1. Tỷ lệ cải thiện được tính từ artifact hiện tại, không gõ cố định.", "success")
else:
    caveat("M1 có Silhouette cao hơn M2 trong không gian RFM3D. Không so native WCSS giữa RFM3D và eRFM5D.", "warning")

left, right = st.columns([1.05, .95])
with left:
    long = quality.melt(id_vars="ModelID", value_vars=["Silhouette_RFM3D", "Silhouette_ERFM5D"], var_name="Space", value_name="Silhouette")
    long["Space"] = long["Space"].str.replace("Silhouette_", "", regex=False)
    st.plotly_chart(space_bar(long.rename(columns={"ModelID": "Model"}), "Model", "Silhouette", "Silhouette theo evaluation space"), use_container_width=True)
with right:
    sse_plot = sse.rename(columns={"model_id": "Model", "space": "Space", "sse_per_customer": "SSE/customer"})
    st.plotly_chart(space_bar(sse_plot, "Model", "SSE/customer", "SSE/customer trong common space"), use_container_width=True)

left, right = st.columns(2)
with left:
    balance = quality[["ModelID", "Entropy", "ClusterCV"]].melt(id_vars="ModelID", var_name="Metric", value_name="Value")
    st.plotly_chart(apply_theme(px.bar(balance, x="ModelID", y="Value", color="Metric", barmode="group", title="Entropy và Cluster CV", color_discrete_sequence=["#000080", "#581845"])), use_container_width=True)
with right:
    share_plot = quality[["ModelID", "ShareRangePct"]].rename(columns={"ModelID": "Model", "ShareRangePct": "Share range (pp)"})
    st.plotly_chart(apply_theme(px.bar(share_plot, x="Model", y="Share range (pp)", title="Khoảng min/max cluster share", color="Model", color_discrete_map={"B0": "#5B6770", "M1": "#000080", "M2": "#581845"})), use_container_width=True)

section_title("Metadata và số liệu đầy đủ", "Tất cả cột dưới đây giữ nguyên evaluation space của Primary Three-Model Quality Comparison.")
table = quality[["ModelID", "ModelName", "TrainingFeatureSpace", "K", "InitMode", sil_col, sse_col, "Entropy", "ClusterCV", "MinClusterRatioPct", "MaxClusterRatioPct"]].copy()
table.columns = ["Model", "Tên", "Feature space", "K", "Initialization", f"Silhouette {space}", f"SSE/customer {space}", "Entropy", "Cluster CV", "Min share %", "Max share %"]
st.dataframe(table.style.format({c: "{:.6f}" for c in table.columns[5:10]}), hide_index=True, use_container_width=True)
source_note("Primary Three-Model Quality Comparison là authority cho quality comparison; B0/M1/M2 không có ground truth classification và không hiển thị Accuracy, F1-score hoặc Confusion Matrix.")
