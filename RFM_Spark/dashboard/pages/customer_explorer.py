from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from components.ui import caveat, empty_state, metric_grid, page_header, section_title, source_note
from services.artifacts import load_customer_lookup, load_segments
from styles.plotly_theme import MODEL_COLORS, apply_theme


page_header(
    "Customer explorer",
    "Tra cứu customer_id thực tế và xem prediction đã lưu của B0, M1 và M2.",
)

with st.form("customer_lookup_form"):
    customer_id = st.text_input("Customer ID", placeholder="Nhập customer_id có thật trong output/features/erfm")
    submitted = st.form_submit_button("Tra cứu khách hàng")

if submitted:
    st.session_state["explorer_customer_id"] = customer_id.strip()

active_id = st.session_state.get("explorer_customer_id", "")
if not active_id:
    st.info("Nhập ID rồi bấm Tra cứu. Không có customer record nào được tạo giả.")
    st.stop()

lookup = load_customer_lookup(active_id)
features = lookup["features"]
if features is None:
    empty_state("Không tìm thấy customer", "ID không có trong output/features/erfm hoặc artifact không khả dụng. Kiểm tra lại định dạng và nguồn dữ liệu.")
    st.stop()

section_title("Customer overview")
feature_cols = ["Recency", "Frequency", "Monetary", "AvgBasketSize", "ProductVariety"]
metric_items = []
for col in feature_cols:
    if col not in features:
        continue
    decimals = 4 if col == "Monetary" else 2
    metric_items.append((col, f"{float(features[col]):,.{decimals}f}", None))
metric_grid(metric_items)
st.caption(f"Customer ID: `{active_id}`")

section_title("Model predictions")
prediction_columns = st.columns(3)
model_labels = {"B0": "B0 · Basic K-Means", "M1": "M1 · Main K-Means", "M2": "M2 · Enhanced K-Means"}
for col, model_id in zip(prediction_columns, ["B0", "M1", "M2"]):
    with col:
        with st.container(border=True):
            st.subheader(model_labels[model_id])
            pred = lookup["predictions"].get(model_id)
            if pred is None:
                st.info("Không có dữ liệu prediction cho customer này.")
                if model_id == "B0":
                    st.caption("B0 chỉ có prediction trên cohort so sánh 1M.")
                continue
            cluster = int(pred.get("prediction", pred.get("cluster")))
            profile = load_segments(model_id)
            row = profile[profile["SegmentId"].eq(cluster)]
            st.metric("Cluster ID", cluster)
            if row.empty:
                st.caption("Có prediction nhưng không có profile tương ứng trong artifact profile.")
            else:
                name_col = "segment_name_en" if model_id == "M2" else "SegmentNameEN"
                count_col = "customer_count" if model_id == "M2" else "CustomerCount"
                ratio_col = "customer_ratio" if model_id == "M2" else "RatioPct"
                st.write(str(row.iloc[0][name_col]))
                ratio_value = float(row.iloc[0][ratio_col]) * 100 if model_id == "M2" else float(row.iloc[0][ratio_col])
                st.caption(f"Cluster size: {int(row.iloc[0][count_col]):,} · Ratio: {ratio_value:.2f}%")

section_title("Feature comparison", "So sánh customer với raw median của profile trong cùng model/feature space.")
m2_pred = lookup["predictions"].get("M2")
med_row = pd.DataFrame()
if m2_pred is not None:
    cluster = int(m2_pred.get("prediction", m2_pred.get("cluster")))
    m2_profile = load_segments("M2")
    med_row = m2_profile[m2_profile["SegmentId"].eq(cluster)]
    if not med_row.empty:
        med_map = {"Recency": "median_recency", "Frequency": "median_frequency", "Monetary": "median_monetary", "AvgBasketSize": "median_avg_basket_size", "ProductVariety": "median_product_variety"}
        comp = pd.DataFrame({"Feature": feature_cols, "Customer": [features.get(c) for c in feature_cols], "Segment median": [med_row.iloc[0].get(med_map[c]) for c in feature_cols]})
        long = comp.melt(id_vars="Feature", var_name="Series", value_name="Value")
        st.plotly_chart(apply_theme(px.bar(long, x="Feature", y="Value", color="Series", barmode="group", title="eRFM5D raw value vs M2 median", color_discrete_sequence=[MODEL_COLORS["M2"], "#E8C3BA"])), use_container_width=True)
        st.dataframe(comp.style.format({"Customer": "{:.4f}", "Segment median": "{:.4f}"}), hide_index=True, use_container_width=True)
    else:
        st.info("M2 prediction có nhưng profile cluster chưa có trong artifact profile.")
else:
    st.info("Không có M2 prediction cho customer này nên không vẽ feature comparison.")

section_title("Vì sao customer thuộc nhóm này?")
if m2_pred is not None and not med_row.empty:
    bullets = []
    for key in feature_cols:
        value, median = float(features[key]), float(comp.loc[comp["Feature"].eq(key), "Segment median"].iloc[0])
        relation = "thấp hơn" if value < median else "cao hơn" if value > median else "xấp xỉ"
        bullets.append(f"{key}: {value:.4f} so với median cụm {median:.4f} ({relation}).")
    for item in bullets:
        st.markdown(f"- {item}")
    caveat("Đây là giải thích mô tả theo profile quan sát được, không phải prediction confidence hay xác suất thuộc cụm.")
else:
    st.info("Cần M2 prediction và profile để tạo phần giải thích theo dữ liệu.")

source_note("Cluster ID giữa các model không có ý nghĩa chung; dashboard luôn hiển thị model context và không mặc định Cụm 0 của M1 giống Cụm 0 của M2.")
