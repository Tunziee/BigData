from __future__ import annotations

import plotly.express as px
import streamlit as st

from components.charts import polish
from components.loaders import load_fp_growth
from components.metrics import metric_row, observation


st.set_page_config(page_title="FP-Growth", layout="wide")
st.title("FP-Growth — association rules")
thresholds, rules, summary = load_fp_growth()
metric_row([("Basket definition", summary["basket_proxy_definition"], None), ("Basket count", f"{summary['basket_count']:,}", None), ("Selected minSupport", f"{summary['selected_minSupport']:.4f}", None), ("Selected rules", f"{summary['association_rule_count']:,}", None)])
st.caption("Do dữ liệu không có order/invoice ID, basket là proxy customer_id + t_dat. Các slider bên dưới chỉ lọc rules đã sinh, không rerun FP-Growth.")

st.markdown("### Threshold sensitivity")
st.dataframe(thresholds, hide_index=True, use_container_width=True)
st.plotly_chart(polish(px.line(thresholds[thresholds["version"].eq("V2")], x="min_support", y="association_rule_count", markers=True, title="Số association rules theo support V2", log_x=True, color_discrete_sequence=["#2563eb"])), use_container_width=True)

st.markdown("### Bộ lọc rules đã sinh")
minimum_support = st.slider("Minimum Support", 0.0, 0.001, 0.0, 0.00001, format="%.5f")
minimum_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.0, 0.01)
minimum_lift = st.slider("Minimum Lift", 0.0, 3000.0, 0.0, 10.0)
filtered = rules[(rules["support"] >= minimum_support) & (rules["confidence"] >= minimum_confidence) & (rules["lift"] >= minimum_lift)].copy()
st.metric("Filtered rule count", len(filtered))
st.dataframe(filtered[["Antecedent", "Consequent", "support", "confidence", "lift", "Product group"]].head(100).style.format({"support": "{:.6f}", "confidence": "{:.4f}", "lift": "{:.2f}"}), hide_index=True, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Top rules by lift")
    st.dataframe(filtered.sort_values("lift", ascending=False)[["Antecedent", "Consequent", "confidence", "lift"]].head(10).style.format({"confidence": "{:.4f}", "lift": "{:.2f}"}), hide_index=True, use_container_width=True)
with right:
    st.subheader("Top rules by confidence")
    st.dataframe(filtered.sort_values("confidence", ascending=False)[["Antecedent", "Consequent", "support", "confidence", "lift"]].head(10).style.format({"support": "{:.6f}", "confidence": "{:.4f}", "lift": "{:.2f}"}), hide_index=True, use_container_width=True)

st.plotly_chart(polish(px.scatter(filtered, x="support", y="confidence", size="lift", color="lift", hover_name="Antecedent", title="Support–confidence của rules đã sinh", color_continuous_scale="Viridis")), use_container_width=True)
top_rule = rules.sort_values("lift", ascending=False).iloc[0]
st.success(f"Top rule theo lift: **{top_rule['Antecedent']} → {top_rule['Consequent']}** · support {top_rule['support']:.7f} · confidence {top_rule['confidence']:.4f} · lift {top_rule['lift']:.2f} · nhóm {top_rule['Product group']}.")
observation("Association does not imply causality. FP-Growth confidence cũng không phải accuracy; dashboard chỉ mô tả các quan hệ đồng xuất hiện trong basket proxy.")
