from __future__ import annotations

import plotly.express as px
import streamlit as st

from components.charts import polish
from components.loaders import load_cache, load_cache_break_even, load_runtime, load_shuffle, load_storage
from components.metrics import metric_row, observation


st.set_page_config(page_title="Spark benchmark", layout="wide")
st.title("Benchmark Spark và lưu trữ")
tabs = st.tabs(["MODEL RUNTIME", "CSV VS PARQUET", "CACHE", "SPARK EXECUTION"])

with tabs[0]:
    runtime = load_runtime().rename(columns={"ModelID": "Model", "MedianFitOnlySec": "Fit", "MedianAssignAfterFitSec": "Assignment", "MedianTotalFitPlusAssignSec": "Total"})
    long = runtime.melt(id_vars="Model", value_vars=["Fit", "Assignment", "Total"], var_name="Phase", value_name="Seconds")
    st.plotly_chart(polish(px.bar(long, x="Model", y="Seconds", color="Phase", barmode="group", title="Stage 23 Track B — median runtime", color_discrete_sequence=["#2563eb", "#f59e0b", "#0f766e"])), use_container_width=True)
    st.dataframe(runtime.style.format({c: "{:.4f}" for c in ["Fit", "Assignment", "Total"]}), hide_index=True, use_container_width=True)
    observation("M2 có tổng fit + assignment thấp hơn B0/M1 trong Track B, trong khi assignment riêng cao hơn; đây là benchmark đã kiểm soát, không phải kết luận hiệu năng phổ quát.")

with tabs[1]:
    storage, query = load_storage()
    csv_bytes = float(storage.loc[storage["format"].eq("CSV"), "bytes"].iloc[0])
    pq_bytes = float(storage.loc[storage["format"].eq("Parquet"), "bytes"].iloc[0])
    reduction = (1 - pq_bytes / csv_bytes) * 100
    metric_row([("CSV", f"{csv_bytes / 1e9:.3f} GB", None), ("Parquet", f"{pq_bytes / 1e9:.3f} GB", None), ("Storage reduction", f"{reduction:.4f}%", None)])
    qlong = query.melt(id_vars="QueryName", var_name="Format", value_name="Seconds")
    qlong["Format"] = qlong["Format"].str.replace("MedianSec", "", regex=False)
    st.plotly_chart(polish(px.bar(qlong, x="QueryName", y="Seconds", color="Format", barmode="group", title="Runtime CSV vs Parquet", color_discrete_sequence=["#dc2626", "#0f766e"])), use_container_width=True)
    observation("Parquet giảm dung lượng khoảng 77,8% và đạt runtime tốt hơn trong workload đọc lặp đã đo. Đây là controlled repeated-read benchmark, not cold disk.")

with tabs[2]:
    cache = load_cache()
    long = cache.melt(id_vars="QueryID", var_name="Mode", value_name="Seconds")
    long["Mode"] = long["Mode"].str.replace("MedianSec", "", regex=False)
    st.plotly_chart(polish(px.bar(long, x="QueryID", y="Seconds", color="Mode", barmode="group", title="Cache vs no-cache", color_discrete_sequence=["#64748b", "#2563eb"])), use_container_width=True)
    break_even = load_cache_break_even()["queries"]["Q3"]["break_even_executions"]
    metric_row([("Finding", "CACHE_QUERY_DEPENDENT", None), ("Q3 break-even", f"≈ {break_even} executions", None)])
    observation("Cache không cải thiện mọi workload; Q1/Q2 chậm hơn trong benchmark này, Q3 chỉ cải thiện nhẹ. No-cache ở đây không đồng nghĩa cold physical disk.")

with tabs[3]:
    shuffle = load_shuffle()
    plot = shuffle.melt(id_vars="query", value_vars=["shuffle_read_bytes", "shuffle_write_bytes"], var_name="Metric", value_name="Bytes")
    st.plotly_chart(polish(px.bar(plot, x="query", y="Bytes", color="Metric", barmode="group", log_y=True, title="Shuffle bytes (log scale)", color_discrete_sequence=["#f59e0b", "#dc2626"])), use_container_width=True)
    st.dataframe(shuffle, hide_index=True, use_container_width=True)
    observation("Q3 aggregation theo customer tạo shuffle lớn hơn đáng kể Q2: 327.776.023 bytes so với 504 bytes. local[4] là một tiến trình local với bốn thread/cores, không phải cụm bốn node.")
