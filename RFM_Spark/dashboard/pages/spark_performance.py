from __future__ import annotations

import plotly.express as px
import pandas as pd
import streamlit as st

from components.charts import apply_theme
from components.ui import caveat, metric_grid, page_header, section_title, source_note
from services.artifacts import load_cache, load_execution, load_fp_growth, load_runtime, load_shuffle_metrics, load_sql, load_stage27_summary, load_storage


page_header(
    "Spark performance & analytical layer",
    "Controlled runtime, storage, cache, Spark execution evidence and analytical outputs.",
    "Controlled benchmark outputs · Spark SQL Analytical Layer",
    "runtime · storage · cache · execution DAG · SQL",
)

runtime = load_runtime()
storage, storage_runtime = load_storage()
cache = load_cache()
execution = load_execution()
shuffle = load_shuffle_metrics()
stage27 = load_stage27_summary()

tabs = st.tabs(["Model runtime", "CSV vs Parquet", "Cache", "Spark execution"])

with tabs[0]:
    section_title("Controlled Primary B0/M1/M2 Runtime Benchmark", "Median seconds for the locked 1M comparison cohort.")
    runtime_view = runtime.rename(
        columns={
            "ModelID": "Model",
            "MedianFitOnlySec": "Fit",
            "MedianAssignAfterFitSec": "Assignment",
            "MedianTotalFitPlusAssignSec": "Total",
        }
    )
    long = runtime_view.melt(id_vars="Model", value_vars=["Fit", "Assignment", "Total"], var_name="Phase", value_name="Seconds")
    st.plotly_chart(
        apply_theme(px.bar(long, x="Model", y="Seconds", color="Phase", barmode="group", title="Median runtime by model and phase")),
        use_container_width=True,
    )
    st.dataframe(runtime_view.style.format({c: "{:.4f}" for c in ["Fit", "Assignment", "Total"]}), hide_index=True, use_container_width=True)
    fastest = runtime_view.loc[runtime_view["Total"].idxmin()]
    caveat(f"Trong protocol Track B, {fastest['Model']} có tổng Fit + Assignment thấp nhất ({fastest['Total']:.4f}s). Đây là kết quả của benchmark đã kiểm soát, không phải kết luận hiệu năng phổ quát.", "success")

with tabs[1]:
    section_title("Controlled CSV vs Parquet Storage & Query Benchmark", "CSV and Parquet are compared on the same controlled workload.")
    csv_bytes = float(storage.loc[storage["format"].eq("CSV"), "bytes"].iloc[0])
    parquet_bytes = float(storage.loc[storage["format"].eq("Parquet"), "bytes"].iloc[0])
    reduction = (1 - parquet_bytes / csv_bytes) * 100
    metric_grid([
        ("CSV", f"{csv_bytes / 1e9:.3f} GB", "artifact size"),
        ("Parquet", f"{parquet_bytes / 1e9:.3f} GB", "artifact size"),
        ("Storage reduction", f"{reduction:.2f}%", "computed from CSV/Parquet benchmark bytes"),
    ])
    query_runtime = storage_runtime
    query_long = query_runtime.melt(id_vars="QueryName", var_name="Format", value_name="Seconds")
    query_long["Format"] = query_long["Format"].str.replace("MedianSec", "", regex=False)
    st.plotly_chart(
        apply_theme(px.bar(query_long, x="QueryName", y="Seconds", color="Format", barmode="group", title="Repeated-read runtime by format")),
        use_container_width=True,
    )
    st.dataframe(query_runtime.style.format({c: "{:.4f}" for c in ["CSVMedianSec", "ParquetMedianSec"]}), hide_index=True, use_container_width=True)
    caveat("Parquet giảm dung lượng trong controlled storage/query workload. Đây là controlled repeated-read benchmark; không diễn giải thành cold-disk benchmark cho mọi hệ thống.")

with tabs[2]:
    section_title("Controlled Cache vs No-Cache Benchmark", "The dashboard reports measured query medians and the stored break-even analysis.")
    cache_view = cache[0]
    cache_long = cache_view.melt(id_vars="QueryID", var_name="Mode", value_name="Seconds")
    cache_long["Mode"] = cache_long["Mode"].str.replace("MedianSec", "", regex=False)
    st.plotly_chart(
        apply_theme(px.bar(cache_long, x="QueryID", y="Seconds", color="Mode", barmode="group", title="Cache vs no-cache median runtime")),
        use_container_width=True,
    )
    q3 = cache[1]["queries"]["Q3"]
    metric_grid([
        ("Finding", "CACHE_QUERY_DEPENDENT", "Controlled cache benchmark"),
        ("Q3 break-even", f"{q3['break_even_executions']} executions", "stored amortization analysis"),
        ("Q3 time saved", f"{q3['saved_time_per_execution_sec']:.3f}s", "per execution after materialization"),
    ])
    st.dataframe(cache_view.style.format({c: "{:.4f}" for c in ["NoCacheMedianSec", "CacheMedianSec"]}), hide_index=True, use_container_width=True)
    caveat("Cache không cải thiện mọi workload: Q1/Q2 chậm hơn trong artifact hiện tại; Q3 chỉ có lợi sau khi amortize chi phí materialization.")

with tabs[3]:
    section_title("Spark Execution Evidence", "Live UI/StatusTracker evidence is summarized without claiming a physical cluster.")
    q2, q3 = execution["q2"], execution["q3"]
    q2_shuffle = shuffle.loc[shuffle["Query"].eq("Q2_FILTER_AGGREGATION")].iloc[0]
    q3_shuffle = shuffle.loc[shuffle["Query"].eq("Q3_CUSTOMER_AGGREGATION")].iloc[0]
    metric_grid([
        ("Application", execution["spark_application_id"], "Spark execution evidence"),
        ("Q2 Spark stages / tasks", f"{q2['stage_count']} / {q2['task_count']}", "customer count workload"),
        ("Q3 Spark stages / tasks", f"{q3['stage_count']} / {q3['task_count']}", "customer aggregation workload"),
        ("Q3 shuffle records", f"{int(q3_shuffle['shuffle_read_records']):,}", "confirmed in artifact"),
    ])
    execution_df = pd.DataFrame(
        [
            {"Query": "Q2", "Spark stages": q2["stage_count"], "Tasks": q2["task_count"], "Shuffle bytes": q2_shuffle["shuffle_read_bytes"], "Shuffle records": q2_shuffle["shuffle_read_records"]},
            {"Query": "Q3", "Spark stages": q3["stage_count"], "Tasks": q3["task_count"], "Shuffle bytes": q3_shuffle["shuffle_read_bytes"], "Shuffle records": q3_shuffle["shuffle_read_records"]},
        ]
    )
    st.plotly_chart(
        apply_theme(px.bar(execution_df, x="Query", y="Shuffle bytes", log_y=True, title="Shuffle read bytes — log scale", color="Query", color_discrete_map={"Q2": "#000080", "Q3": "#581845"})),
        use_container_width=True,
    )
    st.dataframe(execution_df, hide_index=True, use_container_width=True)
    caveat("`local[4]` là một tiến trình local với bốn thread/cores, không phải một cụm bốn node. Q3 tạo shuffle lớn hơn đáng kể Q2 theo execution evidence.")

section_title("Spark SQL Analytical Layer", "Analytical outputs are available as locked read-only artifacts.")
monthly, products, ages, segments = load_sql()
peak = monthly.loc[monthly["monetary_sum"].idxmax()]
top_group = products.loc[products["monetary_sum"].idxmax(), "product_group"]
largest_segment = segments.loc[segments["customer_count"].idxmax(), "segment_name_en"]
metric_grid([
    ("Rows processed", f"{stage27['transaction_row_count']:,}", "Spark SQL Analytical Layer transaction rows"),
    ("Peak month", str(peak["month_label"]), "highest monetary_sum"),
    ("Top product group", str(top_group), "highest monetary_sum"),
    ("Largest M2 segment", str(largest_segment), "highest customer_count"),
])
source_note("Spark SQL Analytical Layer tables remain read-only in the dashboard; no SQL or Spark job is re-executed at runtime.")

section_title("FP-Growth Association-Rule Mining V2", "Association rules are displayed as stored; the dashboard does not rerun the miner.")
thresholds, rules, fp_summary = load_fp_growth()
selected = thresholds.loc[thresholds["selected"].eq(True)]
metric_grid([
    ("Basket proxy", "customer_id + t_dat", "distinct customer-day baskets"),
    ("Selected minSupport", f"{fp_summary['selected_minSupport']:.5f}", "read from FP-Growth V2 summary"),
    ("Association rules", f"{fp_summary['association_rule_count']:,}", "selected threshold"),
    ("Manual validation", "PASS" if fp_summary["manual_rule_validation_pass"] else "FAIL", "FP-Growth V2"),
])
st.dataframe(selected[["min_support", "frequent_itemset_count", "pair_or_larger_itemset_count", "association_rule_count"]], hide_index=True, use_container_width=True)
st.dataframe(rules[["Antecedent", "Consequent", "support", "confidence", "lift"]].head(10).style.format({"support": "{:.6f}", "confidence": "{:.4f}", "lift": "{:.2f}"}), hide_index=True, use_container_width=True)
caveat("Basket là proxy customer_id + t_dat; association không đồng nghĩa causality và confidence của FP-Growth không phải classification accuracy.")
