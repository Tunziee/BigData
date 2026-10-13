# Dashboard artifact inventory

Authority: locked outputs from `D:\Doan_bigdata\BigData\BigData\RFM_Spark`. The dashboard is read-only and does not rerun Spark or FP-Growth.

| Layer | Artifact | Schema / function used | Status and limitation |
|---|---|---|---|
| Quality | `output/comparison/stage22/primary_three_model_comparison_matrix.csv` | B0/M1/M2, K, Silhouette RFM3D/eRFM5D, SSE/customer | FOUND; common 1M cohort only |
| Profiles | `output/comparison/stage22/raw_profiles_1m.csv` | Segment IDs/names, count, ratio, feature medians | FOUND; B0/M1 are 1M-cohort profiles |
| M2 profiles | `output/sql/stage27/q6_m2_segment_analytics.csv` | segment names, count/ratio, R/F/M and eRFM summaries | FOUND; full M2 analytical layer |
| Balance | `output/visualization/stage29/data/fig06_common_space_sse_per_customer.csv`; `fig07a_cluster_entropy.csv`; `fig07b_cluster_cv.csv`; `fig07c_cluster_share_range.csv` | common-space SSE, entropy, CV, share range | FOUND; no native-WCSS cross-space comparison |
| Runtime | `output/visualization/stage29/data/fig12_model_runtime_benchmark.csv` | fit, assignment, total seconds | FOUND; controlled Stage 23 protocol |
| Storage | `fig13_csv_parquet_storage.csv`, `fig14_csv_parquet_runtime.csv` | bytes and query medians | FOUND; controlled Stage 24 repeated-read benchmark |
| Cache | `fig15_cache_vs_no_cache.csv`, `output/benchmark/stage25/cache_break_even_analysis.json` | query medians and break-even | FOUND; cache effect is workload-dependent |
| Execution | `output/analysis/stage26/stage26_final_summary.json` | application, jobs/stages/tasks, shuffle bytes | FOUND; live UI/StatusTracker evidence, not a physical cluster claim |
| SQL | `output/visualization/stage29/data/fig17_monthly_monetary_trend.csv`; `fig18_top_product_groups.csv`; `fig19_age_band_monetary.csv`; `output/sql/stage27/q6_m2_segment_analytics.csv`; `output/sql/stage27/stage27_final_summary.json` | monthly, product group, age band, segments, dataset KPI | FOUND; read-only Stage 27 outputs |
| FP-Growth | `output/fpgrowth/stage28_v2/stage28_v2_final_summary.json`, `selected_association_rules.parquet`, `enriched_top_rules.csv` | threshold protocol, 104 rules, support/confidence/lift | FOUND; basket is customer_id + t_dat proxy; association is not causality |
| Thresholds | `output/visualization/stage29/data/fig20_fp_growth_threshold_sensitivity.csv` | V1/V2 support ladder and counts | FOUND; selected support is read from artifact |
| Features | `output/features/erfm/` | customer_id, Recency, Frequency, Monetary, AvgBasketSize, ProductVariety, last_purchase_date | FOUND; used for real customer lookup |
| Predictions | `output/final/basic_kmeans_baseline_predictions/`, `main_kmeans_predictions/`, `enhanced_kmeans_predictions/` | customer_id and prediction/cluster | FOUND; B0/M1 may be unavailable outside the locked 1M cohort |
| Raw transactions | `output/clean/transactions/` | not loaded by dashboard | INTENTIONALLY NOT USED; prevents unnecessary runtime scan |

## Data contract notes

- Frequency is distinct purchase days, not invoice count.
- AvgBasketSize is the documented proxy based on product rows per distinct purchase day.
- No Accuracy, F1-score, or Confusion Matrix is shown because there is no ground-truth classification target.
- Spark `local[4]` is not presented as a four-node cluster.
- FP-Growth confidence is not model accuracy.
