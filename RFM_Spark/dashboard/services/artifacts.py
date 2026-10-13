from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "output"


def artifact(*parts: str) -> Path:
    return OUTPUT_ROOT.joinpath(*parts)


@st.cache_data(show_spinner=False)
def read_json(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def read_csv(path: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


@st.cache_data(show_spinner=False)
def read_parquet(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def load_model_quality() -> pd.DataFrame:
    return read_csv(str(artifact("comparison", "stage22", "primary_three_model_comparison_matrix.csv")))


def load_common_sse() -> pd.DataFrame:
    return read_csv(str(artifact("visualization", "stage29", "data", "fig06_common_space_sse_per_customer.csv")))


def load_entropy_balance() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = artifact("visualization", "stage29", "data")
    entropy = read_csv(str(root / "fig07a_cluster_entropy.csv")).rename(columns={"model_id": "ModelID", "value": "Entropy"})
    cv = read_csv(str(root / "fig07b_cluster_cv.csv")).rename(columns={"model_id": "ModelID", "value": "ClusterCV"})
    share = read_csv(str(root / "fig07c_cluster_share_range.csv")).rename(columns={"model_id": "ModelID", "value": "ShareRangePct"})
    return entropy, cv, share


def load_raw_profiles(model_id: str) -> pd.DataFrame:
    raw = read_csv(str(artifact("comparison", "stage22", "raw_profiles_1m.csv")))
    raw = raw[raw["ModelID"].eq(model_id)].copy()
    summary = raw.drop_duplicates(["SegmentId", "SegmentNameVI", "SegmentNameEN", "CustomerCount", "RatioPct"])
    medians = raw.pivot_table(index=["SegmentId", "SegmentNameVI", "SegmentNameEN", "CustomerCount", "RatioPct"], columns="Feature", values="median").reset_index()
    return summary.merge(medians, on=["SegmentId", "SegmentNameVI", "SegmentNameEN", "CustomerCount", "RatioPct"], how="left")


def load_m2_segments() -> pd.DataFrame:
    df = read_csv(str(artifact("sql", "stage27", "q6_m2_segment_analytics.csv"))).rename(columns={"prediction": "SegmentId"})
    return df.sort_values("SegmentId").reset_index(drop=True)


def load_segments(model_id: str) -> pd.DataFrame:
    return load_m2_segments() if model_id == "M2" else load_raw_profiles(model_id)


def load_runtime() -> pd.DataFrame:
    return read_csv(str(artifact("visualization", "stage29", "data", "fig12_model_runtime_benchmark.csv")))


def load_storage() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = artifact("visualization", "stage29", "data")
    return read_csv(str(root / "fig13_csv_parquet_storage.csv")), read_csv(str(root / "fig14_csv_parquet_runtime.csv"))


def load_cache() -> tuple[pd.DataFrame, dict[str, Any]]:
    df = read_csv(str(artifact("visualization", "stage29", "data", "fig15_cache_vs_no_cache.csv")))
    summary = read_json(str(artifact("benchmark", "stage25", "cache_break_even_analysis.json")))
    return df, summary


def load_execution() -> dict[str, Any]:
    return read_json(str(artifact("analysis", "stage26", "stage26_final_summary.json")))


def load_shuffle_metrics() -> pd.DataFrame:
    raw = read_json(str(artifact("analysis", "stage26", "shuffle_metrics.json")))
    rows = []
    for query, values in raw.items():
        rows.append({"Query": query, **values})
    return pd.DataFrame(rows)


def load_stage27_summary() -> dict[str, Any]:
    return read_json(str(artifact("sql", "stage27", "stage27_final_summary.json")))


def load_sql() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = artifact("visualization", "stage29", "data")
    return (
        read_csv(str(root / "fig17_monthly_monetary_trend.csv")),
        read_csv(str(root / "fig18_top_product_groups.csv")),
        read_csv(str(root / "fig19_age_band_monetary.csv")),
        load_m2_segments(),
    )


def load_fp_growth() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root = artifact("visualization", "stage29", "data")
    thresholds = read_csv(str(root / "fig20_fp_growth_threshold_sensitivity.csv"))
    summary = read_json(str(artifact("fpgrowth", "stage28_v2", "stage28_v2_final_summary.json")))
    try:
        rules = read_parquet(str(artifact("fpgrowth", "stage28_v2", "selected_association_rules.parquet")))
        rules = normalize_rules(rules)
    except (ImportError, OSError, ValueError):
        rules = read_csv(str(artifact("fpgrowth", "stage28_v2", "enriched_top_rules.csv")))
        rules = normalize_rules(rules)
    return thresholds, rules, summary


def normalize_rules(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in ["support", "confidence", "lift"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    if "antecedent_products" in result.columns:
        result["Antecedent"] = result["antecedent_products"].astype(str)
        result["Consequent"] = result["consequent_products"].astype(str)
        result["Product group"] = ""
    else:
        result["Antecedent"] = result.get("antecedent_str", result["antecedent"].astype(str)).astype(str)
        result["Consequent"] = result.get("consequent_str", result["consequent"].astype(str)).astype(str)
        result["Product group"] = ""
    return result[["Antecedent", "Consequent", "support", "confidence", "lift", "Product group"]].sort_values("lift", ascending=False).reset_index(drop=True)


def _dataset(path: Path):
    import pyarrow.dataset as ds
    return ds.dataset(str(path), format="parquet")


def _customer_scalar(dataset, customer_id: str):
    import pyarrow as pa
    field = dataset.schema.field("customer_id")
    if pa.types.is_integer(field.type):
        return int(customer_id)
    return customer_id


def scan_customer(path: Path, customer_id: str, columns: list[str] | None = None) -> dict[str, Any] | None:
    if not path.exists():
        return None
    dataset = _dataset(path)
    if "customer_id" not in dataset.schema.names:
        return None
    import pyarrow.dataset as ds
    selected = [c for c in (columns or dataset.schema.names) if c in dataset.schema.names]
    table = dataset.to_table(filter=ds.field("customer_id") == _customer_scalar(dataset, customer_id), columns=selected)
    rows = table.to_pylist()
    return rows[0] if rows else None


def sample_customer_id() -> str | None:
    path = artifact("features", "erfm")
    if not path.exists():
        return None
    dataset = _dataset(path)
    rows = dataset.head(1, columns=["customer_id"]).to_pylist()
    return str(rows[0]["customer_id"]) if rows else None


def load_customer_lookup(customer_id: str) -> dict[str, Any]:
    feature_cols = ["customer_id", "Recency", "Frequency", "Monetary", "AvgBasketSize", "ProductVariety", "last_purchase_date"]
    features = scan_customer(artifact("features", "erfm"), customer_id, feature_cols)
    predictions: dict[str, dict[str, Any] | None] = {}
    for model_id, folder in {"B0": "basic_kmeans_baseline_predictions", "M1": "main_kmeans_predictions", "M2": "enhanced_kmeans_predictions"}.items():
        predictions[model_id] = scan_customer(artifact("final", folder), customer_id, ["customer_id", "prediction", "cluster"])
    return {"customer_id": customer_id, "features": features, "predictions": predictions}
