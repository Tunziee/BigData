"""Artifact-only data loaders for the supplementary Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
OUTPUT_ROOT = PROJECT_ROOT / "output"


def artifact_path(*parts: str) -> Path:
    return OUTPUT_ROOT.joinpath(*parts)


@st.cache_data(show_spinner=False)
def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_overview() -> dict[str, Any]:
    kpi = read_json(artifact_path("sql", "stage27", "q1_dataset_kpi.json"))
    stage28 = read_json(artifact_path("fpgrowth", "stage28_v2", "stage28_v2_final_summary.json"))
    stage16 = read_json(artifact_path("analysis", "stage16", "cluster_profiles.json"))
    return {
        "transaction_rows": int(kpi["transaction_item_rows"]),
        "customers": int(kpi["distinct_customers"]),
        "articles": int(kpi["distinct_articles"]),
        "purchase_dates": int(kpi["distinct_purchase_dates"]),
        "start_date": kpi["min_purchase_date"],
        "end_date": kpi["max_purchase_date"],
        "clusters": int(stage16["cluster_count"]),
        "rules": int(stage28["association_rule_count"]),
        "monetary_sum": float(kpi["monetary_sum"]),
        "validation_pass": bool(kpi["validation_pass"]),
    }


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame:
    path = artifact_path("comparison", "stage22", "primary_three_model_comparison_matrix.csv")
    df = read_csv(path)
    df["ModelID"] = pd.Categorical(df["ModelID"], ["B0", "M1", "M2"], ordered=True)
    return df.sort_values("ModelID").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_quality_series() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = artifact_path("visualization", "stage29", "data")
    quality = read_csv(root / "fig05_primary_model_quality_comparison.csv")
    sse = read_csv(root / "fig06_common_space_sse_per_customer.csv")
    entropy = read_csv(root / "fig07a_cluster_entropy.csv").rename(columns={"model_id": "ModelID", "value": "Entropy"})
    cv = read_csv(root / "fig07b_cluster_cv.csv").rename(columns={"model_id": "ModelID", "value": "Cluster CV"})
    share = read_csv(root / "fig07c_cluster_share_range.csv").rename(columns={"model_id": "ModelID", "value": "Share range"})
    quality = quality.merge(entropy, on="ModelID").merge(cv, on="ModelID").merge(share, on="ModelID")
    return quality, sse, entropy


@st.cache_data(show_spinner=False)
def load_segments() -> pd.DataFrame:
    path = artifact_path("sql", "stage27", "q6_m2_segment_analytics.csv")
    df = read_csv(path)
    df = df.rename(columns={"prediction": "Cluster"})
    return df.sort_values("Cluster").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_transformation() -> pd.DataFrame:
    return read_csv(artifact_path("visualization", "stage29", "data", "fig02_skewness_transformation.csv"))


@st.cache_data(show_spinner=False)
def load_runtime() -> pd.DataFrame:
    return read_csv(artifact_path("visualization", "stage29", "data", "fig12_model_runtime_benchmark.csv"))


@st.cache_data(show_spinner=False)
def load_storage() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = artifact_path("visualization", "stage29", "data")
    return read_csv(root / "fig13_csv_parquet_storage.csv"), read_csv(root / "fig14_csv_parquet_runtime.csv")


@st.cache_data(show_spinner=False)
def load_cache() -> pd.DataFrame:
    return read_csv(artifact_path("visualization", "stage29", "data", "fig15_cache_vs_no_cache.csv"))


@st.cache_data(show_spinner=False)
def load_shuffle() -> pd.DataFrame:
    return read_csv(artifact_path("visualization", "stage29", "data", "fig16_spark_shuffle_q2_vs_q3.csv"))


@st.cache_data(show_spinner=False)
def load_sql() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = artifact_path("visualization", "stage29", "data")
    return (
        read_csv(root / "fig17_monthly_monetary_trend.csv"),
        read_csv(root / "fig18_top_product_groups.csv"),
        read_csv(root / "fig19_age_band_monetary.csv"),
        load_segments(),
    )


@st.cache_data(show_spinner=False)
def load_fp_growth() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root = artifact_path("visualization", "stage29", "data")
    thresholds = read_csv(root / "fig20_fp_growth_threshold_sensitivity.csv")
    parquet_path = artifact_path("fpgrowth", "stage28_v2", "selected_association_rules.parquet")
    csv_path = artifact_path("fpgrowth", "stage28_v2", "enriched_top_rules.csv")
    try:
        rules = read_parquet(parquet_path)
    except (ImportError, OSError, ValueError):
        rules = read_csv(csv_path)
    rules = normalize_rules(rules)
    summary = read_json(artifact_path("fpgrowth", "stage28_v2", "stage28_v2_final_summary.json"))
    return thresholds, rules, summary


@st.cache_data(show_spinner=False)
def load_cache_break_even() -> dict[str, Any]:
    return read_json(artifact_path("benchmark", "stage25", "cache_break_even_analysis.json"))


def _product_names(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item.get("prod_name", item.get("article_id", ""))) for item in value if isinstance(item, dict)]
    if isinstance(value, str):
        try:
            return _product_names(json.loads(value))
        except json.JSONDecodeError:
            return [value]
    return [str(value)] if pd.notna(value) else []


def normalize_rules(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in ["support", "confidence", "lift"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    if {"antecedent_products", "consequent_products"}.issubset(result.columns):
        result["Antecedent"] = result["antecedent_products"].map(lambda value: " + ".join(_product_names(value)))
        result["Consequent"] = result["consequent_products"].map(lambda value: " + ".join(_product_names(value)))
        result["Product group"] = result["antecedent_products"].map(_product_groups)
    else:
        result["Antecedent"] = result.get("antecedent_str", result["antecedent"].astype(str)).map(_clean_rule_id)
        result["Consequent"] = result.get("consequent_str", result["consequent"].astype(str)).map(_clean_rule_id)
        result["Product group"] = ""
        enrichment = artifact_path("fpgrowth", "stage28_v2", "enriched_top_rules.csv")
        if enrichment.exists():
            enriched = pd.read_csv(enrichment, dtype={"antecedent": str, "consequent": str})
            names: dict[tuple[str, str], tuple[str, str, str]] = {}
            for _, item in enriched.iterrows():
                a_names = _product_names(item.get("antecedent_products", ""))
                c_names = _product_names(item.get("consequent_products", ""))
                groups = _product_groups(item.get("antecedent_products", ""))
                names[(_clean_rule_id(item["antecedent"]), _clean_rule_id(item["consequent"]))] = (" + ".join(a_names), " + ".join(c_names), groups)
            for index, item in result.iterrows():
                key = (_clean_rule_id(item.get("antecedent_str", item.get("antecedent", ""))), _clean_rule_id(item.get("consequent_str", item.get("consequent", ""))))
                if key in names:
                    result.at[index, "Antecedent"], result.at[index, "Consequent"], result.at[index, "Product group"] = names[key]
    return result[["Antecedent", "Consequent", "support", "confidence", "lift", "Product group"]].sort_values("lift", ascending=False).reset_index(drop=True)


def _clean_rule_id(value: Any) -> str:
    text = str(value).strip().strip("[]()")
    text = text.replace("'", "").replace('"', "")
    return text.split(",")[0].strip()


def _product_groups(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return ""
    if not isinstance(value, list):
        return ""
    return ", ".join(sorted({str(item.get("product_group_name", "")) for item in value if isinstance(item, dict) and item.get("product_group_name")}))


@st.cache_data(show_spinner=False)
def load_story() -> pd.DataFrame:
    return read_csv(artifact_path("visualization", "stage29", "data", "fig24_project_analytics_story.csv"))
