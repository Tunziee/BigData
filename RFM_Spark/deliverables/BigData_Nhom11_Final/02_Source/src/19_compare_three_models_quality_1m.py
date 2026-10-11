"""Stage 19: formal three-model quality comparison on the locked 1M cohort.

This stage evaluates the locked Rule/R1, Main K-Means/K1, and Enhanced
K-Means/K2 assignments in the two common feature spaces.  It deliberately
does not fit or retrain any model and does not compute classification scores.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    abs as spark_abs,
    avg,
    col,
    count,
    element_at,
    lit,
    max as spark_max,
    min as spark_min,
    pow as spark_pow,
    sqrt,
    sum as spark_sum,
)
from pyspark.storagelevel import StorageLevel


EXPECTED_COUNT = 1_000_000
COHORT_FINGERPRINT = "ca082ec1d828df9c34a10105ae11599c3cc0653dc5b4d2c11af18ec82b781cf5"
COHORT_AUTHORITY = "DETERMINISTIC_1M_COMPARISON_COHORT"
PROTOCOL_AUTHORITY = "THREE_MODEL_COMPARISON_PROTOCOL_V1"
RULE_AUTHORITY = "RULE_BASED_RFM_BASELINE_V1"
METHOD_AUTHORITY = "RULE_BASED_RFM_QUINTILE_V1"
RAW_FEATURES = ["Recency", "Frequency", "Monetary", "AvgBasketSize", "ProductVariety"]
MODEL_SPECS = {
    "R1": {"name": "Rule-based RFM", "k": 5, "label": "rule_prediction"},
    "K1": {"name": "Main K-Means", "k": 4, "label": "main_prediction"},
    "K2": {"name": "Enhanced K-Means", "k": 5, "label": "enhanced_prediction"},
}
SPACE_SPECS = {
    "RFM3D": {"name": "COMMON_RFM_3D", "feature": "eval_rfm3d_scaled", "dimension": 3},
    "ERFM5D": {"name": "COMMON_ERFM_5D", "feature": "eval_erfm5d_scaled", "dimension": 5},
}
TOLERANCE = 1e-8
QUANTILE_ERROR = 0.001


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def enable_windows_java_filesystem_fallback(spark: SparkSession) -> None:
    jvm = spark._jvm
    shell_class = jvm.java.lang.Class.forName("org.apache.hadoop.util.Shell")
    windows_field = shell_class.getDeclaredField("WINDOWS")
    windows_field.setAccessible(True)
    try:
        windows_field.setBoolean(None, False)
    except Exception:
        unsafe_class = jvm.java.lang.Class.forName("sun.misc.Unsafe")
        unsafe_field = unsafe_class.getDeclaredField("theUnsafe")
        unsafe_field.setAccessible(True)
        unsafe = unsafe_field.get(None)
        unsafe.putBoolean(unsafe.staticFieldBase(windows_field), unsafe.staticFieldOffset(windows_field), False)


def audit_authorities(project_root: Path) -> List[str]:
    blockers: List[str] = []
    stage17 = read_json(project_root / "output" / "comparison" / "stage17" / "stage17_final_summary.json")
    stage18 = read_json(project_root / "output" / "comparison" / "stage18" / "stage18_final_summary.json")
    stage17_ok = (
        stage17.get("stage") == 17
        and stage17.get("cohort_authority") == COHORT_AUTHORITY
        and stage17.get("protocol_authority") == PROTOCOL_AUTHORITY
        and stage17.get("cohort_count") == EXPECTED_COUNT
        and stage17.get("cohort_customer_id_sha256") == COHORT_FINGERPRINT
        and stage17.get("source_alignment_pass") is True
        and stage17.get("deterministic_rerun_validation", {}).get("pass") is True
        and stage17.get("cohort_readback", {}).get("pass") is True
        and stage17.get("protocol_validation_pass") is True
        and stage17.get("ready_for_stage18") is True
    )
    stage18_ok = (
        stage18.get("stage") == 18
        and stage18.get("authority") == RULE_AUTHORITY
        and stage18.get("method_authority") == METHOD_AUTHORITY
        and stage18.get("cohort_authority") == COHORT_AUTHORITY
        and stage18.get("protocol_authority") == PROTOCOL_AUTHORITY
        and stage18.get("cohort_count") == EXPECTED_COUNT
        and stage18.get("cohort_customer_id_sha256") == COHORT_FINGERPRINT
        and stage18.get("prediction_validation", {}).get("pass") is True
        and stage18.get("reproducibility", {}).get("pass") is True
        and stage18.get("customer_universe_match", {}).get("pass") is True
        and stage18.get("prediction_readback", {}).get("pass") is True
        and stage18.get("ready_for_stage19") is True
    )
    if not stage17_ok:
        blockers.append("Stage 17 authority, fingerprint, or locked gate mismatch.")
    if not stage18_ok:
        blockers.append("Stage 18 authority, fingerprint, or locked gate mismatch.")
    return blockers


def validate_vectors(df: DataFrame, column: str, dimension: int) -> Dict[str, Any]:
    values = vector_to_array(col(column))
    result = {"column": column, "dimension": dimension}
    result["null_vector"] = df.where(col(column).isNull()).count()
    result["wrong_dimension"] = df.where(col(column).isNotNull() & (size_expr(values) != dimension)).count()
    result["nan_total"] = sum(df.where(element_at(values, i) != element_at(values, i)).count() for i in range(1, dimension + 1))
    result["infinity_total"] = sum(df.where((element_at(values, i) == float("inf")) | (element_at(values, i) == float("-inf"))).count() for i in range(1, dimension + 1))
    result["pass"] = all(result[key] == 0 for key in ["null_vector", "wrong_dimension", "nan_total", "infinity_total"])
    return result


def size_expr(values: Any) -> Any:
    # Kept as a helper to make vector validation readable without importing a
    # second alias named size in the public metric code.
    from pyspark.sql.functions import size
    return size(values)


def evaluate_silhouette(df: DataFrame, feature_col: str, label_col: str) -> float:
    evaluator = ClusteringEvaluator(
        predictionCol=label_col,
        featuresCol=feature_col,
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )
    return float(evaluator.evaluate(df.select(feature_col, label_col)))


def vector_components(df: DataFrame, feature_col: str, dimension: int, label_col: str) -> DataFrame:
    values = vector_to_array(col(feature_col))
    return df.select(label_col, *[element_at(values, i).alias(f"v{i - 1}") for i in range(1, dimension + 1)])


def compute_sse(df: DataFrame, feature_col: str, label_col: str, k: int, dimension: int) -> Dict[str, Any]:
    components = vector_components(df, feature_col, dimension, label_col).persist(StorageLevel.MEMORY_AND_DISK)
    component_names = [f"v{i}" for i in range(dimension)]
    centroid_rows = components.groupBy(label_col).agg(*[avg(col(name)).alias(name) for name in component_names]).orderBy(label_col).collect()
    if len(centroid_rows) != k or [int(row[label_col]) for row in centroid_rows] != list(range(k)):
        raise ValueError(f"Centroid coverage invalid for {label_col}: {[int(row[label_col]) for row in centroid_rows]}")
    centroids = {str(int(row[label_col])): [float(row[name]) for name in component_names] for row in centroid_rows}
    centroid_df = components.groupBy(label_col).agg(*[avg(col(name)).alias(f"c_{name}") for name in component_names])
    joined = components.join(centroid_df, label_col, "inner")
    squared = sum((col(name) - col(f"c_{name}")) * (col(name) - col(f"c_{name}")) for name in component_names)
    sse = float(joined.select(spark_sum(squared).alias("sse")).first()["sse"] or 0.0)
    components.unpersist()
    return {"sse": sse, "sse_per_customer": sse / EXPECTED_COUNT, "dimension": dimension, "segment_count": k, "customer_count": EXPECTED_COUNT, "centroids": centroids, "assignment_source": "actual locked assignments; centroids recomputed from assigned 1M rows", "training_cost_used": False}


def balance_metrics(df: DataFrame, label_col: str, k: int) -> Dict[str, Any]:
    rows = df.groupBy(label_col).count().orderBy(label_col).collect()
    counts = [int(row["count"]) for row in rows]
    if len(counts) != k or sum(counts) != EXPECTED_COUNT:
        raise ValueError(f"Cluster balance coverage invalid for {label_col}.")
    ratios = [value / EXPECTED_COUNT for value in counts]
    mean = sum(counts) / k
    population_std = math.sqrt(sum((value - mean) ** 2 for value in counts) / k)
    entropy = -sum(p * math.log(p) for p in ratios if p > 0)
    normalized = entropy / math.log(k)
    return {"cluster_count": k, "cluster_sizes": [{"cluster_id": i, "count": counts[i], "ratio": ratios[i], "ratio_pct": ratios[i] * 100.0} for i in range(k)], "min_count": min(counts), "max_count": max(counts), "min_ratio": min(ratios), "max_ratio": max(ratios), "min_ratio_pct": min(ratios) * 100.0, "max_ratio_pct": max(ratios) * 100.0, "below_1pct": sum(value < 0.01 * EXPECTED_COUNT for value in counts), "below_2pct": sum(value < 0.02 * EXPECTED_COUNT for value in counts), "below_5pct": sum(value < 0.05 * EXPECTED_COUNT for value in counts), "mean_count": mean, "population_std_count": population_std, "coefficient_of_variation": population_std / mean, "entropy_natural_log": entropy, "normalized_entropy": normalized}


def partition_agreement(df: DataFrame, left: str, right: str, left_k: int, right_k: int) -> Dict[str, Any]:
    contingency = df.groupBy(left, right).count().collect()
    table = [[0 for _ in range(right_k)] for _ in range(left_k)]
    for row in contingency:
        table[int(row[left])][int(row[right])] = int(row["count"])
    comb2 = lambda n: n * (n - 1) / 2.0
    n = float(EXPECTED_COUNT)
    row_sums = [sum(row) for row in table]
    col_sums = [sum(table[i][j] for i in range(left_k)) for j in range(right_k)]
    sum_nij = sum(comb2(value) for row in table for value in row)
    sum_rows = sum(comb2(value) for value in row_sums)
    sum_cols = sum(comb2(value) for value in col_sums)
    total_pairs = comb2(n)
    expected_index = (sum_rows * sum_cols) / total_pairs
    max_index = 0.5 * (sum_rows + sum_cols)
    ari = (sum_nij - expected_index) / (max_index - expected_index) if max_index != expected_index else 1.0
    p_rows = [value / n for value in row_sums]
    p_cols = [value / n for value in col_sums]
    mi = 0.0
    for i in range(left_k):
        for j in range(right_k):
            value = table[i][j]
            if value:
                p_ij = value / n
                mi += p_ij * math.log(p_ij / (p_rows[i] * p_cols[j]))
    h_left = -sum(p * math.log(p) for p in p_rows if p > 0)
    h_right = -sum(p * math.log(p) for p in p_cols if p > 0)
    nmi = (2.0 * mi / (h_left + h_right)) if (h_left + h_right) else 1.0
    return {"left_model": left, "right_model": right, "left_k": left_k, "right_k": right_k, "customer_count": EXPECTED_COUNT, "contingency_table": table, "row_sums": row_sums, "column_sums": col_sums, "adjusted_rand_index": ari, "mutual_information_natural_log": mi, "entropy_left_natural_log": h_left, "entropy_right_natural_log": h_right, "normalized_mutual_information_arithmetic_mean": nmi, "nmi_definition": "MI / ((H(left)+H(right))/2), natural logs", "ari_definition": "Manual Hubert-Ar 둘? ARI from pair-count contingency formula".replace("Hubert-Ar 둘?", "Hubert-Arnold")}


def profile_one_segment(segment_df: DataFrame) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for feature in RAW_FEATURES:
        row = segment_df.agg(avg(col(feature)).alias("mean"), spark_min(col(feature)).alias("min"), spark_max(col(feature)).alias("max"), count(col(feature)).alias("count")).first()
        quantiles = segment_df.approxQuantile(feature, [0.05, 0.25, 0.5, 0.75, 0.95], QUANTILE_ERROR)
        result[feature] = {"count": int(row["count"]), "mean": float(row["mean"]), "min": float(row["min"]), "p05": float(quantiles[0]), "p25": float(quantiles[1]), "median": float(quantiles[2]), "p75": float(quantiles[3]), "p95": float(quantiles[4]), "max": float(row["max"]), "quantile_relative_error": QUANTILE_ERROR}
    return result


def raw_profiles(df: DataFrame, model_id: str, label_col: str, k: int, names: Dict[str, Any]) -> Dict[str, Any]:
    segments: Dict[str, Any] = {}
    for segment_id in range(k):
        part = df.where(col(label_col) == segment_id)
        count_value = part.count()
        profile = profile_one_segment(part)
        name = names.get(str(segment_id), {})
        segments[str(segment_id)] = {"segment_id": segment_id, "segment_name_vi": name.get("name_vi", name.get("segment_name_vi", "")), "segment_name_en": name.get("name_en", name.get("segment_name_en", "")), "count": count_value, "ratio_pct": count_value / EXPECTED_COUNT * 100.0, "raw_features": profile}
    return {"model_id": model_id, "model_name": MODEL_SPECS[model_id]["name"], "feature_order": RAW_FEATURES, "segments": segments}


def distinctness(profiles: Dict[str, Any], global_iqr: Dict[str, float]) -> Dict[str, Any]:
    rows = []
    ids = sorted(profiles["segments"], key=int)
    for left, right in combinations(ids, 2):
        terms = []
        for feature in RAW_FEATURES:
            iqr = global_iqr[feature]
            delta = abs(profiles["segments"][left]["raw_features"][feature]["median"] - profiles["segments"][right]["raw_features"][feature]["median"])
            terms.append(0.0 if iqr == 0 else delta / iqr)
        distance = math.sqrt(sum(term * term for term in terms))
        rows.append({"segment_a": int(left), "segment_b": int(right), "normalized_median_distance": distance})
    values = [row["normalized_median_distance"] for row in rows]
    closest = min(rows, key=lambda item: item["normalized_median_distance"])
    farthest = max(rows, key=lambda item: item["normalized_median_distance"])
    return {"model_id": profiles["model_id"], "global_iqr": global_iqr, "pairwise_distances": rows, "min": min(values), "mean": sum(values) / len(values), "median": sorted(values)[len(values) // 2] if len(values) % 2 else (sorted(values)[len(values) // 2 - 1] + sorted(values)[len(values) // 2]) / 2.0, "max": max(values), "closest_pair": closest, "farthest_pair": farthest, "normalization": "Euclidean distance of raw-feature median differences divided by global 1M-cohort IQR per feature; zero IQR contributes zero."}


def criterion_winners(matrix: List[Dict[str, Any]], distinct: Dict[str, Any]) -> Dict[str, Any]:
    def winner(field: str, reverse: bool = True) -> Dict[str, Any]:
        item = sorted(matrix, key=lambda row: float(row[field]), reverse=reverse)[0]
        return {"model_id": item["ModelID"], "model_name": item["ModelName"], "value": item[field], "metric": field}
    winners = {"highest_silhouette_rfm3d": winner("Silhouette_RFM3D"), "highest_silhouette_erfm5d": winner("Silhouette_ERFM5D"), "lowest_sse_per_customer_rfm3d": winner("SSEPerCustomer_RFM3D", False), "lowest_sse_per_customer_erfm5d": winner("SSEPerCustomer_ERFM5D", False), "highest_normalized_entropy": winner("NormalizedEntropy"), "lowest_cluster_size_cv": winner("ClusterSizeCV", False), "largest_raw_distinctness_min": winner("RawDistinctnessMin"), "largest_raw_distinctness_mean": winner("RawDistinctnessMean")}
    winner_ids = {item["model_id"] for item in winners.values()}
    return {"criterion_winners": winners, "winner_model_ids": sorted(winner_ids), "no_single_universal_winner": len(winner_ids) > 1, "weighted_score_used": False, "classification_accuracy_used": False}


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "output" / "comparison" / "stage19"
    report_path = project_root / "reports" / "stage19_three_model_quality_comparison.txt"
    summary_path = output_dir / "stage19_final_summary.json"
    blockers: List[str] = []
    warnings: List[str] = []
    spark = None
    evaluation = None
    try:
        blockers.extend(audit_authorities(project_root))
        if blockers:
            failed = {"stage": 19, "authority": "THREE_MODEL_QUALITY_COMPARISON_V1", "formal_quality_comparison_complete": False, "ready_for_stage20": False, "blockers": blockers}
            write_json(summary_path, failed)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("STAGE 19 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_20: False\n", encoding="utf-8")
            return 3

        stage11_names = read_json(project_root / "output" / "analysis" / "stage11" / "segment_names.json").get("segments", {})
        stage16_names = read_json(project_root / "output" / "analysis" / "stage16" / "segment_names.json").get("segments", {})
        rule_names = {"0": {"name_vi": "RFM rất thấp", "name_en": "Very Low RFM"}, "1": {"name_vi": "RFM thấp", "name_en": "Low RFM"}, "2": {"name_vi": "RFM trung bình", "name_en": "Medium RFM"}, "3": {"name_vi": "RFM cao", "name_en": "High RFM"}, "4": {"name_vi": "RFM rất cao", "name_en": "Very High RFM"}}
        names_by_model = {"R1": rule_names, "K1": stage11_names, "K2": stage16_names}

        spark = SparkSession.builder.master("local[4]").appName("Stage19-Three-Model-Quality-Comparison").config("spark.sql.shuffle.partitions", "64").config("spark.sql.adaptive.enabled", "true").config("spark.local.dir", str(project_root.parent / "spark_tmp")).getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        enable_windows_java_filesystem_fallback(spark)
        cohort_path = project_root / "output" / "comparison" / "stage17" / "comparison_cohort_1m"
        rule_path = project_root / "output" / "comparison" / "stage18" / "rule_based_rfm_predictions_1m"
        cohort = spark.read.parquet(str(cohort_path))
        rule = spark.read.parquet(str(rule_path)).select("customer_id", col("rule_segment_id").alias("rule_prediction"))
        needed = ["customer_id", *RAW_FEATURES, "main_prediction", "enhanced_prediction", "eval_rfm3d_scaled", "eval_erfm5d_scaled"]
        missing = [column for column in needed if column not in cohort.columns]
        if missing:
            raise ValueError(f"Locked Stage 17 cohort missing required columns: {missing}")
        evaluation = cohort.select(*needed).join(rule, "customer_id", "inner").persist(StorageLevel.MEMORY_AND_DISK)
        evaluation.count()
        labels_ok = all(evaluation.where(col(spec["label"]).isNull() | (~col(spec["label"]).isin(list(range(spec["k"]))))).count() == 0 for spec in MODEL_SPECS.values())
        row_count = evaluation.count()
        distinct_count = evaluation.select("customer_id").distinct().count()
        duplicate_count = row_count - distinct_count
        vector_checks = {key: validate_vectors(evaluation, spec["feature"], spec["dimension"]) for key, spec in SPACE_SPECS.items()}
        null_cells = {column: evaluation.where(col(column).isNull()).count() for column in [*RAW_FEATURES, "rule_prediction", "main_prediction", "enhanced_prediction"]}
        validation = {"row_count": row_count, "distinct_customer_id": distinct_count, "duplicate_customer_id": duplicate_count, "expected_row_count": EXPECTED_COUNT, "labels_valid": labels_ok, "null_cells": null_cells, "vector_checks": vector_checks, "pass": row_count == EXPECTED_COUNT and distinct_count == EXPECTED_COUNT and duplicate_count == 0 and labels_ok and all(value == 0 for value in null_cells.values()) and all(item["pass"] for item in vector_checks.values()), "persist_level": "MEMORY_AND_DISK", "materialized_before_metrics": True}
        write_json(output_dir / "comparison_dataset_validation.json", validation)
        if not validation["pass"]:
            raise ValueError("Evaluation dataset validation failed.")

        silhouettes: Dict[str, Any] = {}
        sses: Dict[str, Any] = {}
        balances: Dict[str, Any] = {}
        for model_id, model in MODEL_SPECS.items():
            silhouettes[model_id] = {space_id: {"model_id": model_id, "model_name": model["name"], "space": SPACE_SPECS[space_id]["name"], "feature_column": SPACE_SPECS[space_id]["feature"], "silhouette": evaluate_silhouette(evaluation, SPACE_SPECS[space_id]["feature"], model["label"]), "distance_measure": "squaredEuclidean", "metric_name": "silhouette", "customer_count": EXPECTED_COUNT} for space_id in SPACE_SPECS}
            balances[model_id] = balance_metrics(evaluation, model["label"], model["k"])
            sses[model_id] = {space_id: compute_sse(evaluation, SPACE_SPECS[space_id]["feature"], model["label"], model["k"], SPACE_SPECS[space_id]["dimension"]) for space_id in SPACE_SPECS}
        write_json(output_dir / "common_space_silhouette.json", silhouettes)
        write_json(output_dir / "common_space_sse.json", sses)
        write_json(output_dir / "cluster_balance_metrics.json", balances)

        agreement: Dict[str, Any] = {}
        for left, right in [("R1", "K1"), ("R1", "K2"), ("K1", "K2")]:
            agreement[f"{left}_vs_{right}"] = partition_agreement(evaluation, MODEL_SPECS[left]["label"], MODEL_SPECS[right]["label"], MODEL_SPECS[left]["k"], MODEL_SPECS[right]["k"])
        write_json(output_dir / "partition_agreement.json", agreement)
        write_json(output_dir / "contingency_tables.json", {key: {"left_model": value["left_model"], "right_model": value["right_model"], "table": value["contingency_table"], "row_sums": value["row_sums"], "column_sums": value["column_sums"]} for key, value in agreement.items()})

        profiles = {model_id: raw_profiles(evaluation, model_id, model["label"], model["k"], names_by_model[model_id]) for model_id, model in MODEL_SPECS.items()}
        global_iqr = {feature: float(evaluation.approxQuantile(feature, [0.75], QUANTILE_ERROR)[0] - evaluation.approxQuantile(feature, [0.25], QUANTILE_ERROR)[0]) for feature in RAW_FEATURES}
        write_json(output_dir / "raw_profiles_1m.json", {"cohort_count": EXPECTED_COUNT, "global_iqr": global_iqr, "models": profiles})
        profile_rows = []
        for model_id, payload in profiles.items():
            for segment_id, segment in payload["segments"].items():
                for feature, stats in segment["raw_features"].items():
                    profile_rows.append({"ModelID": model_id, "ModelName": payload["model_name"], "SegmentId": int(segment_id), "SegmentNameVI": segment["segment_name_vi"], "SegmentNameEN": segment["segment_name_en"], "CustomerCount": segment["count"], "RatioPct": segment["ratio_pct"], "Feature": feature, **{key: value for key, value in stats.items() if key != "quantile_relative_error"}})
        write_csv(output_dir / "raw_profiles_1m.csv", ["ModelID", "ModelName", "SegmentId", "SegmentNameVI", "SegmentNameEN", "CustomerCount", "RatioPct", "Feature", "count", "mean", "min", "p05", "p25", "median", "p75", "p95", "max"], profile_rows)
        distinct = {model_id: distinctness(payload, global_iqr) for model_id, payload in profiles.items()}
        write_json(output_dir / "raw_profile_distinctness.json", distinct)

        matrix = []
        interpretation = {"R1": "Transparent ordinal quintile baseline with directly explainable R/F/M bands.", "K1": "Compactness and four-segment behavioral grouping from the locked 3D RFM model; names remain descriptive, not causal.", "K2": "Five-dimensional behavioral grouping using recency, frequency, monetary, basket size, and product variety; names remain descriptive, not causal."}
        for model_id, model in MODEL_SPECS.items():
            matrix.append({"ModelID": model_id, "ModelName": model["name"], "K": model["k"], "Silhouette_RFM3D": silhouettes[model_id]["RFM3D"]["silhouette"], "Silhouette_ERFM5D": silhouettes[model_id]["ERFM5D"]["silhouette"], "SSE_RFM3D": sses[model_id]["RFM3D"]["sse"], "SSEPerCustomer_RFM3D": sses[model_id]["RFM3D"]["sse_per_customer"], "SSE_ERFM5D": sses[model_id]["ERFM5D"]["sse"], "SSEPerCustomer_ERFM5D": sses[model_id]["ERFM5D"]["sse_per_customer"], "MinClusterRatioPct": balances[model_id]["min_ratio_pct"], "MaxClusterRatioPct": balances[model_id]["max_ratio_pct"], "ClusterSizeCV": balances[model_id]["coefficient_of_variation"], "NormalizedEntropy": balances[model_id]["normalized_entropy"], "RawDistinctnessMin": distinct[model_id]["min"], "RawDistinctnessMean": distinct[model_id]["mean"], "InterpretabilitySummary": interpretation[model_id]})
        matrix_fields = ["ModelID", "ModelName", "K", "Silhouette_RFM3D", "Silhouette_ERFM5D", "SSE_RFM3D", "SSEPerCustomer_RFM3D", "SSE_ERFM5D", "SSEPerCustomer_ERFM5D", "MinClusterRatioPct", "MaxClusterRatioPct", "ClusterSizeCV", "NormalizedEntropy", "RawDistinctnessMin", "RawDistinctnessMean", "InterpretabilitySummary"]
        write_csv(output_dir / "three_model_comparison_matrix.csv", matrix_fields, matrix)
        findings = criterion_winners(matrix, distinct)
        findings["interpretability_evidence"] = interpretation
        findings["recommendation"] = "NO_SINGLE_UNIVERSAL_WINNER" if findings["no_single_universal_winner"] else f"{next(iter(findings['winner_model_ids']))}_MODEL_BY_CRITERIA"
        findings["use_case_guidance"] = {"compactness": "Select the criterion winner separately for the required common space.", "balance": "Use the highest normalized entropy / lowest CV winner when balanced coverage is the priority.", "interpretability": "Use R1 when transparent ordinal rules are the primary requirement; use K1/K2 when behavioral grouping is the requirement and profile evidence supports it."}
        write_json(output_dir / "criterion_findings.json", findings)

        # Independent recomputation of all six silhouettes, three entropies,
        # three ARIs and three NMIs. No cached metric is reused.
        repro_sil = {model_id: {space_id: evaluate_silhouette(evaluation, space["feature"], model["label"]) for space_id, space in SPACE_SPECS.items()} for model_id, model in MODEL_SPECS.items()}
        repro_bal = {model_id: balance_metrics(evaluation, model["label"], model["k"])["normalized_entropy"] for model_id, model in MODEL_SPECS.items()}
        repro_agreement = {key: partition_agreement(evaluation, value["left_model"], value["right_model"], value["left_k"], value["right_k"]) for key, value in agreement.items()}
        diffs = {"silhouette": {m: {s: abs(silhouettes[m][s]["silhouette"] - repro_sil[m][s]) for s in SPACE_SPECS} for m in MODEL_SPECS}, "normalized_entropy": {m: abs(balances[m]["normalized_entropy"] - repro_bal[m]) for m in MODEL_SPECS}, "ari": {key: abs(agreement[key]["adjusted_rand_index"] - repro_agreement[key]["adjusted_rand_index"]) for key in agreement}, "nmi": {key: abs(agreement[key]["normalized_mutual_information_arithmetic_mean"] - repro_agreement[key]["normalized_mutual_information_arithmetic_mean"]) for key in agreement}}
        repro = {"tolerance": TOLERANCE, "independent_recomputation": {"silhouette": repro_sil, "normalized_entropy": repro_bal, "partition_agreement": {key: {"ari": value["adjusted_rand_index"], "nmi": value["normalized_mutual_information_arithmetic_mean"]} for key, value in repro_agreement.items()}}, "absolute_differences": diffs}
        # Use an explicit scalar list so the pass condition is auditable.
        scalar_diffs = [diff for model in diffs["silhouette"].values() for diff in model.values()] + list(diffs["normalized_entropy"].values()) + list(diffs["ari"].values()) + list(diffs["nmi"].values())
        repro["max_absolute_difference"] = max(scalar_diffs or [0.0])
        repro["pass"] = repro["max_absolute_difference"] <= TOLERANCE
        write_json(output_dir / "reproducibility_check.json", repro)
        if not repro["pass"]:
            raise ValueError("Independent Stage 19 metric recomputation failed.")

        all_finite = all(finite(value) for model in matrix for key, value in model.items() if key not in ["ModelID", "ModelName", "InterpretabilitySummary"])
        summary = {"stage": 19, "authority": "THREE_MODEL_QUALITY_COMPARISON_V1", "cohort_authority": COHORT_AUTHORITY, "protocol_authority": PROTOCOL_AUTHORITY, "rule_authority": RULE_AUTHORITY, "rule_method_authority": METHOD_AUTHORITY, "cohort_count": EXPECTED_COUNT, "cohort_customer_id_sha256": COHORT_FINGERPRINT, "models": {key: {"model_name": value["name"], "k": value["k"], "label_column": value["label"]} for key, value in MODEL_SPECS.items()}, "common_spaces": SPACE_SPECS, "comparison_matrix": matrix, "criterion_findings": findings, "reproducibility": repro, "all_metrics_finite": all_finite, "classification_accuracy_used": False, "weighted_winner_score_used": False, "models_retrained": False, "gmm_executed": False, "formal_quality_comparison_complete": True, "ready_for_stage20": True, "warnings": warnings, "blockers": blockers}
        write_json(summary_path, summary)
        report_lines = ["STAGE 19 — FORMAL THREE-MODEL QUALITY COMPARISON", "", f"AUTHORITY: {summary['authority']}", f"COHORT: {COHORT_AUTHORITY} / {EXPECTED_COUNT} / {COHORT_FINGERPRINT}", f"MODELS: {summary['models']}", "", "COMPARISON MATRIX:", *[str(row) for row in matrix], "", "CRITERION FINDINGS:", str(findings), "", f"REPRODUCIBILITY_PASS: {repro['pass']}", "CLASSIFICATION_ACCURACY_USED: False", "WEIGHTED_WINNER_SCORE_USED: False", "MODELS_RETRAINED: False", "GMM_EXECUTED: False", "", "FORMAL_QUALITY_COMPARISON_COMPLETE: True", "READY_FOR_STAGE_20: True"]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

        artifact_paths = [output_dir / name for name in ["comparison_dataset_validation.json", "common_space_silhouette.json", "common_space_sse.json", "cluster_balance_metrics.json", "partition_agreement.json", "contingency_tables.json", "raw_profiles_1m.json", "raw_profiles_1m.csv", "raw_profile_distinctness.json", "three_model_comparison_matrix.csv", "criterion_findings.json", "reproducibility_check.json", "stage19_final_summary.json"]] + [report_path, project_root / "reports" / "stage19_run.log"]
        readback: Dict[str, bool] = {}
        for path in artifact_paths:
            key = str(path.relative_to(project_root))
            ok = path.exists() and path.stat().st_size > 0
            if ok and path.suffix == ".json":
                try:
                    json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    ok = False
            if ok and path.suffix == ".csv":
                ok = len(path.read_text(encoding="utf-8").splitlines()) >= 2
            readback[key] = ok
        summary["artifact_readback"] = readback
        summary["artifact_readback_pass"] = all(readback.values())
        summary["ready_for_stage20"] = summary["artifact_readback_pass"]
        write_json(summary_path, summary)
        print("AUTHORITY_AUDIT_PASS=True")
        print("EVALUATION_DATASET_VALIDATION_PASS=True")
        print("FORMAL_QUALITY_COMPARISON_COMPLETE=True")
        print("READY_FOR_STAGE_20=True")
        return 0
    except Exception as error:
        blockers.append(str(error))
        failed = {"stage": 19, "authority": "THREE_MODEL_QUALITY_COMPARISON_V1", "formal_quality_comparison_complete": False, "ready_for_stage20": False, "warnings": warnings, "blockers": blockers}
        write_json(summary_path, failed)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("STAGE 19 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_20: False\n", encoding="utf-8")
        print(f"STAGE 19 FAILED CLOSED: {error}")
        return 3
    finally:
        if evaluation is not None:
            evaluation.unpersist()
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    sys.exit(main())
