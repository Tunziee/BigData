"""Stage 22: primary formal B0/M1/M2 quality comparison on the locked 1M cohort.

All assignments are loaded from locked artifacts.  This stage computes
quality metrics only; it does not fit, tune, or benchmark any model.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import avg, col, count, element_at, max as spark_max, min as spark_min, sum as spark_sum
from pyspark.storagelevel import StorageLevel


COHORT_COUNT = 1_000_000
COHORT_FINGERPRINT = "ca082ec1d828df9c34a10105ae11599c3cc0653dc5b4d2c11af18ec82b781cf5"
RAW_FEATURES = ["Recency", "Frequency", "Monetary", "AvgBasketSize", "ProductVariety"]
MODEL_SPECS = {
    "B0": {"name": "Basic K-Means Baseline", "k": 4, "label": "b0_prediction", "training_space": "RFM3D", "init": "random", "role": "technical_baseline"},
    "M1": {"name": "Project Main K-Means", "k": 4, "label": "main_prediction", "training_space": "RFM3D", "init": "k-means||", "role": "project_main"},
    "M2": {"name": "Enhanced K-Means", "k": 5, "label": "enhanced_prediction", "training_space": "eRFM5D", "init": "k-means||", "role": "enhanced_model"},
}
SPACE_SPECS = {
    "RFM3D": {"name": "COMMON_RFM_3D", "feature": "eval_rfm3d_scaled", "dimension": 3},
    "ERFM5D": {"name": "COMMON_ERFM_5D", "feature": "eval_erfm5d_scaled", "dimension": 5},
}
TOLERANCE = 1e-8
QUANTILE_ERROR = 0.001
HISTORICAL = {"M1_RFM3D": 0.6858449879433209, "M1_ERFM5D": 0.3565725452534894, "M2_RFM3D": 0.21521780350092465, "M2_ERFM5D": 0.481806967562302, "B0_RFM3D": 0.6858111002170737, "B0_SSE": 769136.5781161708, "B0_SSE_PER_CUSTOMER": 0.7691365781161709, "B0_ENTROPY": 0.7569022092473077}


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


def audit_authorities(project_root: Path) -> Tuple[List[str], Dict[str, Any]]:
    blockers: List[str] = []
    stage17 = read_json(project_root / "output" / "comparison" / "stage17" / "stage17_final_summary.json")
    stage21 = read_json(project_root / "output" / "analysis" / "stage21" / "stage21_final_summary.json")
    stage10 = read_json(project_root / "output" / "analysis" / "stage10" / "final_model_summary.json")
    stage15 = read_json(project_root / "output" / "analysis" / "stage15" / "final_enhanced_model_summary.json")
    stage19 = read_json(project_root / "output" / "comparison" / "stage19" / "stage19_final_summary.json")
    stage17_ok = stage17.get("stage") == 17 and stage17.get("cohort_authority") == "DETERMINISTIC_1M_COMPARISON_COHORT" and stage17.get("cohort_count") == COHORT_COUNT and stage17.get("cohort_customer_id_sha256") == COHORT_FINGERPRINT and stage17.get("ready_for_stage18") is True
    stage21_ok = stage21.get("stage") == 21 and stage21.get("authority") == "BASIC_KMEANS_BASELINE_V1" and stage21.get("full_training_count") == 1362281 and stage21.get("comparison_cohort_count") == COHORT_COUNT and stage21.get("b0_prediction_sha256") == "98e75c0b34cff5ffade7e8b62cd3176b2983062c414b2cf62be14f3f0819bf09" and stage21.get("primary_reproducibility_pass") is True and stage21.get("customer_universe_match_pass") is True and stage21.get("ready_for_stage22") is True
    stage10_ok = stage10.get("stage") == 10 and stage10.get("authority") == "MAIN_KMEANS_FINAL" and stage10.get("selected_k") == 4 and stage10.get("ready_for_stage11") is True
    stage15_ok = stage15.get("stage") == 15 and stage15.get("authority") == "ENHANCED_KMEANS_FINAL" and stage15.get("selected_k") == 5 and stage15.get("ready_for_stage16") is True
    stage19_ok = stage19.get("stage") == 19 and stage19.get("authority") == "THREE_MODEL_QUALITY_COMPARISON_V1" and stage19.get("formal_quality_comparison_complete") is True
    if not stage17_ok:
        blockers.append("Stage 17 authority, fingerprint, or gate mismatch.")
    if not stage21_ok:
        blockers.append("Stage 21 B0 authority, fingerprint, reproducibility, or gate mismatch.")
    if not stage10_ok:
        blockers.append("Stage 10 M1 authority mismatch.")
    if not stage15_ok:
        blockers.append("Stage 15 M2 authority mismatch.")
    if not stage19_ok:
        blockers.append("Stage 19 historical quality authority mismatch.")
    return blockers, {"stage17": stage17, "stage21": stage21, "stage10": stage10, "stage15": stage15, "stage19": stage19}


def validate_vector(df: DataFrame, column: str, dimension: int) -> Dict[str, Any]:
    values = vector_to_array(col(column))
    from pyspark.sql.functions import size
    null_count = df.where(col(column).isNull()).count()
    wrong_dimension = df.where(col(column).isNotNull() & (size(values) != dimension)).count()
    nan_count = sum(df.where(element_at(values, index) != element_at(values, index)).count() for index in range(1, dimension + 1))
    infinity_count = sum(df.where((element_at(values, index) == float("inf")) | (element_at(values, index) == float("-inf"))).count() for index in range(1, dimension + 1))
    return {"column": column, "dimension": dimension, "null": null_count, "wrong_dimension": wrong_dimension, "nan": nan_count, "infinity": infinity_count, "pass": null_count == 0 and wrong_dimension == 0 and nan_count == 0 and infinity_count == 0}


def dataset_validation(df: DataFrame) -> Dict[str, Any]:
    row_count = df.count()
    distinct_count = df.select("customer_id").distinct().count()
    label_checks = {}
    for model_id, spec in MODEL_SPECS.items():
        label_checks[model_id] = {"null": df.where(col(spec["label"]).isNull()).count(), "invalid": df.where(~col(spec["label"]).isin(list(range(spec["k"])))).count()}
    vector_checks = {space_id: validate_vector(df, space["feature"], space["dimension"]) for space_id, space in SPACE_SPECS.items()}
    return {"row_count": row_count, "distinct_customer_id": distinct_count, "duplicate_customer_id": row_count - distinct_count, "label_checks": label_checks, "vector_checks": vector_checks, "pass": row_count == COHORT_COUNT and distinct_count == COHORT_COUNT and row_count == distinct_count and all(item["null"] == 0 and item["invalid"] == 0 for item in label_checks.values()) and all(item["pass"] for item in vector_checks.values())}


def silhouette(df: DataFrame, feature_col: str, label_col: str) -> float:
    evaluator = ClusteringEvaluator(featuresCol=feature_col, predictionCol=label_col, metricName="silhouette", distanceMeasure="squaredEuclidean")
    return float(evaluator.evaluate(df.select(feature_col, label_col)))


def compute_sse(df: DataFrame, feature_col: str, label_col: str, k: int) -> Dict[str, Any]:
    values = vector_to_array(col(feature_col))
    dimension = SPACE_SPECS["RFM3D"]["dimension"] if feature_col == "eval_rfm3d_scaled" else SPACE_SPECS["ERFM5D"]["dimension"]
    components = df.select(label_col, *[element_at(values, index).alias(f"v{index - 1}") for index in range(1, dimension + 1)])
    names = [f"v{index}" for index in range(dimension)]
    centers_df = components.groupBy(label_col).agg(*[avg(col(name)).alias(f"c_{name}") for name in names])
    rows = centers_df.orderBy(label_col).collect()
    if len(rows) != k or [int(row[label_col]) for row in rows] != list(range(k)):
        raise ValueError(f"SSE centroid coverage invalid for {label_col}: {[int(row[label_col]) for row in rows]}")
    centers = {str(int(row[label_col])): [float(row[f"c_{name}"]) for name in names] for row in rows}
    joined = components.join(centers_df, label_col, "inner")
    squared = sum((col(name) - col(f"c_{name}")) * (col(name) - col(f"c_{name}")) for name in names)
    sse = float(joined.select(spark_sum(squared).alias("sse")).first()["sse"] or 0.0)
    return {"sse": sse, "sse_per_customer": sse / COHORT_COUNT, "centroids": centers, "dimension": dimension, "segment_count": k, "customer_count": COHORT_COUNT, "assignment_source": "locked assignments; centroids recomputed from actual 1M rows", "trainingCost_used": False}


def balance(df: DataFrame, label_col: str, k: int) -> Dict[str, Any]:
    rows = df.groupBy(label_col).count().orderBy(label_col).collect()
    counts = [int(row["count"]) for row in rows]
    if len(counts) != k or sum(counts) != COHORT_COUNT:
        raise ValueError(f"Balance coverage invalid for {label_col}.")
    ratios = [value / COHORT_COUNT for value in counts]
    mean = sum(counts) / k
    population_std = math.sqrt(sum((value - mean) ** 2 for value in counts) / k)
    entropy = -sum(value * math.log(value) for value in ratios if value > 0)
    return {"cluster_count": k, "cluster_sizes": [{"cluster_id": i, "count": counts[i], "ratio": ratios[i], "ratio_pct": ratios[i] * 100.0} for i in range(k)], "min_count": min(counts), "max_count": max(counts), "min_ratio_pct": min(ratios) * 100.0, "max_ratio_pct": max(ratios) * 100.0, "below_1pct": sum(value < 0.01 * COHORT_COUNT for value in counts), "below_2pct": sum(value < 0.02 * COHORT_COUNT for value in counts), "below_5pct": sum(value < 0.05 * COHORT_COUNT for value in counts), "mean_cluster_size": mean, "population_std_cluster_size": population_std, "cluster_size_cv": population_std / mean, "entropy_natural_log": entropy, "normalized_entropy": entropy / math.log(k)}


def partition_agreement(df: DataFrame, left: str, right: str, left_k: int, right_k: int) -> Dict[str, Any]:
    rows = df.groupBy(left, right).count().collect()
    table = [[0 for _ in range(right_k)] for _ in range(left_k)]
    for row in rows:
        table[int(row[left])][int(row[right])] = int(row["count"])
    comb2 = lambda value: value * (value - 1) / 2.0
    n = float(COHORT_COUNT)
    row_sums = [sum(row) for row in table]
    col_sums = [sum(table[i][j] for i in range(left_k)) for j in range(right_k)]
    sum_nij = sum(comb2(value) for row in table for value in row)
    sum_rows = sum(comb2(value) for value in row_sums)
    sum_cols = sum(comb2(value) for value in col_sums)
    total_pairs = comb2(n)
    expected = (sum_rows * sum_cols) / total_pairs
    maximum = 0.5 * (sum_rows + sum_cols)
    ari = (sum_nij - expected) / (maximum - expected) if maximum != expected else 1.0
    p_rows = [value / n for value in row_sums]
    p_cols = [value / n for value in col_sums]
    mi = 0.0
    for i in range(left_k):
        for j in range(right_k):
            value = table[i][j]
            if value:
                p_ij = value / n
                mi += p_ij * math.log(p_ij / (p_rows[i] * p_cols[j]))
    h_left = -sum(value * math.log(value) for value in p_rows if value > 0)
    h_right = -sum(value * math.log(value) for value in p_cols if value > 0)
    nmi = 2.0 * mi / (h_left + h_right) if h_left + h_right else 1.0
    return {"left_model": left, "right_model": right, "left_k": left_k, "right_k": right_k, "contingency_table": table, "row_sums": row_sums, "column_sums": col_sums, "adjusted_rand_index": ari, "mutual_information_natural_log": mi, "entropy_left_natural_log": h_left, "entropy_right_natural_log": h_right, "normalized_mutual_information": nmi, "nmi_definition": "2 * MI / (H_A + H_B), natural logarithm", "ari_definition": "Manual adjusted Rand index from contingency pair counts"}


def profile_segments(df: DataFrame, model_id: str, label_col: str, k: int, names: Dict[str, Any]) -> Dict[str, Any]:
    segments: Dict[str, Any] = {}
    for segment_id in range(k):
        part = df.where(col(label_col) == segment_id)
        segment_count = part.count()
        feature_profiles: Dict[str, Any] = {}
        for feature in RAW_FEATURES:
            row = part.agg(count(feature).alias("count"), avg(col(feature)).alias("mean"), spark_min(col(feature)).alias("min"), spark_max(col(feature)).alias("max")).first()
            quantiles = part.approxQuantile(feature, [0.05, 0.25, 0.50, 0.75, 0.95], QUANTILE_ERROR)
            feature_profiles[feature] = {"count": int(row["count"]), "mean": float(row["mean"]), "p05": float(quantiles[0]), "p25": float(quantiles[1]), "median": float(quantiles[2]), "p75": float(quantiles[3]), "p95": float(quantiles[4]), "max": float(row["max"]), "quantile_relative_error": QUANTILE_ERROR}
        name = names.get(str(segment_id), {})
        segments[str(segment_id)] = {"segment_id": segment_id, "segment_name_vi": name.get("segment_name_vi", name.get("name_vi", f"{model_id} Cluster {segment_id}")), "segment_name_en": name.get("segment_name_en", name.get("name_en", f"{model_id} Cluster {segment_id}")), "count": segment_count, "ratio_pct": segment_count / COHORT_COUNT * 100.0, "raw_features": feature_profiles}
    return {"model_id": model_id, "model_name": MODEL_SPECS[model_id]["name"], "feature_order": RAW_FEATURES, "segments": segments}


def raw_distinctness(profile: Dict[str, Any], global_iqr: Dict[str, float]) -> Dict[str, Any]:
    ids = sorted(profile["segments"], key=int)
    pairs = []
    for left, right in __import__("itertools").combinations(ids, 2):
        terms = []
        for feature in RAW_FEATURES:
            iqr = global_iqr[feature]
            difference = abs(profile["segments"][left]["raw_features"][feature]["median"] - profile["segments"][right]["raw_features"][feature]["median"])
            terms.append(0.0 if iqr == 0 else difference / iqr)
        pairs.append({"segment_a": int(left), "segment_b": int(right), "normalized_median_distance": math.sqrt(sum(term * term for term in terms))})
    values = [item["normalized_median_distance"] for item in pairs]
    ordered = sorted(values)
    median = ordered[len(ordered) // 2] if len(ordered) % 2 else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2.0
    return {"model_id": profile["model_id"], "global_iqr": global_iqr, "pair_count": len(pairs), "pairwise_distances": pairs, "minimum_distance": min(values), "mean_distance": sum(values) / len(values), "median_distance": median, "maximum_distance": max(values), "closest_pair": min(pairs, key=lambda item: item["normalized_median_distance"]), "farthest_pair": max(pairs, key=lambda item: item["normalized_median_distance"]), "normalization": "Raw median differences normalized by global 1M-cohort IQR per feature; Euclidean distance; zero IQR contributes zero."}


def optimal_alignment(df: DataFrame, b0_col: str, m1_col: str) -> Dict[str, Any]:
    rows = df.groupBy(b0_col, m1_col).count().collect()
    table = [[0 for _ in range(4)] for _ in range(4)]
    for row in rows:
        table[int(row[b0_col])][int(row[m1_col])] = int(row["count"])
    best_mapping = None
    best_matched = -1
    for permutation in permutations(range(4)):
        matched = sum(table[b0_id][permutation[b0_id]] for b0_id in range(4))
        if matched > best_matched:
            best_matched = matched
            best_mapping = {str(b0_id): int(permutation[b0_id]) for b0_id in range(4)}
    mismatched = COHORT_COUNT - best_matched
    agreement = partition_agreement(df, b0_col, m1_col, 4, 4)
    return {"mapping_b0_to_m1": best_mapping, "matched_customers": best_matched, "mismatched_customers": mismatched, "mismatch_ratio": mismatched / COHORT_COUNT, "ARI": agreement["adjusted_rand_index"], "NMI": agreement["normalized_mutual_information"], "contingency_table": table, "mapping_is_comparison_only": True, "saved_b0_ids_unchanged": True}


def comparison_winner(matrix: List[Dict[str, Any]], field: str, reverse: bool = True) -> Dict[str, Any]:
    row = sorted(matrix, key=lambda item: float(item[field]), reverse=reverse)[0]
    return {"model_id": row["ModelID"], "model_name": row["ModelName"], "metric": field, "value": row[field]}


def readback_artifacts(paths: List[Path], project_root: Path) -> Dict[str, bool]:
    result: Dict[str, bool] = {}
    for path in paths:
        ok = path.exists() and path.stat().st_size > 0
        if ok and path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                ok = False
        if ok and path.suffix == ".csv":
            ok = len(path.read_text(encoding="utf-8").splitlines()) >= 2
        result[str(path.relative_to(project_root))] = ok
    return result


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "output" / "comparison" / "stage22"
    report_path = project_root / "reports" / "stage22_primary_b0_m1_m2_quality_comparison.txt"
    summary_path = output_dir / "stage22_final_summary.json"
    blockers: List[str] = []
    warnings: List[str] = ["Stage 22 is quality-only; no runtime comparison, retraining, K sweep, rule-based model, or GMM was executed.", "Cluster IDs are arbitrary; B0/M1 raw mismatch uses optimal label alignment."]
    spark = None
    evaluation = None
    try:
        blockers, authorities = audit_authorities(project_root)
        if output_dir.exists() and any(output_dir.iterdir()):
            blockers.append("Existing non-empty Stage 22 output detected; replacement is not authorized by this run.")
        if blockers:
            failed = {"stage": 22, "authority": "PRIMARY_B0_M1_M2_QUALITY_COMPARISON_V1", "primary_quality_comparison_complete": False, "ready_for_stage23": False, "blockers": blockers}
            write_json(summary_path, failed)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("STAGE 22 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_23: False\n", encoding="utf-8")
            return 3
        output_dir.mkdir(parents=True, exist_ok=True)
        cohort_path = project_root / "output" / "comparison" / "stage17" / "comparison_cohort_1m"
        b0_path = project_root / "output" / "comparison" / "stage21" / "basic_kmeans_predictions_1m"
        stage11_names = read_json(project_root / "output" / "analysis" / "stage11" / "segment_names.json").get("segments", {})
        stage16_names = read_json(project_root / "output" / "analysis" / "stage16" / "segment_names.json").get("segments", {})
        b0_names = {str(index): {"segment_name_vi": f"B0 Cụm {index}", "segment_name_en": f"B0 Cluster {index}"} for index in range(4)}
        names_by_model = {"B0": b0_names, "M1": stage11_names, "M2": stage16_names}
        spark = SparkSession.builder.master("local[4]").appName("Stage22-Primary-B0-M1-M2-Quality").config("spark.sql.shuffle.partitions", "64").config("spark.sql.adaptive.enabled", "true").config("spark.local.dir", str(project_root.parent / "spark_tmp")).getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        enable_windows_java_filesystem_fallback(spark)
        cohort = spark.read.parquet(str(cohort_path))
        b0 = spark.read.parquet(str(b0_path)).select("customer_id", col("b0_prediction").cast("int").alias("b0_prediction"))
        required = ["customer_id", *RAW_FEATURES, "eval_rfm3d_scaled", "eval_erfm5d_scaled", "main_prediction", "enhanced_prediction"]
        missing = [column for column in required if column not in cohort.columns]
        if missing:
            raise ValueError(f"Stage 17 cohort missing required columns: {missing}")
        evaluation = cohort.select(*required).join(b0, "customer_id", "inner").persist(StorageLevel.MEMORY_AND_DISK)
        evaluation.count()
        validation = dataset_validation(evaluation)
        write_json(output_dir / "comparison_dataset_validation.json", validation)
        if not validation["pass"]:
            raise ValueError("Stage 22 evaluation dataset validation failed.")

        silhouettes = {model_id: {space_id: {"model_id": model_id, "model_name": spec["name"], "space": space["name"], "feature_column": space["feature"], "silhouette": silhouette(evaluation, space["feature"], spec["label"]), "metric_name": "silhouette", "distance_measure": "squaredEuclidean", "customer_count": COHORT_COUNT} for space_id, space in SPACE_SPECS.items()} for model_id, spec in MODEL_SPECS.items()}
        sses = {model_id: {space_id: compute_sse(evaluation, space["feature"], spec["label"], spec["k"]) for space_id, space in SPACE_SPECS.items()} for model_id, spec in MODEL_SPECS.items()}
        balances = {model_id: balance(evaluation, spec["label"], spec["k"]) for model_id, spec in MODEL_SPECS.items()}
        historical_silhouette_differences = {"M1_RFM3D": abs(silhouettes["M1"]["RFM3D"]["silhouette"] - HISTORICAL["M1_RFM3D"]), "M1_ERFM5D": abs(silhouettes["M1"]["ERFM5D"]["silhouette"] - HISTORICAL["M1_ERFM5D"]), "M2_RFM3D": abs(silhouettes["M2"]["RFM3D"]["silhouette"] - HISTORICAL["M2_RFM3D"]), "M2_ERFM5D": abs(silhouettes["M2"]["ERFM5D"]["silhouette"] - HISTORICAL["M2_ERFM5D"])}
        if max(historical_silhouette_differences.values()) > TOLERANCE:
            raise ValueError(f"Stage 19 historical M1/M2 silhouette cross-check mismatch: {historical_silhouette_differences}")
        write_json(output_dir / "common_space_silhouette.json", silhouettes)
        write_json(output_dir / "common_space_sse.json", sses)
        write_json(output_dir / "cluster_balance_metrics.json", balances)

        agreements = {"B0_vs_M1": partition_agreement(evaluation, "b0_prediction", "main_prediction", 4, 4), "B0_vs_M2": partition_agreement(evaluation, "b0_prediction", "enhanced_prediction", 4, 5), "M1_vs_M2": partition_agreement(evaluation, "main_prediction", "enhanced_prediction", 4, 5)}
        write_json(output_dir / "partition_agreement.json", agreements)
        write_json(output_dir / "contingency_tables.json", {key: {"left_model": value["left_model"], "right_model": value["right_model"], "table": value["contingency_table"], "row_sums": value["row_sums"], "column_sums": value["column_sums"]} for key, value in agreements.items()})
        alignment = optimal_alignment(evaluation, "b0_prediction", "main_prediction")
        write_json(output_dir / "b0_m1_label_alignment.json", alignment)

        profiles = {model_id: profile_segments(evaluation, model_id, spec["label"], spec["k"], names_by_model[model_id]) for model_id, spec in MODEL_SPECS.items()}
        global_iqr = {feature: float(evaluation.approxQuantile(feature, [0.75], QUANTILE_ERROR)[0] - evaluation.approxQuantile(feature, [0.25], QUANTILE_ERROR)[0]) for feature in RAW_FEATURES}
        write_json(output_dir / "raw_profiles_1m.json", {"cohort_count": COHORT_COUNT, "global_iqr": global_iqr, "models": profiles})
        profile_rows = []
        for model_id, payload in profiles.items():
            for segment_id, segment in payload["segments"].items():
                for feature, stats_payload in segment["raw_features"].items():
                    profile_rows.append({"ModelID": model_id, "ModelName": payload["model_name"], "SegmentId": int(segment_id), "SegmentNameVI": segment["segment_name_vi"], "SegmentNameEN": segment["segment_name_en"], "CustomerCount": segment["count"], "RatioPct": segment["ratio_pct"], "Feature": feature, **{key: value for key, value in stats_payload.items() if key != "quantile_relative_error"}})
        write_csv(output_dir / "raw_profiles_1m.csv", ["ModelID", "ModelName", "SegmentId", "SegmentNameVI", "SegmentNameEN", "CustomerCount", "RatioPct", "Feature", "count", "mean", "p05", "p25", "median", "p75", "p95", "max"], profile_rows)
        distinctness = {model_id: raw_distinctness(profile, global_iqr) for model_id, profile in profiles.items()}
        write_json(output_dir / "raw_profile_distinctness.json", distinctness)

        matrix = []
        for model_id, spec in MODEL_SPECS.items():
            matrix.append({"ModelID": model_id, "ModelName": spec["name"], "TrainingFeatureSpace": spec["training_space"], "K": spec["k"], "InitMode": spec["init"], "Silhouette_RFM3D": silhouettes[model_id]["RFM3D"]["silhouette"], "Silhouette_ERFM5D": silhouettes[model_id]["ERFM5D"]["silhouette"], "SSEPerCustomer_RFM3D": sses[model_id]["RFM3D"]["sse_per_customer"], "SSEPerCustomer_ERFM5D": sses[model_id]["ERFM5D"]["sse_per_customer"], "MinClusterRatioPct": balances[model_id]["min_ratio_pct"], "MaxClusterRatioPct": balances[model_id]["max_ratio_pct"], "ClusterSizeCV": balances[model_id]["cluster_size_cv"], "NormalizedEntropy": balances[model_id]["normalized_entropy"], "RawDistinctnessMin": distinctness[model_id]["minimum_distance"], "RawDistinctnessMean": distinctness[model_id]["mean_distance"], "ModelRole": spec["role"]})
        matrix_fields = ["ModelID", "ModelName", "TrainingFeatureSpace", "K", "InitMode", "Silhouette_RFM3D", "Silhouette_ERFM5D", "SSEPerCustomer_RFM3D", "SSEPerCustomer_ERFM5D", "MinClusterRatioPct", "MaxClusterRatioPct", "ClusterSizeCV", "NormalizedEntropy", "RawDistinctnessMin", "RawDistinctnessMean", "ModelRole"]
        write_csv(output_dir / "primary_three_model_comparison_matrix.csv", matrix_fields, matrix)

        b0_m1 = {"full_data_context": {"b0_full_silhouette": authorities["stage21"].get("b0_metrics", {}).get("silhouette_rfm3d"), "m1_full_silhouette": authorities["stage10"].get("training", {}).get("silhouette"), "b0_full_wcss": authorities["stage21"].get("b0_metrics", {}).get("trainingCost_WCSS"), "m1_full_wcss": authorities["stage10"].get("training", {}).get("wcss"), "b0_numIter": authorities["stage21"].get("b0_metrics", {}).get("numIter"), "m1_numIter": authorities["stage10"].get("training", {}).get("num_iter")}, "locked_1m": {"b0_silhouette_rfm3d": silhouettes["B0"]["RFM3D"]["silhouette"], "m1_silhouette_rfm3d": silhouettes["M1"]["RFM3D"]["silhouette"], "b0_sse_per_customer_rfm3d": sses["B0"]["RFM3D"]["sse_per_customer"], "m1_sse_per_customer_rfm3d": sses["M1"]["RFM3D"]["sse_per_customer"], "b0_min_ratio_pct": balances["B0"]["min_ratio_pct"], "m1_min_ratio_pct": balances["M1"]["min_ratio_pct"], "b0_max_ratio_pct": balances["B0"]["max_ratio_pct"], "m1_max_ratio_pct": balances["M1"]["max_ratio_pct"], "b0_entropy": balances["B0"]["normalized_entropy"], "m1_entropy": balances["M1"]["normalized_entropy"], "b0_cv": balances["B0"]["cluster_size_cv"], "m1_cv": balances["M1"]["cluster_size_cv"]}, "silhouette_delta_b0_minus_m1": silhouettes["B0"]["RFM3D"]["silhouette"] - silhouettes["M1"]["RFM3D"]["silhouette"], "sse_per_customer_delta_b0_minus_m1": sses["B0"]["RFM3D"]["sse_per_customer"] - sses["M1"]["RFM3D"]["sse_per_customer"], "balance_delta_b0_minus_m1": {"min_ratio_pp": balances["B0"]["min_ratio_pct"] - balances["M1"]["min_ratio_pct"], "max_ratio_pp": balances["B0"]["max_ratio_pct"] - balances["M1"]["max_ratio_pct"], "cv": balances["B0"]["cluster_size_cv"] - balances["M1"]["cluster_size_cv"], "normalized_entropy": balances["B0"]["normalized_entropy"] - balances["M1"]["normalized_entropy"]}, "ARI": agreements["B0_vs_M1"]["adjusted_rand_index"], "NMI": agreements["B0_vs_M1"]["normalized_mutual_information"], "cluster_membership_mismatch_count_after_optimal_alignment": alignment["mismatched_customers"], "cluster_membership_mismatch_ratio_after_optimal_alignment": alignment["mismatch_ratio"], "initialization_difference_only": True}
        if abs(b0_m1["locked_1m"]["b0_silhouette_rfm3d"] - HISTORICAL["B0_RFM3D"]) > TOLERANCE or abs(b0_m1["locked_1m"]["b0_sse_per_customer_rfm3d"] - HISTORICAL["B0_SSE_PER_CUSTOMER"]) > TOLERANCE:
            raise ValueError("B0 historical Stage 21 cross-check mismatch.")
        m1_m2 = {"ERFM5D": {"m1_silhouette": silhouettes["M1"]["ERFM5D"]["silhouette"], "m2_silhouette": silhouettes["M2"]["ERFM5D"]["silhouette"], "silhouette_absolute_improvement": silhouettes["M2"]["ERFM5D"]["silhouette"] - silhouettes["M1"]["ERFM5D"]["silhouette"], "silhouette_relative_improvement_pct": (silhouettes["M2"]["ERFM5D"]["silhouette"] - silhouettes["M1"]["ERFM5D"]["silhouette"]) / abs(silhouettes["M1"]["ERFM5D"]["silhouette"]) * 100.0, "m1_sse_per_customer": sses["M1"]["ERFM5D"]["sse_per_customer"], "m2_sse_per_customer": sses["M2"]["ERFM5D"]["sse_per_customer"], "sse_per_customer_absolute_change": sses["M2"]["ERFM5D"]["sse_per_customer"] - sses["M1"]["ERFM5D"]["sse_per_customer"], "sse_per_customer_reduction_pct": (sses["M1"]["ERFM5D"]["sse_per_customer"] - sses["M2"]["ERFM5D"]["sse_per_customer"]) / sses["M1"]["ERFM5D"]["sse_per_customer"] * 100.0, "entropy_change": balances["M2"]["normalized_entropy"] - balances["M1"]["normalized_entropy"], "cv_reduction_pct": (balances["M1"]["cluster_size_cv"] - balances["M2"]["cluster_size_cv"]) / balances["M1"]["cluster_size_cv"] * 100.0, "min_cluster_ratio_change_pp": balances["M2"]["min_ratio_pct"] - balances["M1"]["min_ratio_pct"], "max_cluster_ratio_change_pp": balances["M2"]["max_ratio_pct"] - balances["M1"]["max_ratio_pct"], "below_5pct_change": balances["M2"]["below_5pct"] - balances["M1"]["below_5pct"], "raw_distinctness_min_change": distinctness["M2"]["minimum_distance"] - distinctness["M1"]["minimum_distance"], "raw_distinctness_mean_change": distinctness["M2"]["mean_distance"] - distinctness["M1"]["mean_distance"], "m1_training_dimensions": 3, "m2_training_dimensions": 5}, "RFM3D_tradeoff": {"m1_silhouette": silhouettes["M1"]["RFM3D"]["silhouette"], "m2_silhouette": silhouettes["M2"]["RFM3D"]["silhouette"], "silhouette_change": silhouettes["M2"]["RFM3D"]["silhouette"] - silhouettes["M1"]["RFM3D"]["silhouette"], "m1_sse_per_customer": sses["M1"]["RFM3D"]["sse_per_customer"], "m2_sse_per_customer": sses["M2"]["RFM3D"]["sse_per_customer"], "sse_per_customer_change": sses["M2"]["RFM3D"]["sse_per_customer"] - sses["M1"]["RFM3D"]["sse_per_customer"]}}
        raw_dimensional_evidence = {feature: {"global_iqr": global_iqr[feature], "m2_segment_median_range": max(segment["raw_features"][feature]["median"] for segment in profiles["M2"]["segments"].values()) - min(segment["raw_features"][feature]["median"] for segment in profiles["M2"]["segments"].values())} for feature in ["AvgBasketSize", "ProductVariety"]}
        m1_m2["behavioral_feature_evidence"] = raw_dimensional_evidence
        m1_m2["enhancement_supported"] = all([m1_m2["ERFM5D"]["m2_silhouette"] > m1_m2["ERFM5D"]["m1_silhouette"], m1_m2["ERFM5D"]["m2_sse_per_customer"] < m1_m2["ERFM5D"]["m1_sse_per_customer"], balances["M2"]["normalized_entropy"] > balances["M1"]["normalized_entropy"], balances["M2"]["cluster_size_cv"] < balances["M1"]["cluster_size_cv"], balances["M2"]["min_ratio_pct"] > balances["M1"]["min_ratio_pct"], raw_dimensional_evidence["AvgBasketSize"]["m2_segment_median_range"] > 0, raw_dimensional_evidence["ProductVariety"]["m2_segment_median_range"] > 0])
        write_json(output_dir / "b0_to_m1_analysis.json", b0_m1)
        write_json(output_dir / "m1_to_m2_enhancement_analysis.json", m1_m2)

        findings = {"BEST_RFM3D_SILHOUETTE": comparison_winner(matrix, "Silhouette_RFM3D"), "BEST_ERFM5D_SILHOUETTE": comparison_winner(matrix, "Silhouette_ERFM5D"), "LOWEST_RFM3D_SSE_PER_CUSTOMER": comparison_winner(matrix, "SSEPerCustomer_RFM3D", False), "LOWEST_ERFM5D_SSE_PER_CUSTOMER": comparison_winner(matrix, "SSEPerCustomer_ERFM5D", False), "BEST_BALANCE_ENTROPY": comparison_winner(matrix, "NormalizedEntropy"), "LOWEST_CLUSTER_SIZE_CV": comparison_winner(matrix, "ClusterSizeCV", False), "BEST_RAW_DISTINCTNESS_MIN": comparison_winner(matrix, "RawDistinctnessMin"), "BEST_RAW_DISTINCTNESS_MEAN": comparison_winner(matrix, "RawDistinctnessMean"), "weighted_score_used": False}
        b0_m1_finding = "QUALITY_SIMILAR_CONVERGENCE_IMPROVED" if abs(b0_m1["silhouette_delta_b0_minus_m1"]) <= 1e-3 and abs(b0_m1["sse_per_customer_delta_b0_minus_m1"]) <= 1e-3 and b0_m1["full_data_context"]["m1_numIter"] < b0_m1["full_data_context"]["b0_numIter"] else "QUALITY_SIMILAR"
        findings["B0_TO_M1_FINDING"] = b0_m1_finding
        findings["M1_TO_M2_FINDING"] = "M2_IMPROVES_ENHANCED_MULTIDIMENSIONAL_BEHAVIORAL_SEGMENTATION" if m1_m2["enhancement_supported"] else "M2_ENHANCEMENT_NOT_FULLY_SUPPORTED_BY_ALL_REQUIRED_DIRECTIONS"
        write_json(output_dir / "criterion_findings.json", findings)

        # Independent reproducibility pass: recompute every formal metric and
        # both analyses from the same materialized evaluation dataset.
        repro_silhouettes = {model_id: {space_id: silhouette(evaluation, space["feature"], spec["label"]) for space_id, space in SPACE_SPECS.items()} for model_id, spec in MODEL_SPECS.items()}
        repro_sses = {model_id: {space_id: compute_sse(evaluation, space["feature"], spec["label"], spec["k"])["sse"] for space_id, space in SPACE_SPECS.items()} for model_id, spec in MODEL_SPECS.items()}
        repro_balances = {model_id: balance(evaluation, spec["label"], spec["k"])["normalized_entropy"] for model_id, spec in MODEL_SPECS.items()}
        repro_agreements = {key: partition_agreement(evaluation, value["left_model"], value["right_model"], value["left_k"], value["right_k"]) for key, value in agreements.items()}
        repro_alignment = optimal_alignment(evaluation, "b0_prediction", "main_prediction")
        differences = {"silhouette": {model_id: {space_id: abs(silhouettes[model_id][space_id]["silhouette"] - repro_silhouettes[model_id][space_id]) for space_id in SPACE_SPECS} for model_id in MODEL_SPECS}, "sse": {model_id: {space_id: abs(sses[model_id][space_id]["sse"] - repro_sses[model_id][space_id]) for space_id in SPACE_SPECS} for model_id in MODEL_SPECS}, "normalized_entropy": {model_id: abs(balances[model_id]["normalized_entropy"] - repro_balances[model_id]) for model_id in MODEL_SPECS}, "ari": {key: abs(agreements[key]["adjusted_rand_index"] - repro_agreements[key]["adjusted_rand_index"]) for key in agreements}, "nmi": {key: abs(agreements[key]["normalized_mutual_information"] - repro_agreements[key]["normalized_mutual_information"]) for key in agreements}, "alignment_mismatch": abs(alignment["mismatched_customers"] - repro_alignment["mismatched_customers"]), "alignment_mapping_equal": alignment["mapping_b0_to_m1"] == repro_alignment["mapping_b0_to_m1"], "m1_m2_silhouette_delta": abs(m1_m2["ERFM5D"]["silhouette_absolute_improvement"] - (repro_silhouettes["M2"]["ERFM5D"] - repro_silhouettes["M1"]["ERFM5D"]))}
        scalar_differences = [difference for group in [differences["silhouette"], differences["sse"], differences["normalized_entropy"]] for nested in group.values() for difference in (nested.values() if isinstance(nested, dict) else [nested])] + list(differences["ari"].values()) + list(differences["nmi"].values()) + [differences["alignment_mismatch"], differences["m1_m2_silhouette_delta"]]
        repro = {"tolerance": TOLERANCE, "absolute_differences": differences, "independent_metric_counts": {"silhouette": 6, "sse": 6, "normalized_entropy": 3, "ari": 3, "nmi": 3, "b0_m1_alignment": 1, "m1_m2_deltas": 1}, "max_scalar_difference": max(scalar_differences or [0.0]), "pass": differences["alignment_mapping_equal"] and max(scalar_differences or [0.0]) <= TOLERANCE}
        write_json(output_dir / "reproducibility_check.json", repro)
        if not repro["pass"]:
            raise ValueError("Stage 22 independent reproducibility check failed.")

        summary = {"stage": 22, "authority": "PRIMARY_B0_M1_M2_QUALITY_COMPARISON_V1", "cohort_authority": "DETERMINISTIC_1M_COMPARISON_COHORT", "cohort_count": COHORT_COUNT, "cohort_customer_id_sha256": COHORT_FINGERPRINT, "models": {model_id: {"model_name": spec["name"], "k": spec["k"], "training_feature_space": spec["training_space"], "initMode": spec["init"], "role": spec["role"]} for model_id, spec in MODEL_SPECS.items()}, "b0_to_m1_analysis": b0_m1, "m1_to_m2_enhancement_analysis": m1_m2, "m2_enhancement_supported": m1_m2["enhancement_supported"], "criterion_findings": findings, "models_retrained": False, "classification_accuracy_used": False, "weighted_winner_score_used": False, "runtime_quality_combined_score_used": False, "rule_based_primary_model_used": False, "gmm_executed": False, "stage19_auxiliary_preserved": True, "primary_quality_comparison_complete": True, "ready_for_stage23": True, "warnings": warnings, "blockers": []}
        write_json(summary_path, summary)
        artifact_paths = [output_dir / name for name in ["comparison_dataset_validation.json", "common_space_silhouette.json", "common_space_sse.json", "cluster_balance_metrics.json", "partition_agreement.json", "contingency_tables.json", "b0_m1_label_alignment.json", "raw_profiles_1m.json", "raw_profiles_1m.csv", "raw_profile_distinctness.json", "b0_to_m1_analysis.json", "m1_to_m2_enhancement_analysis.json", "criterion_findings.json", "primary_three_model_comparison_matrix.csv", "reproducibility_check.json", "stage22_final_summary.json"]] + [report_path, project_root / "reports" / "stage22_run.log"]
        report_lines = ["STAGE 22 — PRIMARY FORMAL B0/M1/M2 QUALITY COMPARISON", "", f"AUTHORITY: {summary['authority']}", f"COHORT: {COHORT_COUNT} / {COHORT_FINGERPRINT}", f"B0_TO_M1_FINDING: {b0_m1_finding}", f"M1_TO_M2_FINDING: {findings['M1_TO_M2_FINDING']}", f"M2_ENHANCEMENT_SUPPORTED: {m1_m2['enhancement_supported']}", "", "COMPARISON MATRIX:", *[str(row) for row in matrix], "", "B0/M1 ALIGNMENT:", str(alignment), "", "REPRODUCIBILITY:", str(repro), "", "NO RUNTIME COMPARISON; NO RETRAINING; NO RULE-BASED PRIMARY MODEL; NO GMM.", "", "PRIMARY_QUALITY_COMPARISON_COMPLETE: True", "READY_FOR_STAGE_23: True"]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        readback = readback_artifacts(artifact_paths, project_root)
        summary["artifact_readback"] = readback
        summary["artifact_readback_pass"] = all(readback.values())
        summary["ready_for_stage23"] = bool(summary["artifact_readback_pass"])
        if not summary["ready_for_stage23"]:
            summary["primary_quality_comparison_complete"] = False
            summary["blockers"] = ["Stage 22 artifact read-back failed."]
        write_json(summary_path, summary)
        print("AUTHORITY_AUDIT_PASS=True")
        print("EVALUATION_DATASET_VALIDATION_PASS=True")
        print("HISTORICAL_CROSSCHECK_PASS=True")
        print("REPRODUCIBILITY_PASS=True")
        print(f"M2_ENHANCEMENT_SUPPORTED={m1_m2['enhancement_supported']}")
        print(f"PRIMARY_QUALITY_COMPARISON_COMPLETE={summary['primary_quality_comparison_complete']}")
        print(f"READY_FOR_STAGE_23={summary['ready_for_stage23']}")
        return 0 if summary["ready_for_stage23"] else 3
    except Exception as error:
        blockers.append(str(error))
        failed = {"stage": 22, "authority": "PRIMARY_B0_M1_M2_QUALITY_COMPARISON_V1", "primary_quality_comparison_complete": False, "ready_for_stage23": False, "warnings": warnings, "blockers": blockers}
        write_json(summary_path, failed)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("STAGE 22 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_23: False\n", encoding="utf-8")
        print(f"STAGE 22 FAILED CLOSED: {error}")
        return 3
    finally:
        if evaluation is not None:
            try:
                evaluation.unpersist(blocking=False)
            except Exception:
                pass
        if spark is not None:
            try:
                spark.stop()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
