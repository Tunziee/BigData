"""Stage 20: controlled runtime benchmark on the locked 1M cohort.

The benchmark has two independent tracks:
  A. assignment/inference using locked authorities;
  B. comparison-only build/fit plus assignment.

No Stage 17-19 authority or final model is modified by this script.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, element_at, lit, when
from pyspark.storagelevel import StorageLevel


EXPECTED_COUNT = 1_000_000
COHORT_FINGERPRINT = "ca082ec1d828df9c34a10105ae11599c3cc0653dc5b4d2c11af18ec82b781cf5"
COHORT_AUTHORITY = "DETERMINISTIC_1M_COMPARISON_COHORT"
RULE_AUTHORITY = "RULE_BASED_RFM_BASELINE_V1"
RULE_METHOD_AUTHORITY = "RULE_BASED_RFM_QUINTILE_V1"
QUALITY_AUTHORITY = "THREE_MODEL_QUALITY_COMPARISON_V1"
RFM_FEATURES = ["Recency", "Frequency", "Monetary"]
REQUIRED_COLUMNS = ["customer_id", *RFM_FEATURES, "eval_rfm3d_scaled", "eval_erfm5d_scaled", "main_prediction", "enhanced_prediction"]
R1_COUNTS = [267989, 159627, 199853, 142542, 229989]
K1_COUNTS = [535490, 145123, 301103, 18284]
K2_COUNTS = [154437, 126550, 197817, 252287, 268909]
EXPECTED_RULE_THRESHOLDS = {
    "Recency": [35.0, 96.0, 232.0, 460.0],
    "Frequency": [1.0, 2.0, 4.0, 10.0],
    "Monetary": [0.06859322033898305, 0.1642881355932203, 0.36584745762711857, 0.8910847457627119],
}
QUANTILE_PROBABILITIES = [0.2, 0.4, 0.6, 0.8]
EXPECTED_MODEL_CONFIG = {"seed": 42, "distanceMeasure": "euclidean", "initMode": "k-means||", "maxIter": 100, "tol": 1e-4}
TOLERANCE = 1e-8


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
    stage18 = read_json(project_root / "output" / "comparison" / "stage18" / "stage18_final_summary.json")
    stage19 = read_json(project_root / "output" / "comparison" / "stage19" / "stage19_final_summary.json")
    stage17_ok = (
        stage17.get("stage") == 17
        and stage17.get("cohort_authority") == COHORT_AUTHORITY
        and stage17.get("cohort_count") == EXPECTED_COUNT
        and stage17.get("cohort_customer_id_sha256") == COHORT_FINGERPRINT
        and stage17.get("source_alignment_pass") is True
        and stage17.get("deterministic_rerun_validation", {}).get("pass") is True
        and stage17.get("cohort_readback", {}).get("pass") is True
        and stage17.get("ready_for_stage18") is True
    )
    stage18_thresholds = stage18.get("thresholds", {})
    thresholds_ok = all(
        len(stage18_thresholds.get(feature, [])) == 4
        and all(abs(float(a) - float(b)) <= 1e-12 for a, b in zip(stage18_thresholds[feature], EXPECTED_RULE_THRESHOLDS[feature]))
        for feature in RFM_FEATURES
    )
    stage18_ok = (
        stage18.get("stage") == 18
        and stage18.get("authority") == RULE_AUTHORITY
        and stage18.get("method_authority") == RULE_METHOD_AUTHORITY
        and stage18.get("cohort_authority") == COHORT_AUTHORITY
        and stage18.get("cohort_count") == EXPECTED_COUNT
        and stage18.get("cohort_customer_id_sha256") == COHORT_FINGERPRINT
        and stage18.get("ready_for_stage19") is True
        and thresholds_ok
    )
    stage19_ok = (
        stage19.get("stage") == 19
        and stage19.get("authority") == QUALITY_AUTHORITY
        and stage19.get("cohort_count") == EXPECTED_COUNT
        and stage19.get("cohort_customer_id_sha256") == COHORT_FINGERPRINT
        and stage19.get("formal_quality_comparison_complete") is True
        and stage19.get("ready_for_stage20") is True
    )
    if not stage17_ok:
        blockers.append("Stage 17 authority, fingerprint, or locked gate mismatch.")
    if not stage18_ok:
        blockers.append("Stage 18 authority, fingerprint, threshold, or locked gate mismatch.")
    if not stage19_ok:
        blockers.append("Stage 19 authority, fingerprint, or locked gate mismatch.")
    return blockers, {"stage17": stage17, "stage18": stage18, "stage19": stage19, "stage18_thresholds": stage18_thresholds, "thresholds_match_expected": thresholds_ok}


def validate_vector(df: DataFrame, column: str, dimension: int) -> Dict[str, Any]:
    values = vector_to_array(col(column))
    from pyspark.sql.functions import size
    null_count = df.where(col(column).isNull()).count()
    wrong_dimension = df.where(col(column).isNotNull() & (size(values) != dimension)).count()
    nan_count = sum(df.where(element_at(values, index) != element_at(values, index)).count() for index in range(1, dimension + 1))
    infinity_count = sum(df.where((element_at(values, index) == float("inf")) | (element_at(values, index) == float("-inf"))).count() for index in range(1, dimension + 1))
    return {"column": column, "dimension": dimension, "null": null_count, "wrong_dimension": wrong_dimension, "nan": nan_count, "infinity": infinity_count, "pass": null_count == 0 and wrong_dimension == 0 and nan_count == 0 and infinity_count == 0}


def input_validation(df: DataFrame) -> Dict[str, Any]:
    row_count = df.count()
    distinct_count = df.select("customer_id").distinct().count()
    nulls = {feature: df.where(col(feature).isNull()).count() for feature in RFM_FEATURES}
    nan_counts = {feature: df.where(col(feature) != col(feature)).count() for feature in RFM_FEATURES}
    return {"row_count": row_count, "distinct_customer_id": distinct_count, "duplicate_customer_id": row_count - distinct_count, "rfm_nulls": nulls, "rfm_nan": nan_counts, "eval_rfm3d_scaled": validate_vector(df, "eval_rfm3d_scaled", 3), "eval_erfm5d_scaled": validate_vector(df, "eval_erfm5d_scaled", 5), "pass": row_count == EXPECTED_COUNT and distinct_count == EXPECTED_COUNT and row_count == distinct_count and all(value == 0 for value in nulls.values()) and all(value == 0 for value in nan_counts.values())}


def thresholds_from_data(df: DataFrame) -> Dict[str, List[float]]:
    return {feature: [float(value) for value in df.approxQuantile(feature, QUANTILE_PROBABILITIES, 0.0)] for feature in RFM_FEATURES}


def rule_assign(df: DataFrame, thresholds: Dict[str, List[float]]) -> DataFrame:
    r20, r40, r60, r80 = thresholds["Recency"]
    f20, f40, f60, f80 = thresholds["Frequency"]
    m20, m40, m60, m80 = thresholds["Monetary"]
    r_score = when(col("Recency") <= lit(r20), 5).when(col("Recency") <= lit(r40), 4).when(col("Recency") <= lit(r60), 3).when(col("Recency") <= lit(r80), 2).otherwise(1)
    f_score = when(col("Frequency") <= lit(f20), 1).when(col("Frequency") <= lit(f40), 2).when(col("Frequency") <= lit(f60), 3).when(col("Frequency") <= lit(f80), 4).otherwise(5)
    m_score = when(col("Monetary") <= lit(m20), 1).when(col("Monetary") <= lit(m40), 2).when(col("Monetary") <= lit(m60), 3).when(col("Monetary") <= lit(m80), 4).otherwise(5)
    scored = df.withColumn("r_score", r_score).withColumn("f_score", f_score).withColumn("m_score", m_score)
    scored = scored.withColumn("rfm_total_score", col("r_score") + col("f_score") + col("m_score"))
    return scored.withColumn("rule_segment_id", when(col("rfm_total_score") <= 5, 0).when(col("rfm_total_score") <= 7, 1).when(col("rfm_total_score") <= 10, 2).when(col("rfm_total_score") <= 12, 3).otherwise(4))


def distribution(df: DataFrame, label_col: str, expected: List[int]) -> Dict[str, Any]:
    rows = df.groupBy(label_col).count().orderBy(label_col).collect()
    actual = [int(row["count"]) for row in rows]
    return {"counts": actual, "cluster_ids": [int(row[label_col]) for row in rows], "expected_counts": expected, "mismatch_total": sum(abs(a - b) for a, b in zip(actual, expected)) if len(actual) == len(expected) else EXPECTED_COUNT, "pass": actual == expected and [int(row[label_col]) for row in rows] == list(range(len(expected)))}


def compare_prediction_rows(df: DataFrame, prediction_df: DataFrame, expected_col: str) -> int:
    actual = prediction_df.select("customer_id", col("prediction").cast("long").alias("actual"))
    expected = df.select("customer_id", col(expected_col).cast("long").alias("expected"))
    return actual.join(expected, "customer_id", "inner").where(col("actual") != col("expected")).count()


def stats(values: List[float]) -> Dict[str, Any]:
    return {"min_sec": min(values), "max_sec": max(values), "mean_sec": statistics.mean(values), "median_sec": statistics.median(values), "std_sec": statistics.pstdev(values), "trial_times_sec": values}


def model_centers(model: Any) -> List[List[float]]:
    return [[float(value) for value in center] for center in model.clusterCenters()]


def run_assignment_trial(model_id: str, df: DataFrame, model: Any, thresholds: Dict[str, List[float]], expected_counts: List[int], expected_prediction_col: str | None = None) -> Dict[str, Any]:
    start = time.perf_counter()
    if model_id == "R1":
        result = rule_assign(df, thresholds)
        counts = distribution(result, "rule_segment_id", expected_counts)
    else:
        features_col = "features" if model_id == "K1" else "features_improved"
        source_col = "eval_rfm3d_scaled" if model_id == "K1" else "eval_erfm5d_scaled"
        result = model.transform(df.withColumn(features_col, col(source_col)))
        counts = distribution(result, "prediction", expected_counts)
    elapsed = time.perf_counter() - start
    return {"model_id": model_id, "duration_sec": elapsed, "distribution": counts, "timed_action": "groupBy(label).count().collect()", "pass": counts["pass"] and elapsed > 0}


def run_build_rule_trial(df: DataFrame, expected_counts: List[int], expected_thresholds: Dict[str, List[float]]) -> Dict[str, Any]:
    start = time.perf_counter()
    thresholds = thresholds_from_data(df)
    result = rule_assign(df, thresholds)
    counts = distribution(result, "rule_segment_id", expected_counts)
    elapsed = time.perf_counter() - start
    threshold_diff = {feature: [abs(a - b) for a, b in zip(thresholds[feature], expected_thresholds[feature])] for feature in RFM_FEATURES}
    threshold_pass = all(all(value <= 1e-12 for value in values) for values in threshold_diff.values())
    return {"model_id": "R1", "duration_sec": elapsed, "thresholds": thresholds, "threshold_absolute_differences": threshold_diff, "threshold_pass": threshold_pass, "distribution": counts, "timed_action": "approxQuantile(relativeError=0.0) + rules + groupBy(rule_segment_id).count().collect()", "pass": elapsed > 0 and threshold_pass and counts["pass"]}


def run_build_kmeans_trial(model_id: str, df: DataFrame, k: int, source_col: str) -> Dict[str, Any]:
    benchmark_col = "benchmark_features"
    prediction_col = "benchmark_prediction"
    benchmark_df = df.withColumn(benchmark_col, col(source_col))
    estimator = KMeans(k=k, seed=42, featuresCol=benchmark_col, predictionCol=prediction_col, distanceMeasure="euclidean", initMode="k-means||", maxIter=100, tol=1e-4)
    start = time.perf_counter()
    fit_start = time.perf_counter()
    model = estimator.fit(benchmark_df)
    fit_elapsed = time.perf_counter() - fit_start
    assigned = model.transform(benchmark_df)
    rows = assigned.groupBy(prediction_col).count().orderBy(prediction_col).collect()
    total_elapsed = time.perf_counter() - start
    counts = [int(row["count"]) for row in rows]
    assign_elapsed = total_elapsed - fit_elapsed
    summary = model.summary
    return {"model_id": model_id, "duration_sec": total_elapsed, "fit_only_sec": fit_elapsed, "assign_after_fit_sec": assign_elapsed, "total_fit_plus_assign_sec": total_elapsed, "numIter": int(summary.numIter), "trainingCost": float(summary.trainingCost), "cluster_sizes": counts, "cluster_ids": [int(row[prediction_col]) for row in rows], "centers": model_centers(model), "pass": total_elapsed > 0 and len(counts) == k and sum(counts) == EXPECTED_COUNT}


def deterministic_validation(trials: Dict[str, List[Dict[str, Any]]], model_id: str) -> Dict[str, Any]:
    records = trials[model_id]
    reference = records[0]
    num_iter_equal = all(record["numIter"] == reference["numIter"] for record in records)
    cost_differences = [abs(record["trainingCost"] - reference["trainingCost"]) for record in records]
    cluster_equal = all(record["cluster_sizes"] == reference["cluster_sizes"] for record in records)
    center_differences = []
    for record in records:
        for left, right in zip(record["centers"], reference["centers"]):
            center_differences.extend(abs(a - b) for a, b in zip(left, right))
    max_center_difference = max(center_differences or [0.0])
    passed = num_iter_equal and max(cost_differences or [0.0]) <= TOLERANCE and cluster_equal and max_center_difference <= TOLERANCE
    return {"model_id": model_id, "trial_count": len(records), "numIter_equal": num_iter_equal, "trainingCost_absolute_differences": cost_differences, "max_trainingCost_difference": max(cost_differences or [0.0]), "cluster_sizes_equal": cluster_equal, "max_center_difference": max_center_difference, "tolerance": TOLERANCE, "pass": passed}


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
    output_dir = project_root / "output" / "benchmark" / "stage20"
    event_dir = output_dir / "spark-events"
    report_path = project_root / "reports" / "stage20_three_model_runtime_benchmark.txt"
    summary_path = output_dir / "stage20_final_summary.json"
    blockers: List[str] = []
    warnings: List[str] = ["Runtime is machine-, JVM-, Spark-, and cohort-specific; no universal speed claim is made."]
    spark = None
    cached = None
    event_logging_enabled = False
    event_logging_issue: str | None = None
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        event_dir.mkdir(parents=True, exist_ok=True)
        blockers, authority_context = audit_authorities(project_root)
        if blockers:
            failed = {"stage": 20, "authority": "THREE_MODEL_RUNTIME_BENCHMARK_1M_V1", "runtime_benchmark_complete": False, "ready_for_stage21": False, "blockers": blockers}
            write_json(summary_path, failed)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("STAGE 20 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_21: False\n", encoding="utf-8")
            return 3

        cohort_path = project_root / "output" / "comparison" / "stage17" / "comparison_cohort_1m"
        main_model_path = project_root / "output" / "models" / "main_kmeans_final"
        enhanced_model_path = project_root / "output" / "models" / "enhanced_kmeans_final"
        event_uri = event_dir.resolve().as_uri()
        input_read_start = time.perf_counter()
        try:
            spark = SparkSession.builder.master("local[4]").appName("Stage20-Controlled-1M-Runtime-Benchmark").config("spark.sql.shuffle.partitions", "64").config("spark.sql.adaptive.enabled", "true").config("spark.eventLog.enabled", "true").config("spark.eventLog.dir", event_uri).config("spark.local.dir", str(project_root.parent / "spark_tmp")).getOrCreate()
            event_logging_enabled = True
        except Exception as event_error:
            event_logging_issue = f"Spark event logging unavailable on this Windows runtime: {type(event_error).__name__}: {event_error}"
            warnings.append(event_logging_issue)
            spark = None
            # The benchmark remains valid without optional event-log evidence,
            # provided this exact environment limitation is recorded.
            spark = SparkSession.builder.master("local[4]").appName("Stage20-Controlled-1M-Runtime-Benchmark").config("spark.sql.shuffle.partitions", "64").config("spark.sql.adaptive.enabled", "true").config("spark.eventLog.enabled", "false").config("spark.local.dir", str(project_root.parent / "spark_tmp")).getOrCreate()
            event_logging_enabled = False
        spark.sparkContext.setLogLevel("ERROR")
        enable_windows_java_filesystem_fallback(spark)
        input_df = spark.read.parquet(str(cohort_path)).select(*REQUIRED_COLUMNS)
        input_read_runtime = time.perf_counter() - input_read_start
        cache_start = time.perf_counter()
        cached = input_df.persist(StorageLevel.MEMORY_AND_DISK)
        cached_count = cached.count()
        cache_materialization_runtime = time.perf_counter() - cache_start
        validation = input_validation(cached)
        write_json(output_dir / "benchmark_input_validation.json", validation)
        if not validation["pass"]:
            raise ValueError("Benchmark input validation failed.")

        stage18_thresholds = authority_context["stage18_thresholds"]
        if not authority_context["thresholds_match_expected"]:
            raise ValueError("Stage 18 threshold artifact does not match the locked threshold contract.")
        thresholds = {feature: [float(value) for value in stage18_thresholds[feature]] for feature in RFM_FEATURES}

        load_times: Dict[str, Any] = {"R1": None}
        load_start = time.perf_counter()
        main_model = KMeansModel.load(str(main_model_path))
        load_times["K1"] = time.perf_counter() - load_start
        load_start = time.perf_counter()
        enhanced_model = KMeansModel.load(str(enhanced_model_path))
        load_times["K2"] = time.perf_counter() - load_start
        models = {"R1": None, "K1": main_model, "K2": enhanced_model}
        expected_counts = {"R1": R1_COUNTS, "K1": K1_COUNTS, "K2": K2_COUNTS}

        track_a_warmup = []
        for model_id in ["R1", "K1", "K2"]:
            warm = run_assignment_trial(model_id, cached, models[model_id], thresholds, expected_counts[model_id])
            track_a_warmup.append({"model_id": model_id, "pass": warm["pass"], "distribution": warm["distribution"]})
        track_a_order = [["R1", "K1", "K2"], ["K1", "K2", "R1"], ["K2", "R1", "K1"]]
        track_a_records: List[Dict[str, Any]] = []
        track_a_by_model = {"R1": [], "K1": [], "K2": []}
        for round_index, order in enumerate(track_a_order, 1):
            for order_index, model_id in enumerate(order, 1):
                record = run_assignment_trial(model_id, cached, models[model_id], thresholds, expected_counts[model_id])
                record.update({"track": "A", "round": round_index, "order_index": order_index})
                track_a_records.append(record)
                track_a_by_model[model_id].append(record)

        # Row-level equality is checked once per locked model after timing. It
        # is deliberately outside timed assignment sections.
        main_validation_df = main_model.transform(cached.withColumn("features", col("eval_rfm3d_scaled")))
        enhanced_validation_df = enhanced_model.transform(cached.withColumn("features_improved", col("eval_erfm5d_scaled")))
        k1_prediction_mismatch = compare_prediction_rows(cached, main_validation_df, "main_prediction")
        k2_prediction_mismatch = compare_prediction_rows(cached, enhanced_validation_df, "enhanced_prediction")
        track_a_equality = {"R1": all(record["distribution"]["pass"] for record in track_a_records if record["model_id"] == "R1"), "K1": k1_prediction_mismatch == 0 and all(record["distribution"]["pass"] for record in track_a_records if record["model_id"] == "K1"), "K2": k2_prediction_mismatch == 0 and all(record["distribution"]["pass"] for record in track_a_records if record["model_id"] == "K2")}
        if not all(track_a_equality.values()):
            raise ValueError("Track A locked assignment equality failed.")
        track_a_summary = {}
        for model_id, records in track_a_by_model.items():
            metric = stats([float(record["duration_sec"]) for record in records])
            track_a_summary[model_id] = {"model_id": model_id, "model_name": {"R1": "Rule-based RFM", "K1": "Main K-Means", "K2": "Enhanced K-Means"}[model_id], **metric, "customers_per_second": EXPECTED_COUNT / metric["median_sec"], "model_load_sec": load_times[model_id], "assignment_only": True, "prediction_mismatch_vs_locked": {"R1": 0, "K1": k1_prediction_mismatch, "K2": k2_prediction_mismatch}[model_id]}
        write_json(output_dir / "track_a_assignment_trials.json", {"warmup": track_a_warmup, "measured_order": track_a_order, "trials": track_a_records, "model_load_sec": load_times, "locked_assignment_equality": track_a_equality})
        write_csv(output_dir / "track_a_assignment_summary.csv", ["ModelID", "ModelName", "MedianAssignmentSec", "MeanAssignmentSec", "StdAssignmentSec", "CustomersPerSecond", "ModelLoadSec", "PredictionMismatchVsLocked"], [{"ModelID": model_id, "ModelName": value["model_name"], "MedianAssignmentSec": value["median_sec"], "MeanAssignmentSec": value["mean_sec"], "StdAssignmentSec": value["std_sec"], "CustomersPerSecond": value["customers_per_second"], "ModelLoadSec": value["model_load_sec"], "PredictionMismatchVsLocked": value["prediction_mismatch_vs_locked"]} for model_id, value in track_a_summary.items()])

        track_b_warmup = []
        track_b_sources = {"R1": (None, None), "K1": (4, "eval_rfm3d_scaled"), "K2": (5, "eval_erfm5d_scaled")}
        for model_id in ["R1", "K1", "K2"]:
            if model_id == "R1":
                warm = run_build_rule_trial(cached, expected_counts[model_id], thresholds)
            else:
                warm = run_build_kmeans_trial(model_id, cached, track_b_sources[model_id][0], track_b_sources[model_id][1])
            track_b_warmup.append({"model_id": model_id, "pass": warm["pass"]})
        track_b_order = [["R1", "K1", "K2"], ["K1", "K2", "R1"], ["K2", "R1", "K1"]]
        track_b_records: List[Dict[str, Any]] = []
        track_b_by_model = {"R1": [], "K1": [], "K2": []}
        for round_index, order in enumerate(track_b_order, 1):
            for order_index, model_id in enumerate(order, 1):
                if model_id == "R1":
                    record = run_build_rule_trial(cached, expected_counts[model_id], thresholds)
                else:
                    record = run_build_kmeans_trial(model_id, cached, track_b_sources[model_id][0], track_b_sources[model_id][1])
                record.update({"track": "B", "round": round_index, "order_index": order_index})
                track_b_records.append(record)
                track_b_by_model[model_id].append(record)
        rule_threshold_repro = all(record["thresholds"] == track_b_by_model["R1"][0]["thresholds"] and record["distribution"]["counts"] == R1_COUNTS for record in track_b_by_model["R1"])
        if not rule_threshold_repro:
            raise ValueError("Track B R1 threshold or segment reproducibility failed.")
        determinism = {model_id: deterministic_validation(track_b_by_model, model_id) for model_id in ["K1", "K2"]}
        if not all(item["pass"] for item in determinism.values()):
            warnings.append("Comparison-only K-Means trials were not materially identical under the documented tolerance.")
            raise ValueError("Track B K-Means determinism validation failed.")
        track_b_summary = {}
        for model_id, records in track_b_by_model.items():
            total_stats = stats([float(record["total_fit_plus_assign_sec"]) for record in records]) if model_id != "R1" else stats([float(record["duration_sec"]) for record in records])
            fit_values = [float(record["fit_only_sec"]) for record in records] if model_id != "R1" else []
            assign_values = [float(record["assign_after_fit_sec"]) for record in records] if model_id != "R1" else []
            track_b_summary[model_id] = {"model_id": model_id, "model_name": {"R1": "Rule-based RFM", "K1": "Main K-Means", "K2": "Enhanced K-Means"}[model_id], **total_stats, "median_fit_only_sec": statistics.median(fit_values) if fit_values else None, "median_assign_after_fit_sec": statistics.median(assign_values) if assign_values else None, "customers_per_second_end_to_end": EXPECTED_COUNT / total_stats["median_sec"], "numIter": None if model_id == "R1" else records[0]["numIter"], "rule_reproducibility_pass": rule_threshold_repro if model_id == "R1" else None}
        write_json(output_dir / "track_b_build_assign_trials.json", {"warmup": track_b_warmup, "measured_order": track_b_order, "trials": track_b_records, "rule_reproducibility_pass": rule_threshold_repro})
        write_csv(output_dir / "track_b_build_assign_summary.csv", ["ModelID", "ModelName", "MedianBuildAssignSec", "MeanBuildAssignSec", "StdBuildAssignSec", "MedianFitOnlySec", "MedianAssignAfterFitSec", "CustomersPerSecond_EndToEnd", "NumIter"], [{"ModelID": model_id, "ModelName": value["model_name"], "MedianBuildAssignSec": value["median_sec"], "MeanBuildAssignSec": value["mean_sec"], "StdBuildAssignSec": value["std_sec"], "MedianFitOnlySec": value["median_fit_only_sec"], "MedianAssignAfterFitSec": value["median_assign_after_fit_sec"], "CustomersPerSecond_EndToEnd": value["customers_per_second_end_to_end"], "NumIter": value["numIter"] if value["numIter"] is not None else "N/A"} for model_id, value in track_b_summary.items()])
        write_json(output_dir / "benchmark_determinism_validation.json", {"rule_based_rfm": {"trial_count": 3, "thresholds_match": rule_threshold_repro, "segment_counts_match": rule_threshold_repro, "pass": rule_threshold_repro}, "K1": determinism["K1"], "K2": determinism["K2"]})

        def pair_ratios(summary: Dict[str, Any], time_key: str) -> List[Dict[str, Any]]:
            pairs = []
            for left, right in [("R1", "K1"), ("R1", "K2"), ("K1", "K2")]:
                left_time = float(summary[left][time_key]) if time_key in summary[left] else float(summary[left]["median_sec"])
                right_time = float(summary[right][time_key]) if time_key in summary[right] else float(summary[right]["median_sec"])
                pairs.append({"numerator_model": left, "denominator_model": right, "ratio_definition": f"median {left} runtime / median {right} runtime", "ratio": left_time / right_time})
            return pairs

        runtime_ratios = {"track_a_assignment": pair_ratios(track_a_summary, "median_sec"), "track_b_build_assign": pair_ratios(track_b_summary, "median_sec")}
        write_json(output_dir / "runtime_ratios.json", runtime_ratios)

        application_id = spark.sparkContext.applicationId
        ui_port = spark.conf.get("spark.ui.port", "4040")
        environment = {"stage": 20, "authority": "THREE_MODEL_RUNTIME_BENCHMARK_1M_V1", "cohort_count": EXPECTED_COUNT, "cohort_fingerprint": COHORT_FINGERPRINT, "spark_version": spark.version, "application_id": application_id, "master": "local[4]", "shuffle_partitions": 64, "aqe": True, "python_version": platform.python_version(), "java_version": spark._jvm.java.lang.System.getProperty("java.version"), "os": platform.platform(), "cpu_logical_processors": os.cpu_count(), "system_memory_bytes": None, "cache_policy": "MEMORY_AND_DISK", "timing_api": "time.perf_counter", "warmup_trials": 1, "measured_trials": 3, "primary_runtime_statistic": "median", "input_read_runtime_sec": input_read_runtime, "cache_materialization_runtime_sec": cache_materialization_runtime, "model_load_sec": load_times, "event_log_enabled": event_logging_enabled, "event_log_path": event_uri, "event_log_issue": event_logging_issue, "warnings": ["JIT compilation, garbage collection, OS scheduling, disk spill, and memory pressure can vary runtime.", "Windows Hadoop/native library warnings are retained as environment notes."]}
        try:
            import psutil
            environment["system_memory_bytes"] = int(psutil.virtual_memory().total)
        except Exception:
            pass
        write_json(output_dir / "benchmark_environment.json", environment)
        spark_manifest = {"spark_application_id": application_id, "application_name": "Stage20-Controlled-1M-Runtime-Benchmark", "master": "local[4]", "spark_version": spark.version, "event_log_enabled": event_logging_enabled, "event_log_path": event_uri, "event_log_directory": str(event_dir), "event_log_issue": event_logging_issue, "spark_ui_port": ui_port, "benchmark_operations": ["Track A R1 assignment", "Track A K1 assignment", "Track A K2 assignment", "Track B R1 rule derivation plus assignment", "Track B K1 fit plus assignment", "Track B K2 fit plus assignment"], "instructions": "While the application is running, open Spark UI at http://localhost:<spark_ui_port>. After completion, use the event-log path with a compatible Spark History Server when event logging is enabled; no screenshots are fabricated by Stage 20.", "resource_diagnostics": {"available": False, "reason": "Optional event-log stage/task parsing is not required for core benchmark validity."}}
        write_json(output_dir / "spark_ui_evidence_manifest.json", spark_manifest)

        fastest_assignment = min(track_a_summary, key=lambda model_id: track_a_summary[model_id]["median_sec"])
        fastest_build = min(track_b_summary, key=lambda model_id: track_b_summary[model_id]["median_sec"])
        summary = {"stage": 20, "authority": "THREE_MODEL_RUNTIME_BENCHMARK_1M_V1", "cohort_authority": COHORT_AUTHORITY, "quality_authority": QUALITY_AUTHORITY, "rule_authority": RULE_AUTHORITY, "cohort_count": EXPECTED_COUNT, "cohort_fingerprint": COHORT_FINGERPRINT, "input_validation_pass": True, "track_a_assignment": track_a_summary, "track_b_build_assign": track_b_summary, "track_a_assignment_equality": track_a_equality, "track_b_rule_reproducibility_pass": rule_threshold_repro, "determinism_validation_pass": all(item["pass"] for item in determinism.values()), "spark_evidence": spark_manifest, "fastest_assignment_model": fastest_assignment, "fastest_end_to_end_build_model": fastest_build, "k1_vs_k2_fit_runtime": track_b_summary["K1"]["median_fit_only_sec"] / track_b_summary["K2"]["median_fit_only_sec"], "k1_vs_k2_assignment_runtime": track_b_summary["K1"]["median_assign_after_fit_sec"] / track_b_summary["K2"]["median_assign_after_fit_sec"], "quality_models_modified": False, "final_model_authorities_modified": False, "runtime_quality_combined_score_used": False, "gmm_executed": False, "csv_vs_parquet_benchmark_executed": False, "cache_vs_no_cache_benchmark_executed": False, "warnings": warnings, "blockers": [], "runtime_benchmark_complete": True, "ready_for_stage21": True}
        write_json(summary_path, summary)
        cached.unpersist()
        cached = None
        spark.stop()
        spark = None

        artifact_paths = [output_dir / name for name in ["benchmark_environment.json", "benchmark_input_validation.json", "track_a_assignment_trials.json", "track_a_assignment_summary.csv", "track_b_build_assign_trials.json", "track_b_build_assign_summary.csv", "benchmark_determinism_validation.json", "runtime_ratios.json", "spark_ui_evidence_manifest.json", "stage20_final_summary.json"]] + [report_path, project_root / "reports" / "stage20_run.log"]
        readback = readback_artifacts(artifact_paths, project_root)
        summary["artifact_readback"] = readback
        summary["artifact_readback_pass"] = all(readback.values())
        summary["ready_for_stage21"] = bool(summary["artifact_readback_pass"])
        if not summary["ready_for_stage21"]:
            summary["runtime_benchmark_complete"] = False
            summary["blockers"] = ["One or more Stage 20 artifact read-backs failed."]
        write_json(summary_path, summary)
        report_lines = ["STAGE 20 — CONTROLLED 1M RUNTIME & COMPUTATIONAL BENCHMARK", "", f"AUTHORITY: {summary['authority']}", f"COHORT: {COHORT_AUTHORITY} / {EXPECTED_COUNT} / {COHORT_FINGERPRINT}", f"SPARK_APPLICATION_ID: {application_id}", f"FASTEST_ASSIGNMENT_MODEL: {fastest_assignment}", f"FASTEST_END_TO_END_BUILD_MODEL: {fastest_build}", "", "TRACK A SUMMARY:", str(track_a_summary), "", "TRACK B SUMMARY:", str(track_b_summary), "", "RUNTIME RATIOS:", str(runtime_ratios), "", f"DETERMINISM_PASS: {summary['determinism_validation_pass']}", f"RULE_REPRODUCIBILITY_PASS: {summary['track_b_rule_reproducibility_pass']}", "QUALITY_MODELS_MODIFIED: False", "FINAL_MODEL_AUTHORITIES_MODIFIED: False", "RUNTIME_QUALITY_COMBINED_SCORE_USED: False", "GMM_EXECUTED: False", "", f"RUNTIME_BENCHMARK_COMPLETE: {summary['runtime_benchmark_complete']}", f"READY_FOR_STAGE_21: {summary['ready_for_stage21']}"]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        # The report itself is part of the read-back package; update the final
        # summary one last time after it is written.
        readback = readback_artifacts(artifact_paths, project_root)
        summary["artifact_readback"] = readback
        summary["artifact_readback_pass"] = all(readback.values())
        summary["ready_for_stage21"] = bool(summary["artifact_readback_pass"])
        summary["runtime_benchmark_complete"] = bool(summary["runtime_benchmark_complete"] and summary["ready_for_stage21"])
        write_json(summary_path, summary)
        print("AUTHORITY_AUDIT_PASS=True")
        print("INPUT_VALIDATION_PASS=True")
        print("TRACK_A_ASSIGNMENT_EQUALITY_PASS=True")
        print("TRACK_B_RULE_REPRODUCIBILITY_PASS=True")
        print("KMEANS_DETERMINISM_PASS=True")
        print(f"RUNTIME_BENCHMARK_COMPLETE={summary['runtime_benchmark_complete']}")
        print(f"READY_FOR_STAGE_21={summary['ready_for_stage21']}")
        return 0 if summary["ready_for_stage21"] else 3
    except Exception as error:
        blockers.append(str(error))
        failed = {"stage": 20, "authority": "THREE_MODEL_RUNTIME_BENCHMARK_1M_V1", "runtime_benchmark_complete": False, "ready_for_stage21": False, "warnings": warnings, "blockers": blockers}
        write_json(summary_path, failed)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("STAGE 20 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_21: False\n", encoding="utf-8")
        print(f"STAGE 20 FAILED CLOSED: {error}")
        return 3
    finally:
        if cached is not None:
            cached.unpersist()
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    sys.exit(main())
