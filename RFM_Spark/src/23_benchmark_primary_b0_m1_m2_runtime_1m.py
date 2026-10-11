"""Stage 23: primary controlled B0/M1/M2 runtime benchmark on locked 1M cohort.

Track A measures assignment/inference using the three saved authorities.
Track B measures comparison-only fresh K-Means fit plus assignment.  Track B
never saves or changes a final model.
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
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, element_at, size
from pyspark.storagelevel import StorageLevel


STAGE = 23
AUTHORITY = "PRIMARY_B0_M1_M2_RUNTIME_BENCHMARK_1M_V1"
COHORT_AUTHORITY = "DETERMINISTIC_1M_COMPARISON_COHORT"
COHORT_COUNT = 1_000_000
COHORT_FINGERPRINT = "ca082ec1d828df9c34a10105ae11599c3cc0653dc5b4d2c11af18ec82b781cf5"
QUALITY_AUTHORITY = "PRIMARY_B0_M1_M2_QUALITY_COMPARISON_V1"
TOLERANCE = 1e-8
MODEL_IDS = ["B0", "M1", "M2"]
EXPECTED_COUNTS = {
    "B0": [535416, 145205, 301081, 18298],
    "M1": [535490, 145123, 301103, 18284],
    "M2": [154437, 126550, 197817, 252287, 268909],
}
MODEL_SPECS = {
    "B0": {"name": "Basic K-Means Baseline", "k": 4, "space": "RFM3D", "source": "eval_rfm3d_scaled", "features": "features", "init": "random", "label": "b0_prediction", "dimension": 3},
    "M1": {"name": "Project Main K-Means", "k": 4, "space": "RFM3D", "source": "eval_rfm3d_scaled", "features": "features", "init": "k-means||", "label": "main_prediction", "dimension": 3},
    "M2": {"name": "Enhanced K-Means", "k": 5, "space": "ERFM5D", "source": "eval_erfm5d_scaled", "features": "features_improved", "init": "k-means||", "label": "enhanced_prediction", "dimension": 5},
}


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
    """Avoid Hadoop Windows native filesystem probing for local Spark work."""
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
    stage22 = read_json(project_root / "output" / "comparison" / "stage22" / "stage22_final_summary.json")
    stage10 = read_json(project_root / "output" / "analysis" / "stage10" / "final_model_summary.json")
    stage15 = read_json(project_root / "output" / "analysis" / "stage15" / "final_enhanced_model_summary.json")
    checks = {
        "stage17": stage17.get("stage") == 17 and stage17.get("cohort_authority") == COHORT_AUTHORITY and stage17.get("cohort_count") == COHORT_COUNT and stage17.get("cohort_customer_id_sha256") == COHORT_FINGERPRINT and stage17.get("source_alignment_pass") is True and stage17.get("deterministic_rerun_validation", {}).get("pass") is True and stage17.get("cohort_readback", {}).get("pass") is True and stage17.get("ready_for_stage18") is True,
        "stage21": stage21.get("stage") == 21 and stage21.get("authority") == "BASIC_KMEANS_BASELINE_V1" and stage21.get("comparison_cohort_count") == COHORT_COUNT and stage21.get("b0_prediction_sha256") == "98e75c0b34cff5ffade7e8b62cd3176b2983062c414b2cf62be14f3f0819bf09" and stage21.get("primary_reproducibility_pass") is True and stage21.get("customer_universe_match_pass") is True and stage21.get("ready_for_stage22") is True,
        "stage22": stage22.get("stage") == 22 and stage22.get("authority") == QUALITY_AUTHORITY and stage22.get("cohort_count") == COHORT_COUNT and stage22.get("cohort_customer_id_sha256") == COHORT_FINGERPRINT and stage22.get("primary_quality_comparison_complete") is True and stage22.get("ready_for_stage23") is True and stage22.get("runtime_quality_combined_score_used") is False,
        "stage10": stage10.get("stage") == 10 and stage10.get("authority") == "MAIN_KMEANS_FINAL" and stage10.get("selected_k") == 4 and stage10.get("ready_for_stage11") is True,
        "stage15": stage15.get("stage") == 15 and stage15.get("authority") == "ENHANCED_KMEANS_FINAL" and stage15.get("selected_k") == 5 and stage15.get("ready_for_stage16") is True,
    }
    for key, ok in checks.items():
        if not ok:
            blockers.append(f"{key} authority, fingerprint, or readiness gate mismatch.")
    return blockers, {"stage17": stage17, "stage21": stage21, "stage22": stage22, "stage10": stage10, "stage15": stage15, "checks": checks}


def validate_vector(df: DataFrame, column: str, dimension: int) -> Dict[str, Any]:
    values = vector_to_array(col(column))
    null_count = df.where(col(column).isNull()).count()
    wrong_dimension = df.where(col(column).isNotNull() & (size(values) != dimension)).count()
    nan_count = sum(df.where(element_at(values, i) != element_at(values, i)).count() for i in range(1, dimension + 1))
    infinity_count = sum(df.where((element_at(values, i) == float("inf")) | (element_at(values, i) == float("-inf"))).count() for i in range(1, dimension + 1))
    return {"column": column, "dimension": dimension, "null": null_count, "wrong_dimension": wrong_dimension, "nan": nan_count, "infinity": infinity_count, "pass": null_count == 0 and wrong_dimension == 0 and nan_count == 0 and infinity_count == 0}


def input_validation(df: DataFrame, b0_raw: DataFrame) -> Dict[str, Any]:
    row_count = df.count()
    distinct_count = df.select("customer_id").distinct().count()
    label_checks: Dict[str, Any] = {}
    for model_id, spec in MODEL_SPECS.items():
        label = spec["label"]
        label_checks[model_id] = {"column": label, "null": df.where(col(label).isNull()).count(), "invalid": df.where(~col(label).isin(list(range(spec["k"])))).count()}
    vectors = {"RFM3D": validate_vector(df, "eval_rfm3d_scaled", 3), "ERFM5D": validate_vector(df, "eval_erfm5d_scaled", 5)}
    b0_count = b0_raw.count()
    b0_distinct = b0_raw.select("customer_id").distinct().count()
    return {"row_count": row_count, "distinct_customer_id": distinct_count, "duplicate_customer_id": row_count - distinct_count, "b0_source_row_count": b0_count, "b0_source_distinct_customer_id": b0_distinct, "b0_source_duplicate_customer_id": b0_count - b0_distinct, "label_checks": label_checks, "vector_checks": vectors, "same_universe_after_join": row_count == COHORT_COUNT and distinct_count == COHORT_COUNT and b0_count == COHORT_COUNT and b0_distinct == COHORT_COUNT, "pass": row_count == COHORT_COUNT and distinct_count == COHORT_COUNT and row_count == distinct_count and b0_count == COHORT_COUNT and b0_distinct == COHORT_COUNT and b0_count == b0_distinct and all(item["null"] == 0 and item["invalid"] == 0 for item in label_checks.values()) and all(item["pass"] for item in vectors.values())}


def distribution(df: DataFrame, prediction_col: str, expected: List[int]) -> Dict[str, Any]:
    rows = df.groupBy(prediction_col).count().orderBy(prediction_col).collect()
    ids = [int(row[prediction_col]) for row in rows]
    counts = [int(row["count"]) for row in rows]
    return {"cluster_ids": ids, "counts": counts, "expected_counts": expected, "pass": ids == list(range(len(expected))) and counts == expected and sum(counts) == COHORT_COUNT}


def stats(values: List[float]) -> Dict[str, Any]:
    return {"min_sec": min(values), "max_sec": max(values), "mean_sec": statistics.mean(values), "median_sec": statistics.median(values), "population_std_sec": statistics.pstdev(values), "trial_times_sec": values}


def model_centers(model: Any) -> List[List[float]]:
    return [[float(value) for value in center] for center in model.clusterCenters()]


def load_model(model_id: str, project_root: Path) -> Tuple[KMeansModel, float]:
    paths = {"B0": project_root / "output" / "models" / "basic_kmeans_baseline", "M1": project_root / "output" / "models" / "main_kmeans_final", "M2": project_root / "output" / "models" / "enhanced_kmeans_final"}
    started = time.perf_counter()
    model = KMeansModel.load(str(paths[model_id]))
    return model, time.perf_counter() - started


def run_assignment_trial(model_id: str, df: DataFrame, model: KMeansModel) -> Dict[str, Any]:
    spec = MODEL_SPECS[model_id]
    started = time.perf_counter()
    assigned = model.transform(df.withColumn(spec["features"], col(spec["source"])))
    counts = distribution(assigned, "prediction", EXPECTED_COUNTS[model_id])
    elapsed = time.perf_counter() - started
    return {"model_id": model_id, "duration_sec": elapsed, "distribution": counts, "timed_section": "transform + groupBy(prediction).count().collect()", "pass": elapsed > 0 and counts["pass"]}


def prediction_mismatch(df: DataFrame, model_id: str, model: KMeansModel) -> int:
    spec = MODEL_SPECS[model_id]
    actual = model.transform(df.withColumn(spec["features"], col(spec["source"]))).select("customer_id", col("prediction").cast("long").alias("actual"))
    expected = df.select("customer_id", col(spec["label"]).cast("long").alias("expected"))
    return actual.join(expected, "customer_id", "inner").where(col("actual") != col("expected")).count()


def run_build_trial(model_id: str, df: DataFrame) -> Dict[str, Any]:
    spec = MODEL_SPECS[model_id]
    benchmark_features = "benchmark_features"
    benchmark_prediction = "benchmark_prediction"
    benchmark_df = df.withColumn(benchmark_features, col(spec["source"]))
    estimator = KMeans(k=spec["k"], seed=42, featuresCol=benchmark_features, predictionCol=benchmark_prediction, distanceMeasure="euclidean", initMode=spec["init"], maxIter=100, tol=1e-4)
    fit_start = time.perf_counter()
    fitted = estimator.fit(benchmark_df)
    fit_elapsed = time.perf_counter() - fit_start
    assign_start = time.perf_counter()
    assigned = fitted.transform(benchmark_df)
    rows = assigned.groupBy(benchmark_prediction).count().orderBy(benchmark_prediction).collect()
    assign_elapsed = time.perf_counter() - assign_start
    total_elapsed = time.perf_counter() - fit_start
    counts = [int(row["count"]) for row in rows]
    model_summary = fitted.summary
    return {"model_id": model_id, "fit_only_sec": fit_elapsed, "assign_after_fit_sec": assign_elapsed, "total_fit_plus_assign_sec": total_elapsed, "numIter": int(model_summary.numIter), "trainingCost": float(model_summary.trainingCost), "cluster_ids": [int(row[benchmark_prediction]) for row in rows], "cluster_sizes": counts, "centers": model_centers(fitted), "configuration": {"k": spec["k"], "seed": 42, "featuresCol": benchmark_features, "predictionCol": benchmark_prediction, "distanceMeasure": "euclidean", "initMode": spec["init"], "maxIter": 100, "tol": 1e-4, "source_feature": spec["source"]}, "pass": len(counts) == spec["k"] and sum(counts) == COHORT_COUNT and all(finite(value) for value in [fit_elapsed, assign_elapsed, total_elapsed, model_summary.trainingCost])}


def align_centers(reference: List[List[float]], candidate: List[List[float]], reference_sizes: List[int], candidate_sizes: List[int]) -> Dict[str, Any]:
    best: Dict[str, Any] | None = None
    for perm in permutations(range(len(candidate))):
        squared_cost = 0.0
        absolute_differences: List[float] = []
        for reference_index, candidate_index in enumerate(perm):
            differences = [abs(float(a) - float(b)) for a, b in zip(reference[reference_index], candidate[candidate_index])]
            absolute_differences.extend(differences)
            squared_cost += sum(value * value for value in differences)
        record = {"reference_to_candidate_permutation": list(perm), "sum_squared_center_difference": squared_cost, "max_absolute_center_difference": max(absolute_differences or [0.0]), "matched_cluster_sizes": [candidate_sizes[index] for index in perm], "cluster_sizes_match": [candidate_sizes[index] for index in perm] == reference_sizes}
        if best is None or record["sum_squared_center_difference"] < best["sum_squared_center_difference"]:
            best = record
    assert best is not None
    return best


def deterministic_validation(records: Dict[str, List[Dict[str, Any]]], model_id: str) -> Dict[str, Any]:
    model_records = records[model_id]
    reference = model_records[0]
    checks = []
    for index, record in enumerate(model_records):
        alignment = align_centers(reference["centers"], record["centers"], reference["cluster_sizes"], record["cluster_sizes"])
        cost_difference = abs(float(record["trainingCost"]) - float(reference["trainingCost"]))
        item = {"trial_index": index + 1, "numIter_equal": int(record["numIter"]) == int(reference["numIter"]), "trainingCost_absolute_difference": cost_difference, "center_alignment": alignment, "pass": int(record["numIter"]) == int(reference["numIter"]) and cost_difference <= TOLERANCE and alignment["cluster_sizes_match"] and alignment["max_absolute_center_difference"] <= TOLERANCE}
        checks.append(item)
    return {"model_id": model_id, "trial_count": len(model_records), "tolerance": TOLERANCE, "trials": checks, "pass": all(item["pass"] for item in checks)}


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
        try:
            key = str(path.relative_to(project_root))
        except ValueError:
            key = str(path)
        result[key] = ok
    return result


def stage22_quality_context(authority_context: Dict[str, Any]) -> Dict[str, Any]:
    stage22 = authority_context["stage22"]
    b0_m1 = stage22.get("b0_to_m1_analysis", {})
    m1_m2 = stage22.get("m1_to_m2_enhancement_analysis", {}).get("ERFM5D", {})
    return {"b0_m1": {"ARI": b0_m1.get("ARI"), "NMI": b0_m1.get("NMI"), "mismatch_ratio": b0_m1.get("cluster_membership_mismatch_ratio_after_optimal_alignment"), "silhouette_delta_b0_minus_m1": b0_m1.get("silhouette_delta_b0_minus_m1")}, "m1_m2": {"silhouette_relative_improvement_pct": m1_m2.get("silhouette_relative_improvement_pct"), "sse_per_customer_reduction_pct": m1_m2.get("sse_per_customer_reduction_pct"), "entropy_change": m1_m2.get("entropy_change"), "cv_reduction_pct": m1_m2.get("cv_reduction_pct")}}


def pair_ratios(summary: Dict[str, Any], metric_key: str) -> List[Dict[str, Any]]:
    pairs = []
    for left, right in [("B0", "M1"), ("B0", "M2"), ("M1", "M2")]:
        left_value = float(summary[left][metric_key])
        right_value = float(summary[right][metric_key])
        pairs.append({"numerator_model": left, "denominator_model": right, "ratio_definition": f"median {metric_key} {left} / median {metric_key} {right}", "ratio": left_value / right_value})
    return pairs


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "output" / "benchmark" / "stage23"
    event_dir = output_dir / "spark-events"
    report_path = project_root / "reports" / "stage23_primary_b0_m1_m2_runtime_benchmark.txt"
    log_path = project_root / "reports" / "stage23_run.log"
    summary_path = output_dir / "stage23_final_summary.json"
    blockers: List[str] = []
    warnings: List[str] = ["Runtime is machine-, JVM-, Spark-, cache-, and cohort-specific; no universal speed claim is made.", "Track B is comparison-only and does not replace or serialize any final model."]
    spark: SparkSession | None = None
    cached: DataFrame | None = None
    event_logging_enabled = False
    event_logging_issue: str | None = None
    application_id: str | None = None
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        event_dir.mkdir(parents=True, exist_ok=True)
        blockers, authority_context = audit_authorities(project_root)
        if blockers:
            failed = {"stage": STAGE, "authority": AUTHORITY, "quality_authority": QUALITY_AUTHORITY, "runtime_benchmark_complete": False, "ready_for_stage24": False, "blockers": blockers}
            write_json(summary_path, failed)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("STAGE 23 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_24: False\n", encoding="utf-8")
            return 3

        cohort_path = project_root / "output" / "comparison" / "stage17" / "comparison_cohort_1m"
        b0_predictions_path = project_root / "output" / "comparison" / "stage21" / "basic_kmeans_predictions_1m"
        event_uri = event_dir.resolve().as_uri()
        spark_builder = SparkSession.builder.master("local[4]").appName("Stage23-Primary-B0-M1-M2-Runtime-Benchmark").config("spark.sql.shuffle.partitions", "64").config("spark.sql.adaptive.enabled", "true").config("spark.sql.inMemoryColumnarStorage.partitionPruning", "true").config("spark.storage.memoryMapThreshold", "0").config("spark.local.dir", str(project_root.parent / "spark_tmp"))
        try:
            spark = spark_builder.config("spark.eventLog.enabled", "true").config("spark.eventLog.dir", event_uri).getOrCreate()
            event_logging_enabled = True
        except Exception as event_error:
            event_logging_issue = f"Spark event logging unavailable on this Windows runtime: {type(event_error).__name__}: {event_error}"
            warnings.append(event_logging_issue)
            if spark is not None:
                try:
                    spark.stop()
                except Exception:
                    pass
                spark = None
            spark = spark_builder.config("spark.eventLog.enabled", "false").getOrCreate()
            event_logging_enabled = False
        assert spark is not None
        spark.sparkContext.setLogLevel("ERROR")
        enable_windows_java_filesystem_fallback(spark)

        input_read_start = time.perf_counter()
        cohort = spark.read.parquet(str(cohort_path)).select("customer_id", "eval_rfm3d_scaled", "eval_erfm5d_scaled", "main_prediction", "enhanced_prediction")
        b0_raw_read = spark.read.parquet(str(b0_predictions_path))
        b0_column = "b0_prediction" if "b0_prediction" in b0_raw_read.columns else "prediction"
        b0_raw = b0_raw_read.select("customer_id", col(b0_column).cast("int").alias("b0_prediction"))
        input_read_runtime = time.perf_counter() - input_read_start
        joined = cohort.join(b0_raw, "customer_id", "inner").select("customer_id", "eval_rfm3d_scaled", "eval_erfm5d_scaled", "b0_prediction", "main_prediction", "enhanced_prediction")
        cache_start = time.perf_counter()
        cached = joined.persist(StorageLevel.MEMORY_AND_DISK)
        cached_count = cached.count()
        cache_materialization_runtime = time.perf_counter() - cache_start
        validation = input_validation(cached, b0_raw)
        validation["joined_row_count"] = cached_count
        validation["cohort_path"] = str(cohort_path)
        validation["b0_predictions_path"] = str(b0_predictions_path)
        write_json(output_dir / "benchmark_input_validation.json", validation)
        if not validation["pass"] or cached_count != COHORT_COUNT:
            raise ValueError("Locked Stage 23 benchmark input validation failed.")

        load_times: Dict[str, float] = {}
        models: Dict[str, KMeansModel] = {}
        for model_id in MODEL_IDS:
            models[model_id], load_times[model_id] = load_model(model_id, project_root)

        track_a_warmup = []
        for model_id in MODEL_IDS:
            warm = run_assignment_trial(model_id, cached, models[model_id])
            track_a_warmup.append({"model_id": model_id, "pass": warm["pass"], "distribution": warm["distribution"]})
        track_a_order = [["B0", "M1", "M2"], ["M1", "M2", "B0"], ["M2", "B0", "M1"]]
        track_a_records: List[Dict[str, Any]] = []
        track_a_by_model = {model_id: [] for model_id in MODEL_IDS}
        for round_index, order in enumerate(track_a_order, 1):
            for order_index, model_id in enumerate(order, 1):
                record = run_assignment_trial(model_id, cached, models[model_id])
                record.update({"track": "A", "round": round_index, "order_index": order_index})
                track_a_records.append(record)
                track_a_by_model[model_id].append(record)
        locked_mismatches = {model_id: prediction_mismatch(cached, model_id, models[model_id]) for model_id in MODEL_IDS}
        track_a_equality = {model_id: locked_mismatches[model_id] == 0 and all(record["pass"] for record in track_a_by_model[model_id]) for model_id in MODEL_IDS}
        if not all(track_a_equality.values()):
            raise ValueError(f"Track A locked prediction equality failed: {locked_mismatches}")
        track_a_summary: Dict[str, Any] = {}
        for model_id in MODEL_IDS:
            measured = stats([float(record["duration_sec"]) for record in track_a_by_model[model_id]])
            track_a_summary[model_id] = {"model_id": model_id, "model_name": MODEL_SPECS[model_id]["name"], "k": MODEL_SPECS[model_id]["k"], "dimension": MODEL_SPECS[model_id]["dimension"], "training_space": MODEL_SPECS[model_id]["space"], "initMode": MODEL_SPECS[model_id]["init"], **measured, "customers_per_second": COHORT_COUNT / measured["median_sec"], "model_load_sec": load_times[model_id], "prediction_mismatch_vs_locked": locked_mismatches[model_id], "assignment_only": True}
        write_json(output_dir / "track_a_assignment_trials.json", {"warmup_count": 1, "measured_count": 3, "warmup": track_a_warmup, "measured_order": track_a_order, "trials": track_a_records, "model_load_sec": load_times, "locked_prediction_equality": track_a_equality})
        write_csv(output_dir / "track_a_assignment_summary.csv", ["ModelID", "ModelName", "K", "Dimension", "TrainingSpace", "InitMode", "MedianAssignmentSec", "MeanAssignmentSec", "PopulationStdAssignmentSec", "CustomersPerSecond", "ModelLoadSec", "PredictionMismatchVsLocked"], [{"ModelID": model_id, "ModelName": value["model_name"], "K": value["k"], "Dimension": value["dimension"], "TrainingSpace": value["training_space"], "InitMode": value["initMode"], "MedianAssignmentSec": value["median_sec"], "MeanAssignmentSec": value["mean_sec"], "PopulationStdAssignmentSec": value["population_std_sec"], "CustomersPerSecond": value["customers_per_second"], "ModelLoadSec": value["model_load_sec"], "PredictionMismatchVsLocked": value["prediction_mismatch_vs_locked"]} for model_id, value in track_a_summary.items()])

        track_b_warmup = []
        for model_id in MODEL_IDS:
            warm = run_build_trial(model_id, cached)
            track_b_warmup.append({"model_id": model_id, "pass": warm["pass"], "numIter": warm["numIter"], "trainingCost": warm["trainingCost"], "cluster_sizes": warm["cluster_sizes"]})
        track_b_order = [["B0", "M1", "M2"], ["M1", "M2", "B0"], ["M2", "B0", "M1"]]
        track_b_records: List[Dict[str, Any]] = []
        track_b_by_model = {model_id: [] for model_id in MODEL_IDS}
        for round_index, order in enumerate(track_b_order, 1):
            for order_index, model_id in enumerate(order, 1):
                record = run_build_trial(model_id, cached)
                record.update({"track": "B", "round": round_index, "order_index": order_index})
                track_b_records.append(record)
                track_b_by_model[model_id].append(record)
        determinism = {model_id: deterministic_validation(track_b_by_model, model_id) for model_id in MODEL_IDS}
        if not all(item["pass"] for item in determinism.values()):
            raise ValueError("Track B determinism validation failed under center-alignment tolerance.")
        track_b_summary: Dict[str, Any] = {}
        for model_id in MODEL_IDS:
            records = track_b_by_model[model_id]
            fit_stats = stats([float(record["fit_only_sec"]) for record in records])
            assign_stats = stats([float(record["assign_after_fit_sec"]) for record in records])
            total_stats = stats([float(record["total_fit_plus_assign_sec"]) for record in records])
            track_b_summary[model_id] = {"model_id": model_id, "model_name": MODEL_SPECS[model_id]["name"], "k": MODEL_SPECS[model_id]["k"], "dimension": MODEL_SPECS[model_id]["dimension"], "training_space": MODEL_SPECS[model_id]["space"], "initMode": MODEL_SPECS[model_id]["init"], "fit_only": fit_stats, "assign_after_fit": assign_stats, "total_fit_plus_assign": total_stats, "median_fit_only_sec": fit_stats["median_sec"], "median_assign_after_fit_sec": assign_stats["median_sec"], "median_total_fit_plus_assign_sec": total_stats["median_sec"], "customers_per_second_total": COHORT_COUNT / total_stats["median_sec"], "median_numIter": statistics.median([int(record["numIter"]) for record in records]), "median_trainingCost": statistics.median([float(record["trainingCost"]) for record in records]), "representative_centers": records[0]["centers"], "determinism_pass": determinism[model_id]["pass"]}
        write_json(output_dir / "track_b_build_assign_trials.json", {"warmup_count": 1, "measured_count": 3, "warmup": track_b_warmup, "measured_order": track_b_order, "trials": track_b_records, "final_models_saved": False, "final_model_paths_modified": False})
        write_csv(output_dir / "track_b_build_assign_summary.csv", ["ModelID", "ModelName", "K", "Dimension", "TrainingSpace", "InitMode", "MedianFitOnlySec", "MedianAssignAfterFitSec", "MedianTotalFitPlusAssignSec", "PopulationStdFitSec", "PopulationStdAssignSec", "PopulationStdTotalSec", "CustomersPerSecondTotal", "MedianNumIter", "MedianTrainingCost"], [{"ModelID": model_id, "ModelName": value["model_name"], "K": value["k"], "Dimension": value["dimension"], "TrainingSpace": value["training_space"], "InitMode": value["initMode"], "MedianFitOnlySec": value["median_fit_only_sec"], "MedianAssignAfterFitSec": value["median_assign_after_fit_sec"], "MedianTotalFitPlusAssignSec": value["median_total_fit_plus_assign_sec"], "PopulationStdFitSec": value["fit_only"]["population_std_sec"], "PopulationStdAssignSec": value["assign_after_fit"]["population_std_sec"], "PopulationStdTotalSec": value["total_fit_plus_assign"]["population_std_sec"], "CustomersPerSecondTotal": value["customers_per_second_total"], "MedianNumIter": value["median_numIter"], "MedianTrainingCost": value["median_trainingCost"]} for model_id, value in track_b_summary.items()])
        write_json(output_dir / "benchmark_determinism_validation.json", {"tolerance": TOLERANCE, "models": determinism, "pass": all(item["pass"] for item in determinism.values())})

        runtime_ratios = {"track_a_assignment": pair_ratios({model_id: {"median_runtime_sec": track_a_summary[model_id]["median_sec"]} for model_id in MODEL_IDS}, "median_runtime_sec"), "track_b_fit_only": pair_ratios({model_id: {"median_runtime_sec": track_b_summary[model_id]["median_fit_only_sec"]} for model_id in MODEL_IDS}, "median_runtime_sec"), "track_b_assign_after_fit": pair_ratios({model_id: {"median_runtime_sec": track_b_summary[model_id]["median_assign_after_fit_sec"]} for model_id in MODEL_IDS}, "median_runtime_sec"), "track_b_total_fit_plus_assign": pair_ratios({model_id: {"median_runtime_sec": track_b_summary[model_id]["median_total_fit_plus_assign_sec"]} for model_id in MODEL_IDS}, "median_runtime_sec")}
        write_json(output_dir / "runtime_ratios.json", runtime_ratios)

        quality_context = stage22_quality_context(authority_context)
        b0_m1_fit_b0 = track_b_summary["B0"]["median_fit_only_sec"]
        b0_m1_fit_m1 = track_b_summary["M1"]["median_fit_only_sec"]
        b0_m1_analysis = {"runtime_basis": "Track B median of 3 measured fresh-fit trials", "fit_runtime_sec": {"B0": b0_m1_fit_b0, "M1": b0_m1_fit_m1}, "fit_speedup": {"definition": "B0 median fit runtime / M1 median fit runtime", "B0_divided_by_M1": b0_m1_fit_b0 / b0_m1_fit_m1, "M1_speedup_factor_vs_B0": b0_m1_fit_b0 / b0_m1_fit_m1, "M1_runtime_reduction_pct_vs_B0": (b0_m1_fit_b0 - b0_m1_fit_m1) / b0_m1_fit_b0 * 100.0}, "iteration_reduction": {"B0_median_numIter": track_b_summary["B0"]["median_numIter"], "M1_median_numIter": track_b_summary["M1"]["median_numIter"], "absolute_reduction_B0_minus_M1": track_b_summary["B0"]["median_numIter"] - track_b_summary["M1"]["median_numIter"], "percent_reduction_vs_B0": (track_b_summary["B0"]["median_numIter"] - track_b_summary["M1"]["median_numIter"]) / track_b_summary["B0"]["median_numIter"] * 100.0}, "assignment_runtime_sec": {"B0": track_b_summary["B0"]["median_assign_after_fit_sec"], "M1": track_b_summary["M1"]["median_assign_after_fit_sec"], "B0_divided_by_M1": track_b_summary["B0"]["median_assign_after_fit_sec"] / track_b_summary["M1"]["median_assign_after_fit_sec"]}, "total_runtime_sec": {"B0": track_b_summary["B0"]["median_total_fit_plus_assign_sec"], "M1": track_b_summary["M1"]["median_total_fit_plus_assign_sec"], "B0_divided_by_M1": track_b_summary["B0"]["median_total_fit_plus_assign_sec"] / track_b_summary["M1"]["median_total_fit_plus_assign_sec"]}, "quality_context_from_stage22": quality_context["b0_m1"], "no_quality_score_used": True}
        write_json(output_dir / "b0_m1_initialization_runtime_analysis.json", b0_m1_analysis)

        m1_m2_fit_m1 = track_b_summary["M1"]["median_fit_only_sec"]
        m1_m2_fit_m2 = track_b_summary["M2"]["median_fit_only_sec"]
        m1_m2_assign_m1 = track_b_summary["M1"]["median_assign_after_fit_sec"]
        m1_m2_assign_m2 = track_b_summary["M2"]["median_assign_after_fit_sec"]
        m1_m2_total_m1 = track_b_summary["M1"]["median_total_fit_plus_assign_sec"]
        m1_m2_total_m2 = track_b_summary["M2"]["median_total_fit_plus_assign_sec"]
        m1_m2_tradeoff = {"runtime_basis": "Track B median of 3 measured fresh-fit trials", "feature_dimension": {"M1": 3, "M2": 5}, "cluster_count": {"M1": 4, "M2": 5}, "fit": {"M1_sec": m1_m2_fit_m1, "M2_sec": m1_m2_fit_m2, "M2_divided_by_M1": m1_m2_fit_m2 / m1_m2_fit_m1, "M2_overhead_pct_vs_M1": (m1_m2_fit_m2 / m1_m2_fit_m1 - 1.0) * 100.0}, "assignment_after_fit": {"M1_sec": m1_m2_assign_m1, "M2_sec": m1_m2_assign_m2, "M2_divided_by_M1": m1_m2_assign_m2 / m1_m2_assign_m1, "M2_overhead_pct_vs_M1": (m1_m2_assign_m2 / m1_m2_assign_m1 - 1.0) * 100.0}, "total_fit_plus_assign": {"M1_sec": m1_m2_total_m1, "M2_sec": m1_m2_total_m2, "M2_divided_by_M1": m1_m2_total_m2 / m1_m2_total_m1, "M2_overhead_pct_vs_M1": (m1_m2_total_m2 / m1_m2_total_m1 - 1.0) * 100.0}, "quality_context_from_stage22": quality_context["m1_m2"], "no_weighted_runtime_quality_score_used": True}
        write_json(output_dir / "m1_m2_runtime_tradeoff.json", m1_m2_tradeoff)

        application_id = spark.sparkContext.applicationId
        ui_port = spark.conf.get("spark.ui.port", "4040")
        environment: Dict[str, Any] = {"stage": STAGE, "authority": AUTHORITY, "cohort_count": COHORT_COUNT, "cohort_fingerprint": COHORT_FINGERPRINT, "spark_version": spark.version, "application_id": application_id, "application_name": "Stage23-Primary-B0-M1-M2-Runtime-Benchmark", "master": "local[4]", "shuffle_partitions": 64, "adaptive_enabled": True, "python_version": platform.python_version(), "java_version": spark._jvm.java.lang.System.getProperty("java.version"), "os": platform.platform(), "cpu_logical_processors": os.cpu_count(), "system_memory_bytes": None, "cache_policy": "MEMORY_AND_DISK", "timing_api": "time.perf_counter", "warmup_trials": 1, "measured_trials": 3, "primary_runtime_statistic": "median", "input_read_runtime_sec": input_read_runtime, "cache_materialization_runtime_sec": cache_materialization_runtime, "cached_row_count": cached_count, "model_load_sec": load_times, "event_log_enabled": event_logging_enabled, "event_log_path": event_uri, "event_log_issue": event_logging_issue, "warnings": warnings}
        try:
            import psutil
            environment["system_memory_bytes"] = int(psutil.virtual_memory().total)
        except Exception as memory_error:
            environment["system_memory_unavailable_reason"] = f"{type(memory_error).__name__}: {memory_error}"
        write_json(output_dir / "benchmark_environment.json", environment)

        spark_manifest = {"spark_application_id": application_id, "spark_application_name": environment["application_name"], "spark_ui_url": f"http://localhost:{ui_port}", "spark_ui_port": ui_port, "spark_version": spark.version, "master": "local[4]", "event_log_enabled": event_logging_enabled, "event_log_path": event_uri, "event_log_directory": str(event_dir), "event_log_issue": event_logging_issue, "screenshots_fabricated": False, "evidence_scope": "runtime configuration and application identity; no fake screenshots", "benchmark_operations": ["Track A locked B0 assignment", "Track A locked M1 assignment", "Track A locked M2 assignment", "Track B fresh B0 random K-Means fit plus assignment", "Track B fresh M1 k-means|| K-Means fit plus assignment", "Track B fresh M2 k-means|| K-Means fit plus assignment"]}
        write_json(output_dir / "spark_ui_evidence_manifest.json", spark_manifest)

        summary: Dict[str, Any] = {"stage": STAGE, "authority": AUTHORITY, "quality_authority": QUALITY_AUTHORITY, "cohort_authority": COHORT_AUTHORITY, "cohort_count": COHORT_COUNT, "cohort_customer_id_sha256": COHORT_FINGERPRINT, "authority_audit": authority_context["checks"], "input_validation_pass": validation["pass"], "cache_materialization": {"storage_level": "MEMORY_AND_DISK", "row_count": cached_count, "input_read_runtime_sec": input_read_runtime, "cache_materialization_runtime_sec": cache_materialization_runtime}, "track_a_assignment": track_a_summary, "track_b_build_assign": track_b_summary, "track_a_locked_prediction_equality": track_a_equality, "track_b_determinism_pass": all(item["pass"] for item in determinism.values()), "b0_m1_initialization_analysis": b0_m1_analysis, "m1_m2_runtime_tradeoff": m1_m2_tradeoff, "runtime_ratios": runtime_ratios, "spark_application_id": application_id, "spark_event_log": {"enabled": event_logging_enabled, "issue": event_logging_issue, "path": event_uri}, "spark_ui_evidence": spark_manifest, "quality_models_modified": False, "final_model_authorities_modified": False, "runtime_quality_combined_score_used": False, "stage20_auxiliary_preserved": True, "primary_runtime_benchmark_complete": True, "ready_for_stage24": True, "warnings": warnings, "blockers": []}
        write_json(summary_path, summary)
        if cached is not None:
            cached.unpersist()
            cached = None
        spark.stop()
        spark = None

        artifact_paths = [output_dir / name for name in ["benchmark_environment.json", "benchmark_input_validation.json", "track_a_assignment_trials.json", "track_a_assignment_summary.csv", "track_b_build_assign_trials.json", "track_b_build_assign_summary.csv", "benchmark_determinism_validation.json", "runtime_ratios.json", "b0_m1_initialization_runtime_analysis.json", "m1_m2_runtime_tradeoff.json", "spark_ui_evidence_manifest.json", "stage23_final_summary.json"]] + [report_path, log_path]
        report_lines = ["STAGE 23 — CONTROLLED PRIMARY B0/M1/M2 RUNTIME BENCHMARK", "", f"AUTHORITY: {AUTHORITY}", f"QUALITY_AUTHORITY: {QUALITY_AUTHORITY}", f"COHORT: {COHORT_AUTHORITY} / {COHORT_COUNT} / {COHORT_FINGERPRINT}", f"SPARK_APPLICATION_ID: {application_id}", f"EVENT_LOG_ENABLED: {event_logging_enabled}", f"EVENT_LOG_ISSUE: {event_logging_issue}", "", "TRACK A SUMMARY:", json.dumps(track_a_summary, ensure_ascii=False, indent=2), "", "TRACK B SUMMARY:", json.dumps(track_b_summary, ensure_ascii=False, indent=2), "", "B0/M1 INITIALIZATION ANALYSIS:", json.dumps(b0_m1_analysis, ensure_ascii=False, indent=2), "", "M1/M2 RUNTIME TRADEOFF:", json.dumps(m1_m2_tradeoff, ensure_ascii=False, indent=2), "", "RUNTIME RATIOS:", json.dumps(runtime_ratios, ensure_ascii=False, indent=2), "", f"TRACK_A_LOCKED_PREDICTION_EQUALITY: {all(track_a_equality.values())}", f"TRACK_B_DETERMINISM_PASS: {all(item['pass'] for item in determinism.values())}", "QUALITY_MODELS_MODIFIED: False", "FINAL_MODEL_AUTHORITIES_MODIFIED: False", "RUNTIME_QUALITY_COMBINED_SCORE_USED: False", "STAGE20_AUXILIARY_PRESERVED: True", "", "PRIMARY_RUNTIME_BENCHMARK_COMPLETE: True", "READY_FOR_STAGE_24: True"]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        readback = readback_artifacts(artifact_paths, project_root)
        summary["artifact_readback"] = readback
        summary["artifact_readback_pass"] = all(readback.values())
        summary["primary_runtime_benchmark_complete"] = bool(summary["primary_runtime_benchmark_complete"] and summary["artifact_readback_pass"])
        summary["ready_for_stage24"] = bool(summary["primary_runtime_benchmark_complete"])
        if not summary["artifact_readback_pass"]:
            summary["blockers"] = ["One or more Stage 23 artifacts failed read-back."]
        write_json(summary_path, summary)
        print("AUTHORITY_AUDIT_PASS=True")
        print("INPUT_VALIDATION_PASS=True")
        print("TRACK_A_LOCKED_PREDICTION_EQUALITY_PASS=True")
        print("TRACK_B_DETERMINISM_PASS=True")
        print(f"SPARK_APPLICATION_ID={application_id}")
        print(f"PRIMARY_RUNTIME_BENCHMARK_COMPLETE={summary['primary_runtime_benchmark_complete']}")
        print(f"READY_FOR_STAGE_24={summary['ready_for_stage24']}")
        return 0 if summary["ready_for_stage24"] else 3
    except Exception as error:
        blockers.append(f"{type(error).__name__}: {error}")
        failed = {"stage": STAGE, "authority": AUTHORITY, "quality_authority": QUALITY_AUTHORITY, "primary_runtime_benchmark_complete": False, "ready_for_stage24": False, "warnings": warnings, "blockers": blockers}
        write_json(summary_path, failed)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("STAGE 23 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_24: False\n", encoding="utf-8")
        print(f"STAGE 23 FAILED CLOSED: {error}")
        return 3
    finally:
        if cached is not None:
            try:
                cached.unpersist()
            except Exception:
                pass
        if spark is not None:
            try:
                spark.stop()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
