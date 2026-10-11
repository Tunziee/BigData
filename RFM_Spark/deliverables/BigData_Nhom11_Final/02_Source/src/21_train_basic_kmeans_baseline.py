"""Stage 21: build and lock the Basic K-Means baseline B0.

B0 uses the full Stage 7 standardized RFM3D population and differs from the
locked Main K-Means model M1 only by initialization: random versus k-means||.
Only seed=42 is the B0 authority; other seeds are supplementary diagnostics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import avg, col, count, element_at, max as spark_max, min as spark_min, sum as spark_sum
from pyspark.storagelevel import StorageLevel


FULL_COUNT = 1_362_281
COHORT_COUNT = 1_000_000
COHORT_FINGERPRINT = "ca082ec1d828df9c34a10105ae11599c3cc0653dc5b4d2c11af18ec82b781cf5"
K = 4
PRIMARY_SEED = 42
SENSITIVITY_SEEDS = [7, 21, 42, 84, 123]
FEATURE_COL = "features"
PREDICTION_COL = "prediction"
TOLERANCE = 1e-8
RFM_DIMENSION = 3
RFM_FEATURE_ORDER = ["Recency", "Frequency", "Monetary"]


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
    stage7 = read_json(project_root / "output" / "analysis" / "stage7" / "main_rfm_pipeline_summary.json")
    stage10 = read_json(project_root / "output" / "analysis" / "stage10" / "final_model_summary.json")
    stage20 = read_json(project_root / "output" / "benchmark" / "stage20" / "stage20_final_summary.json")
    stage7_ok = stage7.get("stage") == 7 and stage7.get("ready_for_stage8") is True and stage7.get("features") == RFM_FEATURE_ORDER and stage7.get("output", {}).get("row_count") == FULL_COUNT and stage7.get("output", {}).get("distinct_customers") == FULL_COUNT and stage7.get("output", {}).get("vector_dimension") == RFM_DIMENSION and stage7.get("validation", {}).get("readback_pass") is True
    config = stage10.get("configuration", {})
    stage10_ok = stage10.get("stage") == 10 and stage10.get("authority") == "MAIN_KMEANS_FINAL" and stage10.get("selected_k") == K and config == {"k": K, "seed": 42, "featuresCol": "features", "predictionCol": "prediction", "distanceMeasure": "euclidean", "initMode": "k-means||", "maxIter": 100, "tol": 1e-4} and stage10.get("input_validation", {}).get("row_count") == FULL_COUNT and stage10.get("input_validation", {}).get("distinct_customer_id") == FULL_COUNT and stage10.get("input_validation", {}).get("pass") is True
    stage20_ok = stage20.get("stage") == 20 and stage20.get("authority") == "THREE_MODEL_RUNTIME_BENCHMARK_1M_V1" and stage20.get("runtime_benchmark_complete") is True and stage20.get("ready_for_stage21") is True
    if not stage7_ok:
        blockers.append("Stage 7 standardized RFM3D authority or full-population gate mismatch.")
    if not stage10_ok:
        blockers.append("Stage 10 Main K-Means authority/configuration/full-population gate mismatch.")
    if not stage20_ok:
        blockers.append("Stage 20 auxiliary benchmark authority or completion gate mismatch.")
    return blockers, {"stage7": stage7, "stage10": stage10, "stage20": stage20}


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
    vector_quality = validate_vector(df, FEATURE_COL, RFM_DIMENSION)
    return {"row_count": row_count, "distinct_customer_id": distinct_count, "duplicate_customer_id": row_count - distinct_count, "feature_quality": vector_quality, "pass": row_count == FULL_COUNT and distinct_count == FULL_COUNT and row_count == distinct_count and vector_quality["pass"]}


def cluster_distribution(df: DataFrame, label_col: str = PREDICTION_COL) -> Dict[str, Any]:
    rows = df.groupBy(label_col).count().orderBy(label_col).collect()
    ids = [int(row[label_col]) for row in rows]
    counts = [int(row["count"]) for row in rows]
    ratios = [value / sum(counts) for value in counts] if counts else []
    return {"cluster_sizes": [{"cluster_id": ids[index], "count": counts[index], "ratio": ratios[index], "ratio_pct": ratios[index] * 100.0} for index in range(len(ids))], "cluster_ids": ids, "counts": counts, "sum": sum(counts), "cluster_count": len(ids), "pass": ids == list(range(K)) and len(counts) == K and sum(counts) == FULL_COUNT}


def distribution_for_count(df: DataFrame, label_col: str, expected_total: int = COHORT_COUNT) -> Dict[str, Any]:
    rows = df.groupBy(label_col).count().orderBy(label_col).collect()
    ids = [int(row[label_col]) for row in rows]
    counts = [int(row["count"]) for row in rows]
    ratios = [value / expected_total for value in counts]
    mean = sum(counts) / K
    std = math.sqrt(sum((value - mean) ** 2 for value in counts) / K)
    entropy = -sum(ratio * math.log(ratio) for ratio in ratios if ratio > 0)
    return {"cluster_sizes": [{"cluster_id": ids[index], "count": counts[index], "ratio": ratios[index], "ratio_pct": ratios[index] * 100.0} for index in range(len(ids))], "cluster_ids": ids, "counts": counts, "sum": sum(counts), "cluster_count": len(ids), "min_ratio_pct": min(ratios) * 100.0, "max_ratio_pct": max(ratios) * 100.0, "cluster_size_cv": std / mean, "entropy_natural_log": entropy, "normalized_entropy": entropy / math.log(K), "pass": ids == list(range(K)) and len(counts) == K and sum(counts) == expected_total}


def model_config(seed: int, init_mode: str, prediction_col: str = PREDICTION_COL) -> Dict[str, Any]:
    return {"k": K, "seed": seed, "featuresCol": FEATURE_COL, "predictionCol": prediction_col, "distanceMeasure": "euclidean", "initMode": init_mode, "maxIter": 100, "tol": 1e-4}


def fit_model(df: DataFrame, seed: int, init_mode: str = "random", prediction_col: str = PREDICTION_COL) -> Tuple[Any, float]:
    estimator = KMeans(k=K, seed=seed, featuresCol=FEATURE_COL, predictionCol=prediction_col, distanceMeasure="euclidean", initMode=init_mode, maxIter=100, tol=1e-4)
    start = time.perf_counter()
    model = estimator.fit(df)
    return model, time.perf_counter() - start


def model_centers(model: Any) -> List[List[float]]:
    return [[float(value) for value in center] for center in model.clusterCenters()]


def silhouette(df: DataFrame, prediction_col: str = PREDICTION_COL) -> float:
    evaluator = ClusteringEvaluator(featuresCol=FEATURE_COL, predictionCol=prediction_col, metricName="silhouette", distanceMeasure="squaredEuclidean")
    return float(evaluator.evaluate(df.select(FEATURE_COL, prediction_col)))


def sse_from_assignments(df: DataFrame, prediction_col: str, expected_total: int) -> Dict[str, Any]:
    values = vector_to_array(col(FEATURE_COL))
    components = df.select(prediction_col, *[element_at(values, index).alias(f"v{index - 1}") for index in range(1, RFM_DIMENSION + 1)])
    names = [f"v{index}" for index in range(RFM_DIMENSION)]
    centers_df = components.groupBy(prediction_col).agg(*[avg(col(name)).alias(f"c_{name}") for name in names])
    center_rows = centers_df.orderBy(prediction_col).collect()
    centers = {str(int(row[prediction_col])): [float(row[f"c_{name}"]) for name in names] for row in center_rows}
    joined = components.join(centers_df, prediction_col, "inner")
    squared = sum((col(name) - col(f"c_{name}")) * (col(name) - col(f"c_{name}")) for name in names)
    sse = float(joined.select(spark_sum(squared).alias("sse")).first()["sse"] or 0.0)
    return {"sse": sse, "sse_per_customer": sse / expected_total, "centroids": centers, "customer_count": expected_total, "dimension": RFM_DIMENSION}


def stats(values: List[float]) -> Dict[str, Any]:
    return {"min": min(values), "max": max(values), "mean": statistics.mean(values), "median": statistics.median(values), "std": statistics.pstdev(values)}


def canonical_prediction_fingerprint(df: DataFrame, prediction_col: str) -> str:
    digest = hashlib.sha256()
    for row in df.select("customer_id", prediction_col).orderBy("customer_id").toLocalIterator():
        digest.update(str(row["customer_id"]).encode("utf-8"))
        digest.update(b"|")
        digest.update(str(int(row[prediction_col])).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def canonical_id_fingerprint(df: DataFrame) -> str:
    digest = hashlib.sha256()
    for row in df.select("customer_id").orderBy("customer_id").toLocalIterator():
        digest.update(str(row["customer_id"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def compare_centers(left: List[List[float]], right: List[List[float]]) -> Dict[str, Any]:
    differences = [abs(a - b) for left_row, right_row in zip(left, right) for a, b in zip(left_row, right_row)]
    return {"max_absolute_difference": max(differences or [0.0]), "tolerance": TOLERANCE, "pass": len(left) == len(right) == K and all(len(row) == RFM_DIMENSION for row in left) and all(len(row) == RFM_DIMENSION for row in right) and max(differences or [0.0]) <= TOLERANCE}


def path_nonempty(path: Path) -> bool:
    return path.exists() and (not path.is_dir() or any(path.iterdir()))


def readback_artifacts(paths: List[Path], project_root: Path) -> Dict[str, bool]:
    result: Dict[str, bool] = {}
    for path in paths:
        ok = path.exists() and (any(path.iterdir()) if path.is_dir() else path.stat().st_size > 0)
        if ok and path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                ok = False
        result[str(path.relative_to(project_root))] = ok
    return result


def safe_unpersist(df: Any) -> None:
    if df is not None:
        try:
            df.unpersist(blocking=False)
        except Exception:
            pass


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "output" / "models" / "basic_kmeans_baseline"
    prediction_path = project_root / "output" / "final" / "basic_kmeans_baseline_predictions"
    comparison_dir = project_root / "output" / "comparison" / "stage21"
    analysis_dir = project_root / "output" / "analysis" / "stage21"
    report_path = project_root / "reports" / "stage21_basic_kmeans_baseline.txt"
    summary_path = analysis_dir / "stage21_final_summary.json"
    blockers: List[str] = []
    warnings: List[str] = ["B0 is a technical baseline; no assumption is made that random initialization must underperform M1.", "Frequency means distinct purchase dates, not invoices or order IDs."]
    spark = None
    cached = None
    primary_predictions = None
    saved_full = None
    repro_predictions = None
    cohort_ids = None
    cohort_eval = None
    try:
        blockers, authorities = audit_authorities(project_root)
        if any(path_nonempty(path) for path in [model_path, prediction_path, comparison_dir, analysis_dir]):
            blockers.append("Existing non-empty Stage 21 output detected; replacement is not authorized by this run.")
        if blockers:
            failed = {"stage": 21, "authority": "BASIC_KMEANS_BASELINE_V1", "ready_for_stage22": False, "blockers": blockers}
            write_json(summary_path, failed)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("STAGE 21 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_22: False\n", encoding="utf-8")
            return 3

        stage7 = authorities["stage7"]
        stage10 = authorities["stage10"]
        stage20 = authorities["stage20"]
        analysis_dir.mkdir(parents=True, exist_ok=True)
        comparison_dir.mkdir(parents=True, exist_ok=True)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        source_path = project_root / "output" / "ml" / "main_rfm_scaled"
        cohort_ids_path = project_root / "output" / "comparison" / "stage17" / "comparison_customer_ids_1m"
        spark = SparkSession.builder.master("local[4]").appName("Stage21-Basic-KMeans-Baseline").config("spark.sql.shuffle.partitions", "64").config("spark.sql.adaptive.enabled", "true").config("spark.local.dir", str(project_root.parent / "spark_tmp")).getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        enable_windows_java_filesystem_fallback(spark)

        input_df = spark.read.parquet(str(source_path)).select("customer_id", FEATURE_COL)
        cache_start = time.perf_counter()
        cached = input_df.persist(StorageLevel.MEMORY_AND_DISK)
        cached_count = cached.count()
        cache_runtime = time.perf_counter() - cache_start
        validation = input_validation(cached)
        validation["cache_materialization"] = {"storage_level": "MEMORY_AND_DISK", "materialized_row_count": cached_count, "materialization_runtime_sec": cache_runtime, "materialized_before_timed_fit": True}
        write_json(analysis_dir / "b0_configuration.json", {"stage": 21, "authority": "BASIC_KMEANS_BASELINE_V1", "model_id": "B0", "model_name": "Basic K-Means Baseline", "model_role": "technical_baseline", "feature_authority": "Stage7 standardized RFM3D", "feature_order": RFM_FEATURE_ORDER, "training_customer_count": FULL_COUNT, "k": K, "seed": PRIMARY_SEED, "featuresCol": FEATURE_COL, "predictionCol": PREDICTION_COL, "distanceMeasure": "euclidean", "initMode": "random", "maxIter": 100, "tol": 1e-4, "controlled_reference": "MAIN_KMEANS_FINAL", "controlled_difference": "initMode only"})
        write_json(analysis_dir / "b0_full_data_metrics.json", {"input_validation": validation})
        if not validation["pass"]:
            raise ValueError("Stage 7 B0 input validation failed.")

        primary_model, primary_fit_runtime = fit_model(cached, PRIMARY_SEED, "random")
        primary_predictions = primary_model.transform(cached).persist(StorageLevel.MEMORY_AND_DISK)
        primary_predictions.count()
        full_distribution = cluster_distribution(primary_predictions)
        if not full_distribution["pass"]:
            raise ValueError("Primary B0 did not produce exactly four full-data clusters.")
        primary_silhouette = silhouette(primary_predictions)
        primary_wcss = float(primary_model.summary.trainingCost)
        primary_num_iter = int(primary_model.summary.numIter)
        primary_centers = model_centers(primary_model)
        full_metrics = {"model_id": "B0", "model_name": "Basic K-Means Baseline", "training_population": FULL_COUNT, "fit_runtime_sec": primary_fit_runtime, "numIter": primary_num_iter, "trainingCost_WCSS": primary_wcss, "silhouette_rfm3d": primary_silhouette, "cluster_distribution": full_distribution, "centers_scaled": primary_centers, "configuration": model_config(PRIMARY_SEED, "random"), "metrics_valid": primary_fit_runtime > 0 and 1 <= primary_num_iter <= 100 and finite(primary_wcss) and primary_wcss >= 0 and finite(primary_silhouette) and -1 <= primary_silhouette <= 1 and len(primary_centers) == K and all(len(center) == RFM_DIMENSION and all(finite(value) for value in center) for center in primary_centers)}
        if not full_metrics["metrics_valid"]:
            raise ValueError("Primary B0 full-data metrics invalid.")
        write_json(analysis_dir / "b0_full_data_metrics.json", full_metrics | {"input_validation": validation, "cache_materialization": validation["cache_materialization"]})

        model_path_check = model_path
        primary_model.write().save(str(model_path_check))
        full_write = primary_predictions.select("customer_id", col("prediction").cast("int").alias("prediction"))
        full_write.write.mode("errorifexists").parquet(str(prediction_path))
        saved_full = spark.read.parquet(str(prediction_path)).select("customer_id", "prediction").persist(StorageLevel.MEMORY_AND_DISK)
        saved_full.count()
        loaded_model = KMeansModel.load(str(model_path))
        loaded_centers = model_centers(loaded_model)
        loaded_predictions = loaded_model.transform(cached)
        prediction_mismatch = loaded_predictions.select("customer_id", col("prediction").alias("loaded_prediction")).join(saved_full.select("customer_id", col("prediction").alias("saved_prediction")), "customer_id", "inner").where(col("loaded_prediction") != col("saved_prediction")).count()
        model_readback = {"load_pass": True, "k": loaded_model.getK(), "featuresCol": loaded_model.getFeaturesCol(), "predictionCol": loaded_model.getPredictionCol(), "centers": compare_centers(primary_centers, loaded_centers), "prediction_mismatch_count": prediction_mismatch, "pass": loaded_model.getK() == K and loaded_model.getFeaturesCol() == FEATURE_COL and prediction_mismatch == 0 and compare_centers(primary_centers, loaded_centers)["pass"]}
        saved_full_distribution = cluster_distribution(saved_full)
        prediction_readback = {"row_count": saved_full.count(), "distinct_customer_id": saved_full.select("customer_id").distinct().count(), "duplicate_customer_id": saved_full.count() - saved_full.select("customer_id").distinct().count(), "prediction_null": saved_full.where(col("prediction").isNull()).count(), "invalid_prediction": saved_full.where(~col("prediction").isin(list(range(K)))).count(), "distribution": saved_full_distribution, "pass": saved_full_distribution["pass"] and saved_full.count() == FULL_COUNT and saved_full.select("customer_id").distinct().count() == FULL_COUNT and saved_full.where(col("prediction").isNull()).count() == 0 and saved_full.where(~col("prediction").isin(list(range(K)))).count() == 0}
        if not model_readback["pass"] or not prediction_readback["pass"]:
            raise ValueError("Primary B0 model/prediction read-back failed.")
        safe_unpersist(saved_full)
        saved_full = None

        m1_metrics = {"model_id": "M1", "model_name": "Project Main K-Means", "authority": stage10.get("authority"), "training_population": FULL_COUNT, "initMode": stage10.get("configuration", {}).get("initMode"), "fit_runtime_sec": stage10.get("training", {}).get("fit_runtime_sec"), "numIter": stage10.get("training", {}).get("num_iter"), "trainingCost_WCSS": stage10.get("training", {}).get("wcss"), "silhouette_rfm3d": stage10.get("training", {}).get("silhouette"), "cluster_distribution": stage10.get("cluster_distribution"), "centers_scaled": stage10.get("centers_scaled")}
        controlled_check = {"initialization_difference_only": True, "b0_init": "random", "m1_init": "k-means||", "controlled_difference": "initMode random vs k-means||", "same_full_population": True, "same_feature_space": True, "same_k": True, "same_seed": True, "same_distance": True, "same_maxIter": True, "same_tol": True, "b0": {"silhouette_rfm3d": primary_silhouette, "wcss": primary_wcss, "numIter": primary_num_iter, "cluster_distribution": full_distribution}, "m1": {"silhouette_rfm3d": m1_metrics["silhouette_rfm3d"], "wcss": m1_metrics["trainingCost_WCSS"], "numIter": m1_metrics["numIter"], "cluster_distribution": m1_metrics["cluster_distribution"]}, "silhouette_delta_b0_minus_m1": primary_silhouette - float(m1_metrics["silhouette_rfm3d"]), "wcss_delta_b0_minus_m1": primary_wcss - float(m1_metrics["trainingCost_WCSS"]), "interpretation": "This experiment measures whether initialization changes convergence and final clustering quality in fixed standardized RFM3D K=4; it does not select the final B0/M1/M2 winner."}
        write_json(analysis_dir / "b0_vs_m1_controlled_check.json", controlled_check)

        sensitivity_records = []
        for seed in SENSITIVITY_SEEDS:
            model, fit_runtime = fit_model(cached, seed, "random")
            assigned = model.transform(cached)
            dist = cluster_distribution(assigned)
            sil = silhouette(assigned)
            sensitivity_records.append({"seed": seed, "fit_runtime_sec": fit_runtime, "numIter": int(model.summary.numIter), "silhouette_rfm3d": sil, "trainingCost_WCSS": float(model.summary.trainingCost), "cluster_distribution": dist, "min_cluster_ratio_pct": min(item["ratio_pct"] for item in dist["cluster_sizes"]), "max_cluster_ratio_pct": max(item["ratio_pct"] for item in dist["cluster_sizes"]), "centers_scaled": model_centers(model), "authority": "diagnostic_only_not_saved"})
        sensitivity = {"seed_list": SENSITIVITY_SEEDS, "primary_seed": PRIMARY_SEED, "records": sensitivity_records, "summary": {metric: stats([float(record[metric]) for record in sensitivity_records]) for metric in ["silhouette_rfm3d", "trainingCost_WCSS", "numIter", "fit_runtime_sec"]}, "primary_seed_remains_42": True, "best_seed_not_selected_as_primary": True}
        write_json(analysis_dir / "random_seed_sensitivity.json", sensitivity)

        repro_model, repro_fit_runtime = fit_model(cached, PRIMARY_SEED, "random")
        repro_predictions = repro_model.transform(cached).persist(StorageLevel.MEMORY_AND_DISK)
        repro_predictions.count()
        repro_sil = silhouette(repro_predictions)
        repro_wcss = float(repro_model.summary.trainingCost)
        repro_sizes = cluster_distribution(repro_predictions)
        repro_centers = model_centers(repro_model)
        repro_differences = {"silhouette": abs(primary_silhouette - repro_sil), "wcss": abs(primary_wcss - repro_wcss), "numIter_equal": primary_num_iter == int(repro_model.summary.numIter), "cluster_sizes_equal": full_distribution["counts"] == repro_sizes["counts"], "centers": compare_centers(primary_centers, repro_centers)}
        repro_pass = repro_differences["silhouette"] <= TOLERANCE and repro_differences["wcss"] <= TOLERANCE and repro_differences["numIter_equal"] and repro_differences["cluster_sizes_equal"] and repro_differences["centers"]["pass"]
        b0_repro = {"primary_fit_runtime_sec": primary_fit_runtime, "independent_fit_runtime_sec": repro_fit_runtime, "tolerance": TOLERANCE, "primary": {"silhouette": primary_silhouette, "wcss": primary_wcss, "numIter": primary_num_iter, "cluster_sizes": full_distribution["counts"], "centers": primary_centers}, "independent": {"silhouette": repro_sil, "wcss": repro_wcss, "numIter": int(repro_model.summary.numIter), "cluster_sizes": repro_sizes["counts"], "centers": repro_centers}, "differences": repro_differences, "pass": repro_pass}
        write_json(analysis_dir / "b0_reproducibility.json", b0_repro)
        safe_unpersist(repro_predictions)
        repro_predictions = None
        if not repro_pass:
            raise ValueError("Primary B0 seed=42 reproducibility failed.")

        cohort_ids = spark.read.parquet(str(cohort_ids_path)).select("customer_id").persist(StorageLevel.MEMORY_AND_DISK)
        cohort_ids.count()
        cohort_id_count = cohort_ids.count()
        cohort_id_distinct = cohort_ids.select("customer_id").distinct().count()
        cohort_fingerprint = canonical_id_fingerprint(cohort_ids)
        if cohort_id_count != COHORT_COUNT or cohort_id_distinct != COHORT_COUNT or cohort_fingerprint != COHORT_FINGERPRINT:
            raise ValueError("Stage 17 comparison cohort ID authority mismatch.")
        safe_unpersist(primary_predictions)
        primary_predictions = None
        saved_full_for_projection = spark.read.parquet(str(prediction_path)).select("customer_id", col("prediction").cast("int").alias("b0_prediction"))
        cohort_predictions = saved_full_for_projection.join(cohort_ids, "customer_id", "inner").select("customer_id", "b0_prediction")
        cohort_predictions.count()
        cohort_predictions.write.mode("errorifexists").parquet(str(comparison_dir / "basic_kmeans_predictions_1m"))
        cohort_readback = spark.read.parquet(str(comparison_dir / "basic_kmeans_predictions_1m")).select("customer_id", "b0_prediction")
        cohort_readback.count()
        cohort_distribution = distribution_for_count(cohort_readback, "b0_prediction")
        cohort_fp = canonical_prediction_fingerprint(cohort_readback, "b0_prediction")
        universe_check = {"stage17_ids_minus_b0_predictions": cohort_ids.join(cohort_readback, "customer_id", "left_anti").count(), "b0_predictions_minus_stage17_ids": cohort_readback.join(cohort_ids, "customer_id", "left_anti").count()}
        universe_check["pass"] = universe_check["stage17_ids_minus_b0_predictions"] == 0 and universe_check["b0_predictions_minus_stage17_ids"] == 0
        if not universe_check["pass"]:
            raise ValueError("B0 1M customer universe mismatch.")
        cohort_eval = cached.join(cohort_ids, "customer_id", "inner").join(saved_full_for_projection.select("customer_id", col("b0_prediction").alias(PREDICTION_COL)), "customer_id", "inner").persist(StorageLevel.MEMORY_AND_DISK)
        cohort_eval.count()
        cohort_sil = silhouette(cohort_eval)
        cohort_sse = sse_from_assignments(cohort_eval, PREDICTION_COL, COHORT_COUNT)
        b0_1m_metrics = {"cohort_authority": "DETERMINISTIC_1M_COMPARISON_COHORT", "cohort_count": COHORT_COUNT, "cohort_customer_id_sha256": COHORT_FINGERPRINT, "cluster_distribution": cohort_distribution, "silhouette_rfm3d": cohort_sil, "sse_rfm3d": cohort_sse["sse"], "sse_per_customer_rfm3d": cohort_sse["sse_per_customer"], "centroids_from_1m_assignments": cohort_sse["centroids"], "metrics_valid": cohort_distribution["pass"] and finite(cohort_sil) and -1 <= cohort_sil <= 1 and finite(cohort_sse["sse"]) and cohort_sse["sse"] >= 0}
        write_json(analysis_dir / "b0_1m_metrics.json", b0_1m_metrics)
        safe_unpersist(cohort_eval)
        safe_unpersist(cohort_ids)
        cohort_eval = None
        cohort_ids = None
        manifest = {"authority": "BASIC_KMEANS_BASELINE_V1", "model_id": "B0", "full_prediction_count": FULL_COUNT, "comparison_cohort_count": COHORT_COUNT, "cohort_fingerprint": COHORT_FINGERPRINT, "b0_prediction_sha256": cohort_fp, "customer_universe_match_pass": universe_check["pass"], "prediction_readback_pass": prediction_readback["pass"], "model_readback_pass": model_readback["pass"]}
        write_json(analysis_dir / "b0_prediction_manifest.json", manifest)
        if cohort_fp == "" or not b0_1m_metrics["metrics_valid"]:
            raise ValueError("B0 1M fingerprint or metrics invalid.")

        summary = {"stage": 21, "authority": "BASIC_KMEANS_BASELINE_V1", "model_id": "B0", "model_role": "technical_baseline", "primary_experiment_line": ["B0_BASIC_KMEANS", "M1_PROJECT_MAIN_KMEANS", "M2_ENHANCED_KMEANS"], "full_training_count": FULL_COUNT, "comparison_cohort_count": COHORT_COUNT, "b0_metrics": full_metrics, "m1_reference_metrics": m1_metrics, "controlled_difference": "initMode random vs k-means||", "seed_sensitivity_complete": True, "primary_reproducibility_pass": repro_pass, "b0_prediction_sha256": cohort_fp, "customer_universe_match_pass": universe_check["pass"], "model_readback_pass": model_readback["pass"], "prediction_readback_pass": prediction_readback["pass"], "final_model_authorities_modified": False, "stage20_auxiliary_benchmark_preserved": stage20.get("authority") == "THREE_MODEL_RUNTIME_BENCHMARK_1M_V1", "gmm_executed": False, "erfm5d_training_used": False, "warnings": warnings, "blockers": [], "ready_for_stage22": True}
        write_json(summary_path, summary)
        artifact_paths = [analysis_dir / name for name in ["b0_configuration.json", "b0_full_data_metrics.json", "b0_vs_m1_controlled_check.json", "random_seed_sensitivity.json", "b0_reproducibility.json", "b0_1m_metrics.json", "b0_prediction_manifest.json", "stage21_final_summary.json"]] + [model_path, prediction_path, comparison_dir / "basic_kmeans_predictions_1m", report_path, project_root / "reports" / "stage21_run.log"]
        report_lines = ["STAGE 21 — BUILD & LOCK BASIC K-MEANS BASELINE B0", "", f"AUTHORITY: {summary['authority']}", f"FULL_TRAINING_COUNT: {FULL_COUNT}", f"COHORT_COUNT: {COHORT_COUNT}", f"B0_SILHOUETTE_RFM3D: {primary_silhouette}", f"B0_WCSS: {primary_wcss}", f"M1_SILHOUETTE_RFM3D: {m1_metrics['silhouette_rfm3d']}", f"M1_WCSS: {m1_metrics['trainingCost_WCSS']}", f"SILHOUETTE_DELTA_B0_MINUS_M1: {controlled_check['silhouette_delta_b0_minus_m1']}", f"WCSS_DELTA_B0_MINUS_M1: {controlled_check['wcss_delta_b0_minus_m1']}", f"B0_PREDICTION_SHA256: {cohort_fp}", f"PRIMARY_REPRODUCIBILITY_PASS: {repro_pass}", f"READY_FOR_STAGE_22: {summary['ready_for_stage22']}", "", "RANDOM SEED SENSITIVITY:", str(sensitivity["summary"]), "", "NO B0/M1/M2 FINAL RECOMMENDATION MADE IN STAGE 21."]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        # Re-read after report creation so the final gate covers every
        # required JSON/model/Parquet/report artifact and the Tee log.
        readback = readback_artifacts(artifact_paths, project_root)
        summary["artifact_readback"] = readback
        summary["artifact_readback_pass"] = all(readback.values())
        if not summary["artifact_readback_pass"]:
            summary["ready_for_stage22"] = False
            summary["blockers"] = ["Stage 21 artifact read-back failed."]
        write_json(summary_path, summary)
        print("AUTHORITY_AUDIT_PASS=True")
        print("INPUT_VALIDATION_PASS=True")
        print("B0_PRIMARY_METRICS_PASS=True")
        print("B0_REPRODUCIBILITY_PASS=True")
        print("COHORT_UNIVERSE_PASS=True")
        print(f"READY_FOR_STAGE_22={summary['ready_for_stage22']}")
        return 0 if summary["ready_for_stage22"] else 3
    except Exception as error:
        blockers.append(str(error))
        failed = {"stage": 21, "authority": "BASIC_KMEANS_BASELINE_V1", "ready_for_stage22": False, "warnings": warnings, "blockers": blockers}
        write_json(summary_path, failed)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("STAGE 21 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_22: False\n", encoding="utf-8")
        print(f"STAGE 21 FAILED CLOSED: {error}")
        return 3
    finally:
        safe_unpersist(primary_predictions)
        safe_unpersist(repro_predictions)
        safe_unpersist(saved_full)
        safe_unpersist(cached)
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    sys.exit(main())
