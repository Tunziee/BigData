"""Stage 8 - Main-model K-Means sweep for K=2..8.

This stage produces experiment evidence only.  It does not select an
optimal K, train a final model, interpret clusters, or build improved models.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col,
    count,
    countDistinct,
    element_at,
    isnan,
    size,
    sum as spark_sum,
    when,
)
from pyspark.storagelevel import StorageLevel


PROJECT_FEATURES = ["Recency", "Frequency", "Monetary"]
FEATURE_ORDER = ["Recency", "Frequency", "Monetary"]
K_VALUES = list(range(2, 9))
EXPECTED_ROW_COUNT = 1_362_281
VECTOR_DIMENSION = 3
SEED = 42
DISTANCE_MEASURE = "euclidean"
INIT_MODE = "k-means||"
MAX_ITER = 100
TOL = 1e-4
CACHE_POLICY = "MEMORY_AND_DISK"
POSITIVE_INFINITY = float("inf")
NEGATIVE_INFINITY = float("-inf")


def finite(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def finite_or_none(value: Any) -> Optional[float]:
    return float(value) if finite(value) else None


def enable_windows_java_filesystem_fallback(spark: SparkSession) -> None:
    """Use Java filesystem checks when Hadoop NativeIO is unavailable."""

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
        base = unsafe.staticFieldBase(windows_field)
        offset = unsafe.staticFieldOffset(windows_field)
        unsafe.putBoolean(base, offset, False)
    print("Windows Hadoop NativeIO fallback: enabled")


def vector_quality(df: DataFrame, vector_col: str) -> Dict[str, Any]:
    values = vector_to_array(col(vector_col))
    expressions = [
        spark_sum(when(col(vector_col).isNull(), 1).otherwise(0)).alias(
            "null_vector"
        ),
        spark_sum(
            when(col(vector_col).isNotNull() & (size(values) != VECTOR_DIMENSION), 1)
            .otherwise(0)
        ).alias("wrong_dimension"),
    ]
    for index in range(1, VECTOR_DIMENSION + 1):
        value = element_at(values, index)
        expressions.extend(
            [
                spark_sum(when(isnan(value), 1).otherwise(0)).alias(
                    f"nan_{index - 1}"
                ),
                spark_sum(
                    when(
                        (value == POSITIVE_INFINITY) | (value == NEGATIVE_INFINITY),
                        1,
                    ).otherwise(0)
                ).alias(f"infinity_{index - 1}"),
            ]
        )
    result = df.agg(*expressions).first().asDict()
    result["nan_total"] = int(
        sum(int(result[f"nan_{i}"] or 0) for i in range(VECTOR_DIMENSION))
    )
    result["infinity_total"] = int(
        sum(
            int(result[f"infinity_{i}"] or 0)
            for i in range(VECTOR_DIMENSION)
        )
    )
    result["pass"] = bool(
        result["null_vector"] == 0
        and result["wrong_dimension"] == 0
        and result["nan_total"] == 0
        and result["infinity_total"] == 0
    )
    return result


def validate_input(df: DataFrame) -> Dict[str, Any]:
    row = df.agg(
        count("*").alias("row_count"),
        countDistinct("customer_id").alias("distinct_customers"),
        spark_sum(when(col("customer_id").isNull(), 1).otherwise(0)).alias(
            "null_customer_id"
        ),
    ).first().asDict()
    row["duplicate_customer_id"] = int(
        row["row_count"] - row["distinct_customers"]
    )
    row["features_raw_quality"] = vector_quality(df, "features_raw")
    row["features_quality"] = vector_quality(df, "features")
    row["pass"] = bool(
        row["row_count"] == EXPECTED_ROW_COUNT
        and row["distinct_customers"] == EXPECTED_ROW_COUNT
        and row["duplicate_customer_id"] == 0
        and row["null_customer_id"] == 0
        and row["features_raw_quality"]["pass"]
        and row["features_quality"]["pass"]
    )
    return row


def create_kmeans(k: int) -> KMeans:
    model = KMeans(
        k=k,
        seed=SEED,
        featuresCol="features",
        predictionCol="prediction",
        maxIter=MAX_ITER,
        tol=TOL,
        initMode=INIT_MODE,
        distanceMeasure=DISTANCE_MEASURE,
    )
    return model


def cluster_distribution(predictions: DataFrame, k: int) -> Dict[str, Any]:
    rows = (
        predictions.groupBy("prediction")
        .count()
        .orderBy("prediction")
        .collect()
    )
    sizes = [
        {"cluster_id": int(row["prediction"]), "count": int(row["count"])}
        for row in rows
    ]
    total = sum(item["count"] for item in sizes)
    for item in sizes:
        item["ratio_pct"] = (
            float(item["count"]) / float(total) * 100.0 if total else None
        )
    counts = [item["count"] for item in sizes]
    ratios = [item["ratio_pct"] for item in sizes]
    return {
        "cluster_sizes": sizes,
        "cluster_count_sum": total,
        "num_produced_clusters": len(sizes),
        "min_cluster_size": min(counts) if counts else None,
        "max_cluster_size": max(counts) if counts else None,
        "min_cluster_ratio_pct": min(ratios) if ratios else None,
        "max_cluster_ratio_pct": max(ratios) if ratios else None,
        "pass": bool(total == EXPECTED_ROW_COUNT and len(sizes) == k),
    }


def centers_as_lists(model: Any) -> List[List[float]]:
    centers: List[List[float]] = []
    for center in model.clusterCenters():
        if hasattr(center, "toArray"):
            values = center.toArray().tolist()
        elif hasattr(center, "tolist"):
            values = center.tolist()
        else:
            values = list(center)
        centers.append([float(value) for value in values])
    return centers


def centers_valid(centers: List[List[float]], k: int) -> bool:
    return bool(
        len(centers) == k
        and all(
            len(center) == VECTOR_DIMENSION
            and all(finite(value) for value in center)
            for center in centers
        )
    )


def run_experiment(model_input: DataFrame, k: int, evaluator: ClusteringEvaluator) -> Dict[str, Any]:
    print(f"START K={k}")
    fit_start = time.perf_counter()
    model = create_kmeans(k)
    fitted = model.fit(model_input)
    fit_runtime = time.perf_counter() - fit_start

    transform_eval_start = time.perf_counter()
    predictions = fitted.transform(model_input).select("features", "prediction")
    silhouette = float(evaluator.evaluate(predictions))
    summary = fitted.summary
    wcss = float(summary.trainingCost)
    num_iter: Any = getattr(summary, "numIter", "NOT_EXPOSED")
    try:
        num_iter = int(num_iter)
    except (TypeError, ValueError):
        num_iter = "NOT_EXPOSED"
    distribution = cluster_distribution(predictions, k)
    centers = centers_as_lists(fitted)
    transform_eval_runtime = time.perf_counter() - transform_eval_start
    total_runtime = time.perf_counter() - fit_start

    metric_pass = bool(
        finite(silhouette)
        and -1.0 <= silhouette <= 1.0
        and finite(wcss)
        and wcss > 0.0
        and finite(fit_runtime)
        and fit_runtime > 0.0
        and finite(transform_eval_runtime)
        and transform_eval_runtime > 0.0
        and finite(total_runtime)
        and total_runtime > 0.0
    )
    status = bool(
        metric_pass
        and distribution["pass"]
        and centers_valid(centers, k)
    )
    result = {
        "k": k,
        "status": "PASS" if status else "FAIL",
        "silhouette": finite_or_none(silhouette),
        "wcss": finite_or_none(wcss),
        "fit_runtime_sec": finite_or_none(fit_runtime),
        "transform_and_evaluation_runtime_sec": finite_or_none(
            transform_eval_runtime
        ),
        "total_runtime_sec": finite_or_none(total_runtime),
        "num_iter": num_iter,
        "num_produced_clusters": distribution["num_produced_clusters"],
        "cluster_count_sum": distribution["cluster_count_sum"],
        "cluster_sizes": distribution["cluster_sizes"],
        "cluster_ratios_pct": [
            item["ratio_pct"] for item in distribution["cluster_sizes"]
        ],
        "min_cluster_size": distribution["min_cluster_size"],
        "max_cluster_size": distribution["max_cluster_size"],
        "min_cluster_ratio_pct": distribution["min_cluster_ratio_pct"],
        "max_cluster_ratio_pct": distribution["max_cluster_ratio_pct"],
        "centers_scaled": centers,
        "metric_validation_pass": metric_pass,
        "cluster_validation_pass": distribution["pass"],
        "centers_validation_pass": centers_valid(centers, k),
        "validation_pass": status,
    }
    print(
        f"K={k} status={result['status']} silhouette={result['silhouette']} "
        f"wcss={result['wcss']} fit_runtime_sec={result['fit_runtime_sec']} "
        f"clusters={result['num_produced_clusters']}"
    )
    return result


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, experiments: List[Dict[str, Any]]) -> None:
    fields = [
        "k",
        "status",
        "silhouette",
        "wcss",
        "fit_runtime_sec",
        "transform_and_evaluation_runtime_sec",
        "total_runtime_sec",
        "num_produced_clusters",
        "min_cluster_size",
        "max_cluster_size",
        "min_cluster_ratio_pct",
        "max_cluster_ratio_pct",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for experiment in experiments:
            writer.writerow({field: experiment.get(field) for field in fields})


def validate_written_artifacts(
    json_path: Path,
    csv_path: Path,
    cluster_dir: Path,
    centers_dir: Path,
    expected_k_values: List[int],
) -> Dict[str, Any]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    json_ks = [int(item["k"]) for item in payload.get("experiments", [])]
    csv_ks = [int(row["k"]) for row in csv_rows]
    cluster_files = [cluster_dir / f"k{k}.json" for k in expected_k_values]
    center_files = [centers_dir / f"k{k}.json" for k in expected_k_values]
    file_pass = all(path.exists() for path in cluster_files + center_files)
    return {
        "json_exists": json_path.exists(),
        "csv_exists": csv_path.exists(),
        "json_k_values": json_ks,
        "csv_k_values": csv_ks,
        "cluster_files_exist": file_pass,
        "centers_files_exist": file_pass,
        "pass": bool(
            json_ks == expected_k_values
            and csv_ks == expected_k_values
            and file_pass
        ),
    }


def wcss_observations(experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
    warnings: List[str] = []
    values = [(int(item["k"]), float(item["wcss"])) for item in experiments]
    for (previous_k, previous), (current_k, current) in zip(values, values[1:]):
        if current > previous + max(1e-9, abs(previous) * 1e-9):
            warnings.append(
                f"WCSS increased from K={previous_k} to K={current_k}; retained measured values."
            )
    return {
        "values": [{"k": k, "wcss": value} for k, value in values],
        "non_increasing": not warnings,
        "warnings": warnings,
    }


def write_report(
    report_path: Path,
    project_root: Path,
    input_path: Path,
    input_validation: Dict[str, Any],
    cache_info: Dict[str, Any],
    experiments: List[Dict[str, Any]],
    artifact_validation: Dict[str, Any],
    ready: bool,
    blockers: List[str],
    warnings: List[str],
    rerun: bool,
) -> None:
    lines = [
        "STAGE 8 — MAIN MODEL K-MEANS K=2..8 EXPERIMENT",
        "=" * 110,
        f"PROJECT_ROOT: {project_root}",
        f"INPUT_PATH: {input_path}",
        f"RERUN_STAGE_8: {rerun}",
        "",
        "MAIN_MODEL_AUTHORITY",
        "RFM -> VectorAssembler -> StandardScaler -> K-Means",
        "Stage 8 produces experiment evidence only; it does not select K or train a final model.",
        "",
        "SPARK_CONFIGURATION",
        "Spark master: local[4]",
        "Spark version: recorded in JSON artifact",
        f"CACHE_POLICY: {CACHE_POLICY}",
        f"CACHE_MATERIALIZATION: {cache_info}",
        "",
        "KMEANS_CONFIGURATION",
        f"seed={SEED}; distanceMeasure={DISTANCE_MEASURE}; initMode={INIT_MODE}; maxIter={MAX_ITER}; tol={TOL}",
        "All K values use the same configuration except K.",
        "",
        "PRE-FLIGHT_VALIDATION",
        f"{input_validation}",
        "",
        "EXPERIMENT_RESULTS_TABLE",
        "K | Status | Silhouette | WCSS(trainingCost) | FitRuntimeSec | TransformEvalRuntimeSec | TotalRuntimeSec | MinClusterSize | MaxClusterSize | MinRatioPct | MaxRatioPct | NumProducedClusters",
        "-" * 240,
    ]
    for item in experiments:
        lines.append(
            " | ".join(
                str(item.get(field))
                for field in [
                    "k",
                    "status",
                    "silhouette",
                    "wcss",
                    "fit_runtime_sec",
                    "transform_and_evaluation_runtime_sec",
                    "total_runtime_sec",
                    "min_cluster_size",
                    "max_cluster_size",
                    "min_cluster_ratio_pct",
                    "max_cluster_ratio_pct",
                    "num_produced_clusters",
                ]
            )
        )

    successful_experiments = [
        item
        for item in experiments
        if item.get("status") == "PASS" and item.get("silhouette") is not None
    ]
    if successful_experiments:
        best = max(successful_experiments, key=lambda item: item["silhouette"])
        best_observed = f"K={best['k']} (descriptive observation only; not a final selection)"
    else:
        best_observed = "NOT_AVAILABLE"
    wcss = wcss_observations(successful_experiments) if successful_experiments else {
        "values": [],
        "non_increasing": False,
        "warnings": [],
    }
    lines.extend(
        [
            "",
            f"BEST_SILHOUETTE_K_OBSERVED: {best_observed}",
            "ELBOW_CANDIDATES: deferred to Stage 9; no final decision made.",
            f"WCSS_TREND: {wcss}",
            "",
            "CLUSTER_SIZE_SUMMARY",
        ]
    )
    for item in experiments:
        lines.append(
            f"K={item.get('k')}: sum={item.get('cluster_count_sum')}, "
            f"clusters={item.get('num_produced_clusters')}, sizes={item.get('cluster_sizes', [])}"
        )
    lines.extend(
        [
            "",
            "CENTERS_VALIDATION",
            "Centers are stored in standardized RFM space only; no business interpretation was performed.",
            *[
                f"K={item.get('k')}: count={len(item.get('centers_scaled', []))}, dimensions={[len(center) for center in item.get('centers_scaled', [])]}, pass={item.get('centers_validation_pass', False)}"
                for item in experiments
            ],
            "",
            "RUNTIME_SUMMARY",
            *[
                f"K={item.get('k')}: fit={item.get('fit_runtime_sec')} sec; transform+evaluation={item.get('transform_and_evaluation_runtime_sec')} sec; total={item.get('total_runtime_sec')} sec"
                for item in experiments
            ],
            "",
            f"ARTIFACT_WRITE_READBACK: {artifact_validation}",
            f"WARNINGS: {warnings}",
            f"BLOCKERS: {blockers}",
            f"READY_FOR_STAGE_9: {ready}",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "output" / "ml" / "main_rfm_scaled"
    analysis_root = project_root / "output" / "analysis" / "stage8"
    cluster_dir = analysis_root / "cluster_sizes"
    centers_dir = analysis_root / "centers"
    json_path = analysis_root / "kmeans_k_sweep.json"
    csv_path = analysis_root / "kmeans_k_sweep.csv"
    report_path = project_root / "reports" / "stage8_main_kmeans_k_sweep.txt"

    required_columns = [
        "customer_id",
        "Recency",
        "Frequency",
        "Monetary",
        "features_raw",
        "features",
    ]
    rerun = analysis_root.exists()
    for path in [analysis_root, cluster_dir, centers_dir, report_path.parent]:
        path.mkdir(parents=True, exist_ok=True)

    spark_tmp = project_root.parent / "spark_tmp"
    spark_tmp.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .master("local[4]")
        .appName("Stage8-Main-KMeans-K-Sweep")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.local.dir", str(spark_tmp))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    enable_windows_java_filesystem_fallback(spark)

    input_validation: Dict[str, Any] = {}
    cache_info: Dict[str, Any] = {}
    experiments: List[Dict[str, Any]] = []
    blockers: List[str] = []
    warnings: List[str] = []
    artifact_validation: Dict[str, Any] = {"pass": False}
    ready = False
    cached_df: Optional[DataFrame] = None

    try:
        if not input_path.exists():
            blockers.append(f"Missing Stage 7 input: {input_path}")
            raise FileNotFoundError(blockers[-1])

        source = spark.read.parquet(str(input_path))
        missing_columns = [c for c in required_columns if c not in source.columns]
        if missing_columns:
            blockers.append(f"Missing required columns: {missing_columns}")
            raise ValueError(blockers[-1])

        audit_df = source.select(*required_columns).persist(StorageLevel.MEMORY_AND_DISK)
        materialization_start = time.perf_counter()
        cache_row_count = audit_df.count()
        materialization_runtime = time.perf_counter() - materialization_start
        cache_info = {
            "storage_level": CACHE_POLICY,
            "materialized_row_count": cache_row_count,
            "materialization_runtime_sec": materialization_runtime,
            "materialized_before_k_loop": True,
        }

        input_validation = validate_input(audit_df)
        print(f"Input validation: {input_validation}")
        if not input_validation["pass"]:
            blockers.append("Stage 7 input validation failed; fail-closed.")
            raise ValueError(blockers[-1])

        model_input = audit_df.select("features")
        evaluator = ClusteringEvaluator(
            predictionCol="prediction",
            featuresCol="features",
            metricName="silhouette",
            distanceMeasure="squaredEuclidean",
        )

        for k in K_VALUES:
            try:
                result = run_experiment(model_input, k, evaluator)
                experiments.append(result)
                if result["status"] != "PASS":
                    blockers.append(f"K={k} validation failed.")
                    break
            except Exception as error:
                blockers.append(f"K={k} failed: {error}")
                experiments.append({"k": k, "status": "FAIL", "error": str(error)})
                break

        if len(experiments) != len(K_VALUES):
            blockers.append(
                f"Expected {len(K_VALUES)} experiments, completed {len(experiments)}."
            )

        if experiments and all(item.get("status") == "PASS" for item in experiments):
            wcss_info = wcss_observations(experiments)
            warnings.extend(wcss_info["warnings"])

        base_payload = {
            "stage": 8,
            "model": "Main RFM K-Means",
            "input": {
                "path": str(input_path),
                "row_count": input_validation.get("row_count"),
                "distinct_customers": input_validation.get("distinct_customers"),
                "feature_col": "features",
                "feature_order": FEATURE_ORDER,
            },
            "spark": {
                "master": "local[4]",
                "version": spark.version,
                "cache_policy": CACHE_POLICY,
                "cache_materialization": cache_info,
            },
            "kmeans_config": {
                "seed": SEED,
                "distanceMeasure": DISTANCE_MEASURE,
                "initMode": INIT_MODE,
                "maxIter": MAX_ITER,
                "tol": TOL,
            },
            "evaluator_config": {
                "metricName": "silhouette",
                "distanceMeasure": "squaredEuclidean",
                "note": "Squared Euclidean evaluator metric is consistent with Euclidean K-Means geometry.",
            },
            "experiments": experiments,
            "selection": {
                "best_silhouette_k_observed": (
                    max(experiments, key=lambda item: item["silhouette"])["k"]
                    if experiments and all(item.get("status") == "PASS" for item in experiments)
                    else None
                ),
                "final_k_selected": None,
                "elbow_candidates": "deferred to Stage 9",
            },
        }

        for item in experiments:
            k = int(item["k"])
            write_json(
                cluster_dir / f"k{k}.json",
                {
                    "stage": 8,
                    "k": k,
                    "cluster_sizes": item.get("cluster_sizes", []),
                    "cluster_count_sum": item.get("cluster_count_sum"),
                    "num_produced_clusters": item.get("num_produced_clusters"),
                    "expected_row_count": EXPECTED_ROW_COUNT,
                },
            )
            write_json(
                centers_dir / f"k{k}.json",
                {
                    "stage": 8,
                    "k": k,
                    "space": "standardized RFM",
                    "feature_order": FEATURE_ORDER,
                    "centers_scaled": item.get("centers_scaled", []),
                },
            )
        write_json(json_path, base_payload)
        write_csv(csv_path, experiments)
        artifact_validation = validate_written_artifacts(
            json_path,
            csv_path,
            cluster_dir,
            centers_dir,
            [int(item["k"]) for item in experiments],
        )
        if not artifact_validation["pass"]:
            blockers.append("Stage 8 artifact read-back validation failed.")

        ready = bool(
            len(experiments) == len(K_VALUES)
            and all(item.get("status") == "PASS" for item in experiments)
            and artifact_validation["pass"]
            and input_validation.get("pass")
        )
        base_payload["validation"] = {
            "input_pass": input_validation.get("pass", False),
            "all_k_values_completed": len(experiments) == len(K_VALUES),
            "all_experiments_pass": all(
                item.get("status") == "PASS" for item in experiments
            ),
            "artifact_write_readback_pass": artifact_validation["pass"],
            "wcss_sanity_warnings": warnings,
        }
        base_payload["ready_for_stage9"] = ready
        write_json(json_path, base_payload)

        write_report(
            report_path,
            project_root,
            input_path,
            input_validation,
            cache_info,
            experiments,
            artifact_validation,
            ready,
            blockers,
            warnings,
            rerun,
        )
        print(f"Stage 8 report written: {report_path}")
        print(f"READY_FOR_STAGE_9={ready}")
        return 0 if ready else 3
    except Exception as error:
        if not blockers:
            blockers.append(str(error))
        print(f"STAGE 8 FAILED CLOSED: {error}")
        try:
            failure_payload = {
                "stage": 8,
                "model": "Main RFM K-Means",
                "input": {"path": str(input_path)},
                "experiments": experiments,
                "validation": {"input_pass": input_validation.get("pass", False)},
                "ready_for_stage9": False,
                "blockers": blockers,
            }
            write_json(json_path, failure_payload)
            write_report(
                report_path,
                project_root,
                input_path,
                input_validation,
                cache_info,
                experiments,
                artifact_validation,
                False,
                blockers,
                warnings,
                rerun,
            )
        except Exception as report_error:
            print(f"Could not write failure artifacts: {report_error}")
        return 3
    finally:
        if cached_df is not None:
            cached_df.unpersist()
        if 'audit_df' in locals():
            audit_df.unpersist()
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
