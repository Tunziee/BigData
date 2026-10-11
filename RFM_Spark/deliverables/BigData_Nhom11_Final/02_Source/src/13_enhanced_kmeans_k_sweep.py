"""Stage 13: Enhanced K-Means K=2..8 experiment on the locked 5D space.

This stage produces experiment evidence only.  It does not select K, train a
final Enhanced model, compare against the main model, or run GMM.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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


EXPECTED_ROW_COUNT = 1_362_281
VECTOR_DIMENSION = 5
FEATURE_ORDER = [
    "Recency_improved",
    "Frequency_log1p",
    "Monetary_log1p",
    "AvgBasketSize_log1p",
    "ProductVariety_log1p",
]
K_VALUES = list(range(2, 9))
SEED = 42
DISTANCE_MEASURE = "euclidean"
INIT_MODE = "k-means||"
MAX_ITER = 100
TOL = 1e-4
CACHE_POLICY = "MEMORY_AND_DISK"
POSITIVE_INFINITY = float("inf")
NEGATIVE_INFINITY = float("-inf")

REQUIRED_COLUMNS = [
    "customer_id",
    "Recency",
    "Frequency",
    "Monetary",
    "AvgBasketSize",
    "ProductVariety",
    "Recency_improved",
    "Frequency_log1p",
    "Monetary_log1p",
    "AvgBasketSize_log1p",
    "ProductVariety_log1p",
    "features_improved_raw",
    "features_improved",
]


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def finite_or_none(value: Any) -> Optional[float]:
    return float(value) if finite(value) else None


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def enable_windows_java_filesystem_fallback(spark: SparkSession) -> None:
    """Use Java filesystem checks when Hadoop NativeIO is unavailable on Windows."""

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
        spark_sum(when(col(vector_col).isNull(), 1).otherwise(0)).alias("null_vector"),
        spark_sum(
            when(col(vector_col).isNotNull() & (size(values) != VECTOR_DIMENSION), 1)
            .otherwise(0)
        ).alias("wrong_dimension"),
    ]
    for index in range(1, VECTOR_DIMENSION + 1):
        value = element_at(values, index)
        expressions.extend(
            [
                spark_sum(when(isnan(value), 1).otherwise(0)).alias(f"nan_{index - 1}"),
                spark_sum(
                    when(
                        (value == POSITIVE_INFINITY) | (value == NEGATIVE_INFINITY),
                        1,
                    ).otherwise(0)
                ).alias(f"infinity_{index - 1}"),
            ]
        )
    row = df.agg(*expressions).first().asDict()
    nan_total = sum(int(row[f"nan_{index}"] or 0) for index in range(VECTOR_DIMENSION))
    infinity_total = sum(
        int(row[f"infinity_{index}"] or 0) for index in range(VECTOR_DIMENSION)
    )
    return {
        "null_vector": int(row["null_vector"] or 0),
        "wrong_dimension": int(row["wrong_dimension"] or 0),
        "nan_total": nan_total,
        "infinity_total": infinity_total,
        "pass": bool(
            row["null_vector"] == 0
            and row["wrong_dimension"] == 0
            and nan_total == 0
            and infinity_total == 0
        ),
    }


def validate_input(df: DataFrame) -> Dict[str, Any]:
    row = df.agg(
        count("*").alias("row_count"),
        countDistinct("customer_id").alias("distinct_customer_id"),
        spark_sum(when(col("customer_id").isNull(), 1).otherwise(0)).alias(
            "null_customer_id"
        ),
    ).first().asDict()
    row["duplicate_customer_id"] = int(row["row_count"] - row["distinct_customer_id"])
    row["features_improved_quality"] = vector_quality(df, "features_improved")
    row["features_improved_raw_quality"] = vector_quality(df, "features_improved_raw")
    row["pass"] = bool(
        row["row_count"] == EXPECTED_ROW_COUNT
        and row["distinct_customer_id"] == EXPECTED_ROW_COUNT
        and row["duplicate_customer_id"] == 0
        and row["null_customer_id"] == 0
        and row["features_improved_quality"]["pass"]
        and row["features_improved_raw_quality"]["pass"]
    )
    return row


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


def centers_valid(centers: List[List[float]], expected_k: int) -> bool:
    return bool(
        len(centers) == expected_k
        and all(
            len(center) == VECTOR_DIMENSION
            and all(finite(value) for value in center)
            for center in centers
        )
    )


def cluster_distribution(predictions: DataFrame, k: int) -> Dict[str, Any]:
    rows = predictions.groupBy("prediction").count().orderBy("prediction").collect()
    cluster_sizes = [
        {"cluster_id": int(row["prediction"]), "count": int(row["count"])}
        for row in rows
    ]
    total = sum(item["count"] for item in cluster_sizes)
    for item in cluster_sizes:
        item["ratio_pct"] = float(item["count"]) / float(total) * 100.0 if total else None
    ratios = [float(item["ratio_pct"]) for item in cluster_sizes]
    counts = [int(item["count"]) for item in cluster_sizes]
    return {
        "cluster_sizes": cluster_sizes,
        "cluster_count_sum": total,
        "num_produced_clusters": len(cluster_sizes),
        "min_cluster_size": min(counts) if counts else None,
        "max_cluster_size": max(counts) if counts else None,
        "min_cluster_ratio_pct": min(ratios) if ratios else None,
        "max_cluster_ratio_pct": max(ratios) if ratios else None,
        "clusters_below_0_5_pct": sum(ratio < 0.5 for ratio in ratios),
        "clusters_below_1_pct": sum(ratio < 1.0 for ratio in ratios),
        "clusters_below_2_pct": sum(ratio < 2.0 for ratio in ratios),
        "clusters_below_5_pct": sum(ratio < 5.0 for ratio in ratios),
        "pass": bool(total == EXPECTED_ROW_COUNT and len(cluster_sizes) == k),
    }


def create_kmeans(k: int) -> KMeans:
    return KMeans(
        k=k,
        seed=SEED,
        featuresCol="features_improved",
        predictionCol="prediction",
        distanceMeasure=DISTANCE_MEASURE,
        initMode=INIT_MODE,
        maxIter=MAX_ITER,
        tol=TOL,
    )


def run_experiment(
    model_input: DataFrame,
    k: int,
    evaluator: ClusteringEvaluator,
) -> Dict[str, Any]:
    print(f"START ENHANCED K={k}")
    fit_start = time.perf_counter()
    fitted = create_kmeans(k).fit(model_input)
    fit_runtime = time.perf_counter() - fit_start

    transform_eval_start = time.perf_counter()
    predictions = fitted.transform(model_input).select("features_improved", "prediction")
    silhouette = float(evaluator.evaluate(predictions))
    wcss = float(fitted.summary.trainingCost)
    num_iter: Any = getattr(fitted.summary, "numIter", "NOT_EXPOSED")
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
    validation_pass = bool(metric_pass and distribution["pass"] and centers_valid(centers, k))
    result = {
        "k": k,
        "status": "PASS" if validation_pass else "FAIL",
        "silhouette": finite_or_none(silhouette),
        "wcss_trainingCost": finite_or_none(wcss),
        "fit_runtime_sec": finite_or_none(fit_runtime),
        "transform_evaluation_runtime_sec": finite_or_none(transform_eval_runtime),
        "total_runtime_sec": finite_or_none(total_runtime),
        "num_iter": num_iter,
        "convergence_flag": "CONVERGENCE_LIMIT_REACHED"
        if num_iter == MAX_ITER
        else "NORMAL",
        "num_produced_clusters": distribution["num_produced_clusters"],
        "cluster_count_sum": distribution["cluster_count_sum"],
        "cluster_sizes": distribution["cluster_sizes"],
        "min_cluster_size": distribution["min_cluster_size"],
        "max_cluster_size": distribution["max_cluster_size"],
        "min_cluster_ratio_pct": distribution["min_cluster_ratio_pct"],
        "max_cluster_ratio_pct": distribution["max_cluster_ratio_pct"],
        "clusters_below_0_5_pct": distribution["clusters_below_0_5_pct"],
        "clusters_below_1_pct": distribution["clusters_below_1_pct"],
        "clusters_below_2_pct": distribution["clusters_below_2_pct"],
        "clusters_below_5_pct": distribution["clusters_below_5_pct"],
        "centers_scaled_improved_5d": centers,
        "metric_validation_pass": metric_pass,
        "cluster_validation_pass": distribution["pass"],
        "centers_validation_pass": centers_valid(centers, k),
        "validation_pass": validation_pass,
    }
    print(
        f"ENHANCED K={k} status={result['status']} silhouette={result['silhouette']} "
        f"wcss={result['wcss_trainingCost']} fit_runtime_sec={result['fit_runtime_sec']}"
    )
    return result


def wcss_trend(experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
    values = [(int(item["k"]), float(item["wcss_trainingCost"])) for item in experiments]
    warnings: List[str] = []
    for (previous_k, previous), (current_k, current) in zip(values, values[1:]):
        if current > previous + max(1e-9, abs(previous) * 1e-9):
            warnings.append(
                f"WCSS increased from K={previous_k} to K={current_k}; measured values retained."
            )
    return {
        "values": [{"k": k, "wcss_trainingCost": value} for k, value in values],
        "non_increasing": not warnings,
        "warnings": warnings,
        "space_note": "WCSS is within standardized improved 5D feature space and is not directly comparable in absolute value to main 3D WCSS.",
    }


def write_csv(path: Path, experiments: List[Dict[str, Any]]) -> None:
    fields = [
        "k",
        "status",
        "silhouette",
        "wcss_trainingCost",
        "fit_runtime_sec",
        "transform_evaluation_runtime_sec",
        "total_runtime_sec",
        "num_iter",
        "num_produced_clusters",
        "min_cluster_size",
        "max_cluster_size",
        "min_cluster_ratio_pct",
        "max_cluster_ratio_pct",
        "clusters_below_0_5_pct",
        "clusters_below_1_pct",
        "clusters_below_2_pct",
        "clusters_below_5_pct",
        "convergence_flag",
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
    payload = read_json(json_path)
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


def write_report(
    path: Path,
    payload: Dict[str, Any],
    input_validation: Dict[str, Any],
    cache_info: Dict[str, Any],
    artifact_validation: Dict[str, Any],
    blockers: List[str],
    warnings: List[str],
    ready: bool,
) -> None:
    lines = [
        "STAGE 13 — ENHANCED K-MEANS K=2..8 EXPERIMENT",
        "=" * 115,
        "CURRENT_STAGE: Stage 13",
        f"IMPROVED_FEATURE_AUTHORITY: {payload['feature_authority']}",
        f"INPUT_PATH: {payload['input']['path']}",
        "NO_K_SELECTION: True",
        "NO_FINAL_ENHANCED_MODEL: True",
        "NO_GMM: True",
        "",
        "SPARK_CONFIGURATION",
        f"{payload['spark_configuration']}",
        f"CACHE_POLICY: {CACHE_POLICY}; materialization={cache_info}",
        "",
        "ENHANCED_KMEANS_CONFIGURATION",
        f"{payload['algorithm_config']}",
        "Fairness lock: same algorithm configuration as main model; only feature space differs.",
        "",
        "INPUT_VALIDATION",
        f"{input_validation}",
        "",
        "EXPERIMENT_RESULTS_TABLE",
        "K | Status | Silhouette | WCSS/trainingCost | FitSec | TransformEvalSec | TotalSec | NumIter | Clusters | MinSize | MaxSize | MinRatioPct | MaxRatioPct | Below1Pct | Below2Pct",
    ]
    for item in payload["experiments"]:
        lines.append(
            f"{item.get('k')} | {item.get('status')} | {item.get('silhouette')} | {item.get('wcss_trainingCost')} | "
            f"{item.get('fit_runtime_sec')} | {item.get('transform_evaluation_runtime_sec')} | {item.get('total_runtime_sec')} | "
            f"{item.get('num_iter')} | {item.get('num_produced_clusters')} | {item.get('min_cluster_size')} | {item.get('max_cluster_size')} | "
            f"{item.get('min_cluster_ratio_pct')} | {item.get('max_cluster_ratio_pct')} | {item.get('clusters_below_1_pct')} | {item.get('clusters_below_2_pct')}"
        )
    best = max(payload["experiments"], key=lambda item: item["silhouette"])
    lines.extend(
        [
            "",
            f"BEST_SILHOUETTE_K_OBSERVED: K={best['k']} ({best['silhouette']}); descriptive only, no K selected.",
            f"WCSS_TREND: {payload['wcss_trend']}",
            "",
            "CLUSTER_SIZE_SUMMARY",
        ]
    )
    for item in payload["experiments"]:
        lines.append(
            f"K={item['k']}: sum={item['cluster_count_sum']}; min={item['min_cluster_size']}; max={item['max_cluster_size']}; "
            f"min_ratio={item['min_cluster_ratio_pct']}; max_ratio={item['max_cluster_ratio_pct']}; "
            f"below0.5={item['clusters_below_0_5_pct']}; below1={item['clusters_below_1_pct']}; below2={item['clusters_below_2_pct']}; below5={item['clusters_below_5_pct']}"
        )
    lines.extend(
        [
            "",
            "CONVERGENCE_SUMMARY",
            *[
                f"K={item['k']}: numIter={item['num_iter']}; flag={item['convergence_flag']}"
                for item in payload["experiments"]
            ],
            "",
            "CENTERS_VALIDATION",
            "Centers are retained in standardized improved 5D space only; no raw-RFM inverse interpretation was performed.",
            *[
                f"K={item['k']}: count={len(item['centers_scaled_improved_5d'])}; dimensions={[len(center) for center in item['centers_scaled_improved_5d']]}; pass={item['centers_validation_pass']}"
                for item in payload["experiments"]
            ],
            "",
            "RUNTIME_SUMMARY",
            *[
                f"K={item['k']}: fit={item['fit_runtime_sec']} sec; transform/evaluation={item['transform_evaluation_runtime_sec']} sec; total={item['total_runtime_sec']} sec"
                for item in payload["experiments"]
            ],
            "",
            f"MAIN_MODEL_REFERENCE: {payload['main_model_reference']}",
            f"ARTIFACT_WRITE_READBACK: {artifact_validation}",
            f"WARNINGS: {warnings}",
            f"BLOCKERS: {blockers}",
            f"READY_FOR_STAGE_14: {ready}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "output" / "ml" / "improved_erfm_scaled"
    scaler_metadata_path = project_root / "output" / "ml" / "improved_erfm_scaler_metadata" / "scaler_metadata.json"
    stage12_summary_path = project_root / "output" / "analysis" / "stage12" / "transformation_summary.json"
    stage10_summary_path = project_root / "output" / "analysis" / "stage10" / "final_model_summary.json"
    analysis_dir = project_root / "output" / "analysis" / "stage13"
    cluster_dir = analysis_dir / "cluster_sizes"
    centers_dir = analysis_dir / "centers"
    report_path = project_root / "reports" / "stage13_enhanced_kmeans_k_sweep.txt"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    cluster_dir.mkdir(parents=True, exist_ok=True)
    centers_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = analysis_dir / "enhanced_kmeans_k_sweep.json"
    csv_path = analysis_dir / "enhanced_kmeans_k_sweep.csv"

    blockers: List[str] = []
    warnings: List[str] = []
    try:
        stage12 = read_json(stage12_summary_path)
        scaler_metadata = read_json(scaler_metadata_path)
        main_summary = read_json(stage10_summary_path)
    except Exception as error:
        blockers.append(f"Authority artifact unreadable: {error}")
        failure = {
            "stage": 13,
            "authority": "ENHANCED_KMEANS_K_SWEEP",
            "ready_for_stage14": False,
            "blockers": blockers,
        }
        write_json(json_path, failure)
        report_path.write_text("STAGE 13 FAILED CLOSED\n" + str(error) + "\n", encoding="utf-8")
        return 3

    if not (
        stage12.get("stage") == 12
        and stage12.get("authority") == "IMPROVED_ERFM_FEATURE_SPACE"
        and stage12.get("ready_for_stage13") is True
    ):
        blockers.append("Stage 12 summary authority mismatch.")
    if not (
        scaler_metadata.get("withMean") is True
        and scaler_metadata.get("withStd") is True
        and scaler_metadata.get("scaled_vector_col") == "features_improved"
        and scaler_metadata.get("scaled_vector_dimension") == VECTOR_DIMENSION
        and [
            scaler_metadata.get("feature_order", {}).get(str(index))
            for index in range(VECTOR_DIMENSION)
        ]
        == FEATURE_ORDER
    ):
        blockers.append("Stage 12 scaler metadata authority mismatch.")
    if not (
        main_summary.get("stage") == 10
        and main_summary.get("authority") == "MAIN_KMEANS_FINAL"
        and main_summary.get("selected_k") == 4
    ):
        blockers.append("Main-model reference metadata is invalid.")
    if not input_path.exists():
        blockers.append(f"Missing Stage 12 input: {input_path}")
    if blockers:
        write_json(
            json_path,
            {
                "stage": 13,
                "authority": "ENHANCED_KMEANS_K_SWEEP",
                "ready_for_stage14": False,
                "blockers": blockers,
            },
        )
        report_path.write_text(
            "STAGE 13 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_14: False\n",
            encoding="utf-8",
        )
        return 3

    spark = (
        SparkSession.builder
        .master("local[4]")
        .appName("Stage13-Enhanced-KMeans-K-Sweep")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.local.dir", str(project_root.parent / "spark_tmp"))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    enable_windows_java_filesystem_fallback(spark)
    cached_df: Optional[DataFrame] = None
    try:
        source = spark.read.parquet(str(input_path))
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in source.columns]
        if missing_columns:
            raise ValueError(f"Missing required Stage 12 columns: {missing_columns}")
        selected = source.select(*REQUIRED_COLUMNS)
        cached_df = selected.persist(StorageLevel.MEMORY_AND_DISK)
        cache_start = time.perf_counter()
        cache_row_count = cached_df.count()
        cache_runtime = time.perf_counter() - cache_start
        input_validation = validate_input(cached_df)
        input_validation["cache_materialization"] = {
            "storage_level": CACHE_POLICY,
            "materialized_row_count": cache_row_count,
            "materialization_runtime_sec": cache_runtime,
            "materialized_before_k_loop": True,
        }
        print(f"AUTHORITY_AUDIT_PASS=True")
        print(f"INPUT_VALIDATION={input_validation}")
        if not input_validation["pass"]:
            raise ValueError("Stage 12 input validation failed; fail-closed.")

        model_input = cached_df.select("features_improved")
        evaluator = ClusteringEvaluator(
            predictionCol="prediction",
            featuresCol="features_improved",
            metricName="silhouette",
            distanceMeasure="squaredEuclidean",
        )
        experiments: List[Dict[str, Any]] = []
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
        successful = bool(
            len(experiments) == len(K_VALUES)
            and all(item.get("status") == "PASS" for item in experiments)
        )
        trend = wcss_trend(experiments) if successful else {
            "values": [],
            "non_increasing": False,
            "warnings": [],
            "space_note": "WCSS is within standardized improved 5D feature space.",
        }
        warnings.extend(trend["warnings"])

        best_observed = None
        if successful:
            best = max(experiments, key=lambda item: item["silhouette"])
            best_observed = {
                "k": int(best["k"]),
                "silhouette": float(best["silhouette"]),
                "descriptive_only": True,
                "no_k_selected": True,
            }

        payload = {
            "stage": 13,
            "authority": "ENHANCED_KMEANS_K_SWEEP",
            "model": "Enhanced K-Means",
            "feature_authority": "IMPROVED_ERFM_FEATURE_SPACE",
            "input": {
                "path": str(input_path),
                "row_count": input_validation.get("row_count"),
                "feature_col": "features_improved",
                "dimension": VECTOR_DIMENSION,
                "feature_order": FEATURE_ORDER,
            },
            "algorithm_config": {
                "seed": SEED,
                "distanceMeasure": DISTANCE_MEASURE,
                "initMode": INIT_MODE,
                "maxIter": MAX_ITER,
                "tol": TOL,
            },
            "fairness_lock": {
                "same_algorithm_configuration_as_main_model": True,
                "main_model_difference": "feature engineering / feature space only",
            },
            "spark_configuration": {
                "master": "local[4]",
                "version": spark.version,
                "shuffle_partitions": 64,
                "adaptive_enabled": True,
            },
            "cache_materialization": input_validation["cache_materialization"],
            "experiments": experiments,
            "best_silhouette_k_observed": best_observed,
            "wcss_trend": trend,
            "main_model_reference": {
                "authority": "MAIN_KMEANS_FINAL",
                "k": 4,
                "silhouette": main_summary.get("metrics", {}).get(
                    "silhouette", main_summary.get("training", {}).get("silhouette")
                ),
                "comparison_conclusion": "No model comparison conclusion in Stage 13; WCSS absolute values are not directly comparable across 3D and 5D spaces.",
            },
            "validation": {
                "input_pass": input_validation.get("pass", False),
                "all_k_values_completed": len(experiments) == len(K_VALUES),
                "all_experiments_pass": successful,
                "wcss_trend_sanity_warnings": trend["warnings"],
            },
            "ready_for_stage14": False,
        }

        for item in experiments:
            k = int(item["k"])
            write_json(
                cluster_dir / f"k{k}.json",
                {
                    "stage": 13,
                    "authority": "ENHANCED_KMEANS_K_SWEEP",
                    "k": k,
                    "feature_space": "standardized improved 5D",
                    "cluster_sizes": item.get("cluster_sizes", []),
                    "cluster_count_sum": item.get("cluster_count_sum"),
                    "num_produced_clusters": item.get("num_produced_clusters"),
                },
            )
            write_json(
                centers_dir / f"k{k}.json",
                {
                    "stage": 13,
                    "authority": "ENHANCED_KMEANS_K_SWEEP",
                    "k": k,
                    "space": "standardized improved 5D",
                    "feature_order": FEATURE_ORDER,
                    "centers_scaled_improved_5d": item.get("centers_scaled_improved_5d", []),
                },
            )

        write_json(json_path, payload)
        write_csv(csv_path, experiments)
        artifact_validation = validate_written_artifacts(
            json_path, csv_path, cluster_dir, centers_dir, [int(item["k"]) for item in experiments]
        )
        payload["artifact_write_readback"] = artifact_validation
        ready = bool(
            successful
            and artifact_validation["pass"]
            and input_validation.get("pass")
        )
        payload["validation"]["artifact_write_readback_pass"] = artifact_validation["pass"]
        payload["ready_for_stage14"] = ready
        if not artifact_validation["pass"]:
            blockers.append("Stage 13 artifact write/read-back validation failed.")
            payload["ready_for_stage14"] = False
        write_json(json_path, payload)
        write_report(
            report_path,
            payload,
            input_validation,
            input_validation["cache_materialization"],
            artifact_validation,
            blockers,
            warnings,
            payload["ready_for_stage14"],
        )
        print(f"BEST_SILHOUETTE_K_OBSERVED={best_observed}")
        print(f"READY_FOR_STAGE_14={payload['ready_for_stage14']}")
        return 0 if payload["ready_for_stage14"] else 3
    except Exception as error:
        blockers.append(str(error))
        print(f"STAGE 13 FAILED CLOSED: {error}")
        write_json(
            json_path,
            {
                "stage": 13,
                "authority": "ENHANCED_KMEANS_K_SWEEP",
                "feature_authority": "IMPROVED_ERFM_FEATURE_SPACE",
                "ready_for_stage14": False,
                "blockers": blockers,
            },
        )
        report_path.write_text(
            "STAGE 13 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_14: False\n",
            encoding="utf-8",
        )
        return 3
    finally:
        if cached_df is not None:
            cached_df.unpersist()
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
