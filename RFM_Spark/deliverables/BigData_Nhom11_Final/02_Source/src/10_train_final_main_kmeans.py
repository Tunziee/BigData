"""Stage 10: train and lock the authoritative main K-Means model.

This stage consumes the locked Stage 9 K=4 authority.  It does not select K,
try alternative K values, assign business labels, or build another model.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pyspark.ml.clustering import KMeansModel
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
VECTOR_DIMENSION = 3
FEATURE_ORDER = ["Recency", "Frequency", "Monetary"]
REQUIRED_COLUMNS = [
    "customer_id",
    "Recency",
    "Frequency",
    "Monetary",
    "features_raw",
    "features",
]
SELECTED_K = 4
SEED = 42
DISTANCE_MEASURE = "euclidean"
INIT_MODE = "k-means||"
MAX_ITER = 100
TOL = 1e-4
CACHE_POLICY = "MEMORY_AND_DISK"
NUMERIC_TOLERANCE = 1e-8
POSITIVE_INFINITY = float("inf")
NEGATIVE_INFINITY = float("-inf")


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
    result = df.agg(*expressions).first().asDict()
    result["nan_total"] = sum(int(result[f"nan_{i}"] or 0) for i in range(VECTOR_DIMENSION))
    result["infinity_total"] = sum(
        int(result[f"infinity_{i}"] or 0) for i in range(VECTOR_DIMENSION)
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
        countDistinct("customer_id").alias("distinct_customer_id"),
        spark_sum(when(col("customer_id").isNull(), 1).otherwise(0)).alias(
            "null_customer_id"
        ),
    ).first().asDict()
    row["duplicate_customer_id"] = int(
        row["row_count"] - row["distinct_customer_id"]
    )
    row["features_quality"] = vector_quality(df, "features")
    row["features_raw_quality"] = vector_quality(df, "features_raw")
    row["pass"] = bool(
        row["row_count"] == EXPECTED_ROW_COUNT
        and row["distinct_customer_id"] == EXPECTED_ROW_COUNT
        and row["duplicate_customer_id"] == 0
        and row["null_customer_id"] == 0
        and row["features_quality"]["pass"]
        and row["features_raw_quality"]["pass"]
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
            len(center) == VECTOR_DIMENSION and all(finite(value) for value in center)
            for center in centers
        )
    )


def cluster_distribution(predictions: DataFrame) -> Dict[str, Any]:
    rows = predictions.groupBy("prediction").count().orderBy("prediction").collect()
    sizes = [
        {"cluster_id": int(row["prediction"]), "count": int(row["count"])}
        for row in rows
    ]
    total = sum(item["count"] for item in sizes)
    for item in sizes:
        item["ratio_pct"] = float(item["count"]) / float(total) * 100.0 if total else None
    return {
        "cluster_sizes": sizes,
        "cluster_count_sum": total,
        "num_produced_clusters": len(sizes),
        "pass": bool(
            total == EXPECTED_ROW_COUNT
            and len(sizes) == SELECTED_K
            and all(item["cluster_id"] in range(SELECTED_K) for item in sizes)
        ),
    }


def metric_close(final_value: float, reference_value: float) -> Dict[str, Any]:
    difference = abs(float(final_value) - float(reference_value))
    allowed = NUMERIC_TOLERANCE * max(1.0, abs(float(reference_value)))
    return {
        "final": float(final_value),
        "reference": float(reference_value),
        "absolute_difference": difference,
        "allowed_difference": allowed,
        "pass": difference <= allowed,
    }


def centers_close(
    final_centers: List[List[float]], reference_centers: List[List[float]]
) -> Dict[str, Any]:
    if len(final_centers) != len(reference_centers):
        return {
            "pass": False,
            "center_count_final": len(final_centers),
            "center_count_reference": len(reference_centers),
        }
    differences: List[float] = []
    for final_center, reference_center in zip(final_centers, reference_centers):
        if len(final_center) != len(reference_center):
            return {"pass": False, "reason": "center dimension mismatch"}
        differences.extend(
            abs(float(final_center[index]) - float(reference_center[index]))
            for index in range(len(final_center))
        )
    max_difference = max(differences) if differences else 0.0
    max_reference = max(
        [abs(float(value)) for center in reference_centers for value in center] or [1.0]
    )
    allowed = NUMERIC_TOLERANCE * max(1.0, max_reference)
    return {
        "pass": max_difference <= allowed,
        "max_absolute_difference": max_difference,
        "allowed_difference": allowed,
    }


def inverse_centers(
    centers: List[List[float]], means: List[float], stds: List[float]
) -> List[List[float]]:
    original = [
        [float(center[index]) * float(stds[index]) + float(means[index]) for index in range(3)]
        for center in centers
    ]
    if not all(
        finite(value) and value >= 0.0 for center in original for value in center
    ):
        raise ValueError("Inverse center validation failed: non-finite or negative RFM value.")
    return original


def audit_authorities(
    stage7_metadata_path: Path,
    stage8_path: Path,
    stage9_path: Path,
) -> Dict[str, Any]:
    blockers: List[str] = []
    try:
        stage7 = json.loads(stage7_metadata_path.read_text(encoding="utf-8"))
    except Exception as error:
        stage7 = {}
        blockers.append(f"Stage 7 scaler metadata unreadable: {error}")
    try:
        stage8 = json.loads(stage8_path.read_text(encoding="utf-8"))
    except Exception as error:
        stage8 = {}
        blockers.append(f"Stage 8 evidence unreadable: {error}")
    try:
        stage9 = json.loads(stage9_path.read_text(encoding="utf-8"))
    except Exception as error:
        stage9 = {}
        blockers.append(f"Stage 9 authority unreadable: {error}")

    if not (
        stage9.get("stage") == 9
        and stage9.get("model") == "Main RFM K-Means"
        and stage9.get("selected_k") == SELECTED_K
        and stage9.get("ready_for_stage10") is True
    ):
        blockers.append("Stage 9 authority mismatch: selected_k must be 4 and ready_for_stage10 must be true.")

    stage8_item = next(
        (item for item in stage8.get("experiments", []) if item.get("k") == SELECTED_K),
        None,
    )
    stage8_config = stage8.get("kmeans_config", {})
    expected_config = {
        "seed": SEED,
        "distanceMeasure": DISTANCE_MEASURE,
        "initMode": INIT_MODE,
        "maxIter": MAX_ITER,
        "tol": TOL,
    }
    if stage8_item is None or stage8_item.get("status") != "PASS":
        blockers.append("Stage 8 K=4 PASS evidence is missing.")
    for key, expected in expected_config.items():
        if stage8_config.get(key) != expected:
            blockers.append(f"Stage 8 configuration mismatch for {key}: {stage8_config.get(key)} != {expected}.")

    if stage7.get("input_features") != FEATURE_ORDER:
        blockers.append("Stage 7 feature order mismatch.")
    if stage7.get("withMean") is not True or stage7.get("withStd") is not True:
        blockers.append("Stage 7 scaler must have withMean=True and withStd=True.")
    if not isinstance(stage7.get("means"), list) or len(stage7["means"]) != 3:
        blockers.append("Stage 7 means must have length 3.")
    if not isinstance(stage7.get("stds"), list) or len(stage7["stds"]) != 3:
        blockers.append("Stage 7 stds must have length 3.")

    return {
        "pass": not blockers,
        "blockers": blockers,
        "stage7": stage7,
        "stage8": stage8,
        "stage8_k4": stage8_item,
        "stage9": stage9,
    }


def failure_artifacts(
    paths: Dict[str, Path],
    audit: Dict[str, Any],
    blockers: List[str],
) -> None:
    payload = {
        "stage": 10,
        "authority": "MAIN_KMEANS_FINAL",
        "source_authorities": {
            "stage7": str(paths["stage7_metadata"]),
            "stage8": str(paths["stage8"]),
            "stage9": str(paths["stage9"]),
        },
        "authority_audit": audit,
        "ready_for_stage11": False,
        "blockers": blockers,
    }
    write_json(paths["summary"], payload)
    paths["report"].write_text(
        "STAGE 10 FAILED CLOSED\n"
        + "\n".join(str(blocker) for blocker in blockers)
        + "\nREADY_FOR_STAGE_11: False\n",
        encoding="utf-8",
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "output" / "ml" / "main_rfm_scaled"
    stage7_metadata_path = project_root / "output" / "ml" / "main_rfm_scaler_metadata" / "scaler_metadata.json"
    stage8_path = project_root / "output" / "analysis" / "stage8" / "kmeans_k_sweep.json"
    stage9_path = project_root / "output" / "analysis" / "stage9" / "optimal_k_selection.json"
    model_path = project_root / "output" / "models" / "main_kmeans_final"
    model_metadata_path = project_root / "output" / "models" / "main_kmeans_final_metadata.json"
    prediction_path = project_root / "output" / "final" / "main_kmeans_predictions"
    analysis_dir = project_root / "output" / "analysis" / "stage10"
    report_path = project_root / "reports" / "stage10_final_main_kmeans.txt"
    paths = {
        "stage7_metadata": stage7_metadata_path,
        "stage8": stage8_path,
        "stage9": stage9_path,
        "summary": analysis_dir / "final_model_summary.json",
        "report": report_path,
    }
    for directory in [analysis_dir, model_path.parent, prediction_path.parent, report_path.parent]:
        directory.mkdir(parents=True, exist_ok=True)

    audit = audit_authorities(stage7_metadata_path, stage8_path, stage9_path)
    print(f"AUTHORITY_AUDIT_PASS={audit['pass']}")
    print(f"AUTHORITY_BLOCKERS={audit['blockers']}")
    if not audit["pass"]:
        failure_artifacts(paths, audit, audit["blockers"])
        return 3

    spark = (
        SparkSession.builder
        .master("local[4]")
        .appName("Stage10-Final-Main-KMeans")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.local.dir", str(project_root.parent / "spark_tmp"))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    enable_windows_java_filesystem_fallback(spark)

    cached_input: Optional[DataFrame] = None
    try:
        if not input_path.exists():
            raise FileNotFoundError(f"Missing Stage 7 input: {input_path}")
        source = spark.read.parquet(str(input_path))
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in source.columns]
        if missing_columns:
            raise ValueError(f"Missing required input columns: {missing_columns}")

        cached_input = source.select(*REQUIRED_COLUMNS).persist(StorageLevel.MEMORY_AND_DISK)
        cache_start = time.perf_counter()
        cache_row_count = cached_input.count()
        cache_runtime = time.perf_counter() - cache_start
        input_validation = validate_input(cached_input)
        input_validation["cache_materialization"] = {
            "storage_level": CACHE_POLICY,
            "materialized_row_count": cache_row_count,
            "materialization_runtime_sec": cache_runtime,
            "materialized_before_timed_fit": True,
        }
        print(f"INPUT_VALIDATION={input_validation}")
        if not input_validation["pass"]:
            raise ValueError("Stage 7 input validation failed; fail-closed.")

        model_input = cached_input.select("features")
        from pyspark.ml.clustering import KMeans

        kmeans = KMeans(
            k=SELECTED_K,
            seed=SEED,
            featuresCol="features",
            predictionCol="prediction",
            distanceMeasure=DISTANCE_MEASURE,
            initMode=INIT_MODE,
            maxIter=MAX_ITER,
            tol=TOL,
        )
        fit_start = time.perf_counter()
        fitted_model = kmeans.fit(model_input)
        fit_runtime = time.perf_counter() - fit_start
        summary = fitted_model.summary
        wcss = float(summary.trainingCost)
        num_iter = int(summary.numIter)

        prediction_df = fitted_model.transform(cached_input).select(
            "customer_id", "Recency", "Frequency", "Monetary", "prediction"
        )
        metric_predictions = fitted_model.transform(model_input).select("features", "prediction")
        evaluator = ClusteringEvaluator(
            predictionCol="prediction",
            featuresCol="features",
            metricName="silhouette",
            distanceMeasure="squaredEuclidean",
        )
        silhouette = float(evaluator.evaluate(metric_predictions))
        distribution = cluster_distribution(prediction_df)
        centers_scaled = centers_as_lists(fitted_model)
        if not finite(silhouette) or not (-1.0 <= silhouette <= 1.0):
            raise ValueError("Final silhouette is invalid.")
        if not finite(wcss) or wcss <= 0.0:
            raise ValueError("Final WCSS is invalid.")
        if not centers_valid(centers_scaled, SELECTED_K):
            raise ValueError("Final centers failed count/dimension/finiteness validation.")
        if not distribution["pass"]:
            raise ValueError("Final cluster distribution failed validation.")

        stage8_item = audit["stage8_k4"]
        reproducibility = {
            "tolerance": NUMERIC_TOLERANCE,
            "silhouette": metric_close(silhouette, float(stage8_item["silhouette"])),
            "wcss": metric_close(wcss, float(stage8_item["wcss"])),
            "num_iter": {
                "final": num_iter,
                "reference": int(stage8_item["num_iter"]),
                "pass": num_iter == int(stage8_item["num_iter"]),
            },
            "cluster_sizes": {
                "final": distribution["cluster_sizes"],
                "reference": stage8_item["cluster_sizes"],
                "pass": distribution["cluster_sizes"] == stage8_item["cluster_sizes"],
            },
            "centers_scaled": centers_close(centers_scaled, stage8_item["centers_scaled"]),
        }
        reproducibility["pass"] = bool(
            reproducibility["silhouette"]["pass"]
            and reproducibility["wcss"]["pass"]
            and reproducibility["num_iter"]["pass"]
            and reproducibility["cluster_sizes"]["pass"]
            and reproducibility["centers_scaled"]["pass"]
        )
        print(f"REPRODUCIBILITY_PASS={reproducibility['pass']}")
        if not reproducibility["pass"]:
            raise ValueError("REPRODUCIBILITY_WARNING: final result differs materially from Stage 8 K=4.")

        means = [float(value) for value in audit["stage7"]["means"]]
        stds = [float(value) for value in audit["stage7"]["stds"]]
        centers_original = inverse_centers(centers_scaled, means, stds)
        centers_validation = {
            "pass": bool(
                len(centers_original) == SELECTED_K
                and all(len(center) == VECTOR_DIMENSION for center in centers_original)
                and all(finite(value) and value >= 0.0 for center in centers_original for value in center)
            ),
            "feature_order": FEATURE_ORDER,
            "formula": "original = scaled * std + mean",
            "negative_values": False,
        }
        if not centers_validation["pass"]:
            raise ValueError("Inverse center validation failed.")

        print("START FINAL PREDICTION WRITE")
        prediction_df.write.mode("overwrite").parquet(str(prediction_path))
        prediction_read = spark.read.parquet(str(prediction_path))
        prediction_count = prediction_read.count()
        prediction_distinct = prediction_read.select("customer_id").distinct().count()
        prediction_null = prediction_read.where(col("prediction").isNull()).count()
        invalid_prediction = prediction_read.where(~col("prediction").isin(list(range(SELECTED_K)))).count()
        prediction_distribution = cluster_distribution(prediction_read)
        prediction_readback = {
            "row_count": prediction_count,
            "distinct_customer_id": prediction_distinct,
            "duplicate_customer_id": prediction_count - prediction_distinct,
            "prediction_null": prediction_null,
            "invalid_prediction": invalid_prediction,
            "distinct_predictions": prediction_read.select("prediction").distinct().count(),
            "cluster_count_sum": prediction_distribution["cluster_count_sum"],
            "pass": bool(
                prediction_count == EXPECTED_ROW_COUNT
                and prediction_distinct == EXPECTED_ROW_COUNT
                and prediction_count == prediction_distinct
                and prediction_null == 0
                and invalid_prediction == 0
                and prediction_distribution["num_produced_clusters"] == SELECTED_K
                and prediction_distribution["cluster_count_sum"] == EXPECTED_ROW_COUNT
            ),
        }
        if not prediction_readback["pass"]:
            raise ValueError("Final prediction read-back validation failed.")

        print("START FINAL MODEL WRITE")
        fitted_model.write().overwrite().save(str(model_path))
        reloaded_model = KMeansModel.load(str(model_path))
        reloaded_centers = centers_as_lists(reloaded_model)
        try:
            reloaded_k = int(reloaded_model.getK())
        except Exception:
            reloaded_k = None
        model_readback = {
            "load_pass": True,
            "k": reloaded_k,
            "center_count": len(reloaded_centers),
            "center_dimension": [len(center) for center in reloaded_centers],
            "centers_valid": centers_valid(reloaded_centers, SELECTED_K),
        }
        reloaded_prediction = reloaded_model.transform(cached_input).select(
            "customer_id", "prediction"
        ).withColumnRenamed("prediction", "prediction_reloaded")
        original_prediction = prediction_df.select(
            "customer_id", "prediction"
        ).withColumnRenamed("prediction", "prediction_original")
        mismatch_count = (
            original_prediction.join(reloaded_prediction, "customer_id")
            .where(col("prediction_original") != col("prediction_reloaded"))
            .count()
        )
        model_readback["prediction_mismatch_count"] = mismatch_count
        model_readback["prediction_equality_pass"] = mismatch_count == 0
        model_readback["pass"] = bool(
            model_readback["k"] == SELECTED_K
            and model_readback["center_count"] == SELECTED_K
            and model_readback["centers_valid"]
            and model_readback["prediction_equality_pass"]
        )
        if not model_readback["pass"]:
            raise ValueError("Final model read-back validation failed.")

        training = {
            "fit_runtime_sec": fit_runtime,
            "num_iter": num_iter,
            "wcss": wcss,
            "silhouette": silhouette,
        }
        config = {
            "k": SELECTED_K,
            "seed": SEED,
            "featuresCol": "features",
            "predictionCol": "prediction",
            "distanceMeasure": DISTANCE_MEASURE,
            "initMode": INIT_MODE,
            "maxIter": MAX_ITER,
            "tol": TOL,
        }
        final_summary = {
            "stage": 10,
            "authority": "MAIN_KMEANS_FINAL",
            "model": "Main RFM K-Means",
            "selected_k": SELECTED_K,
            "source_authorities": {
                "stage7": str(stage7_metadata_path),
                "stage8": str(stage8_path),
                "stage9": str(stage9_path),
            },
            "input_validation": input_validation,
            "configuration": config,
            "spark_configuration": {
                "master": "local[4]",
                "version": spark.version,
                "shuffle_partitions": 64,
                "adaptive_enabled": True,
                "cache_policy": CACHE_POLICY,
            },
            "cache_materialization": input_validation["cache_materialization"],
            "training": training,
            "cluster_distribution": distribution,
            "centers_scaled": centers_scaled,
            "centers_original": centers_original,
            "centers_validation": centers_validation,
            "reproducibility": reproducibility,
            "model_save_path": str(model_path),
            "model_readback": model_readback,
            "prediction_save_path": str(prediction_path),
            "prediction_readback": prediction_readback,
            "ready_for_stage11": True,
        }
        model_metadata = {
            "stage": 10,
            "authority": "MAIN_KMEANS_FINAL",
            "model": "Main RFM K-Means",
            "selected_k": SELECTED_K,
            "feature_order": FEATURE_ORDER,
            "input": {"path": str(input_path), "row_count": EXPECTED_ROW_COUNT},
            "kmeans_config": config,
            "training": training,
            "cluster_sizes": distribution["cluster_sizes"],
            "centers_scaled": centers_scaled,
            "reproducibility_vs_stage8": reproducibility,
        }
        write_json(model_metadata_path, model_metadata)
        write_json(analysis_dir / "final_model_summary.json", final_summary)
        write_json(
            analysis_dir / "final_cluster_sizes.json",
            {"stage": 10, "authority": "MAIN_KMEANS_FINAL", **distribution},
        )
        write_json(
            analysis_dir / "final_centers_scaled.json",
            {
                "stage": 10,
                "authority": "MAIN_KMEANS_FINAL",
                "feature_order": FEATURE_ORDER,
                "selected_k": SELECTED_K,
                "centers_scaled": centers_scaled,
            },
        )
        write_json(
            analysis_dir / "final_centers_original.json",
            {
                "stage": 10,
                "authority": "MAIN_KMEANS_FINAL",
                "feature_order": FEATURE_ORDER,
                "selected_k": SELECTED_K,
                "formula": "original = scaled * std + mean",
                "centers_original": centers_original,
                "validation": centers_validation,
            },
        )
        write_json(
            analysis_dir / "reproducibility_check.json",
            {
                "stage": 10,
                "reference": str(stage8_path),
                "selected_k": SELECTED_K,
                **reproducibility,
            },
        )

        report_lines = [
            "STAGE 10 — TRAIN & LOCK FINAL MAIN K-MEANS MODEL",
            "=" * 115,
            "CURRENT_STAGE: Stage 10",
            "VERIFIED_AUTHORITIES",
            f"Stage 7: {stage7_metadata_path}",
            f"Stage 8: {stage8_path}",
            f"Stage 9: {stage9_path}; selected_k=4; ready_for_stage10=True",
            "",
            "SELECTED_K_AUTHORITY: 4",
            f"SPARK_CONFIGURATION: {final_summary['spark_configuration']}",
            f"FINAL_KMEANS_CONFIGURATION: {config}",
            f"INPUT_VALIDATION: {input_validation}",
            f"TRAINING_RESULT: {training}",
            f"FINAL_SILHOUETTE: {silhouette}",
            f"FINAL_WCSS: {wcss}",
            f"FINAL_NUM_ITER: {num_iter}",
            f"FINAL_CLUSTER_DISTRIBUTION: {distribution}",
            f"FINAL_CENTERS_SCALED: {centers_scaled}",
            f"FINAL_CENTERS_ORIGINAL: {centers_original}",
            f"REPRODUCIBILITY_VS_STAGE8_K4: {reproducibility}",
            f"MODEL_SAVE_PATH: {model_path}",
            f"MODEL_READBACK_VALIDATION: {model_readback}",
            f"PREDICTIONS_PATH: {prediction_path}",
            f"PREDICTION_READBACK_VALIDATION: {prediction_readback}",
            "ARTIFACTS_CREATED:",
            f"- {report_path}",
            f"- {model_metadata_path}",
            f"- {analysis_dir / 'final_model_summary.json'}",
            f"- {analysis_dir / 'final_cluster_sizes.json'}",
            f"- {analysis_dir / 'final_centers_scaled.json'}",
            f"- {analysis_dir / 'final_centers_original.json'}",
            f"- {analysis_dir / 'reproducibility_check.json'}",
            "WARNINGS: No cluster names or marketing interpretation assigned. Stage 8 remains experiment evidence; Stage 9 remains K-selection authority.",
            "BLOCKERS: []",
            "READY_FOR_STAGE_11: True",
        ]
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        print(f"FINAL_SILHOUETTE={silhouette}")
        print(f"FINAL_WCSS={wcss}")
        print(f"FINAL_NUM_ITER={num_iter}")
        print("READY_FOR_STAGE_11=True")
        return 0
    except Exception as error:
        print(f"STAGE 10 FAILED CLOSED: {error}")
        failure_artifacts(paths, audit, [str(error)])
        return 3
    finally:
        if cached_input is not None:
            cached_input.unpersist()
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
