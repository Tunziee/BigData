"""Stage 15: train and lock the final Enhanced K-Means model.

This stage consumes the Stage 12 materialized feature authority and the
Stage 14 selected-K authority.  It does not tune, select another K, or run
business profiling.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, element_at, size
from pyspark.storagelevel import StorageLevel


EXPECTED_ROW_COUNT = 1_362_281
SELECTED_K = 5
VECTOR_DIMENSION = 5
FEATURE_COL = "features_improved"
PREDICTION_COL = "prediction"
FEATURE_ORDER = [
    "Recency_improved",
    "Frequency_log1p",
    "Monetary_log1p",
    "AvgBasketSize_log1p",
    "ProductVariety_log1p",
]
RAW_FEATURES = ["Recency", "Frequency", "Monetary", "AvgBasketSize", "ProductVariety"]
SEED = 42
DISTANCE_MEASURE = "euclidean"
INIT_MODE = "k-means||"
MAX_ITER = 100
TOL = 1e-4
NUMERIC_TOLERANCE = 1e-8


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        base = unsafe.staticFieldBase(windows_field)
        offset = unsafe.staticFieldOffset(windows_field)
        unsafe.putBoolean(base, offset, False)
    print("Windows Hadoop NativeIO fallback: enabled")


def centers_as_lists(model: Any) -> List[List[float]]:
    centers: List[List[float]] = []
    for center in model.clusterCenters():
        values = center.toArray().tolist() if hasattr(center, "toArray") else list(center)
        centers.append([float(value) for value in values])
    return centers


def vector_quality(df: DataFrame) -> Dict[str, Any]:
    values = vector_to_array(col(FEATURE_COL))
    null_vector = df.where(col(FEATURE_COL).isNull()).count()
    wrong_dimension = df.where(col(FEATURE_COL).isNotNull() & (size(values) != VECTOR_DIMENSION)).count()
    nan_total = 0
    infinity_total = 0
    for index in range(1, VECTOR_DIMENSION + 1):
        value = element_at(values, index)
        nan_total += df.where(value != value).count()
        infinity_total += df.where((value == float("inf")) | (value == float("-inf"))).count()
    return {
        "null_vector": null_vector,
        "wrong_dimension": wrong_dimension,
        "nan_total": nan_total,
        "infinity_total": infinity_total,
        "pass": null_vector == 0 and wrong_dimension == 0 and nan_total == 0 and infinity_total == 0,
    }


def validate_input(df: DataFrame) -> Dict[str, Any]:
    row_count = df.count()
    distinct_customer_id = df.select("customer_id").distinct().count()
    quality = vector_quality(df)
    return {
        "row_count": row_count,
        "distinct_customer_id": distinct_customer_id,
        "duplicate_customer_id": row_count - distinct_customer_id,
        "feature_quality": quality,
        "pass": row_count == EXPECTED_ROW_COUNT and distinct_customer_id == EXPECTED_ROW_COUNT and row_count == distinct_customer_id and quality["pass"],
    }


def cluster_distribution(predictions: DataFrame, expected_k: int) -> Dict[str, Any]:
    null_prediction = predictions.where(col(PREDICTION_COL).isNull()).count()
    rows = predictions.groupBy(PREDICTION_COL).count().orderBy(PREDICTION_COL).collect()
    sizes = [{"cluster_id": int(row[PREDICTION_COL]), "count": int(row["count"])} for row in rows]
    total = sum(item["count"] for item in sizes)
    for item in sizes:
        item["ratio_pct"] = item["count"] / total * 100.0
    ids = [item["cluster_id"] for item in sizes]
    counts = [item["count"] for item in sizes]
    ratios = [item["ratio_pct"] for item in sizes]
    expected_ids = list(range(expected_k))
    return {
        "cluster_sizes": sizes,
        "cluster_count_sum": total,
        "num_produced_clusters": len(sizes),
        "prediction_null_count": null_prediction,
        "prediction_values": ids,
        "min_cluster_count": min(counts) if counts else None,
        "max_cluster_count": max(counts) if counts else None,
        "min_cluster_ratio_pct": min(ratios) if ratios else None,
        "max_cluster_ratio_pct": max(ratios) if ratios else None,
        "pass": total == EXPECTED_ROW_COUNT and null_prediction == 0 and ids == expected_ids and len(sizes) == expected_k,
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


def centers_close(final_centers: List[List[float]], reference_centers: List[List[float]]) -> Dict[str, Any]:
    differences = [
        abs(final_centers[row][index] - reference_centers[row][index])
        for row in range(len(final_centers))
        for index in range(len(final_centers[row]))
    ]
    reference_max = max(abs(value) for center in reference_centers for value in center)
    maximum = max(differences) if differences else float("inf")
    allowed = NUMERIC_TOLERANCE * max(1.0, reference_max)
    return {"max_absolute_difference": maximum, "allowed_difference": allowed, "pass": maximum <= allowed}


def validate_centers(centers: List[List[float]], expected_k: int) -> Dict[str, Any]:
    return {
        "count": len(centers),
        "dimensions": [len(center) for center in centers],
        "all_finite": all(finite(value) for center in centers for value in center),
        "pass": len(centers) == expected_k and all(len(center) == VECTOR_DIMENSION for center in centers) and all(finite(value) for center in centers for value in center),
    }


def transformed_and_representative_centers(centers: List[List[float]], means: List[float], stds: List[float]) -> List[Dict[str, Any]]:
    rows = []
    for cluster_id, center in enumerate(centers):
        transformed = [center[index] * stds[index] + means[index] for index in range(VECTOR_DIMENSION)]
        representative = [transformed[0]] + [math.expm1(transformed[index]) for index in range(1, VECTOR_DIMENSION)]
        rows.append({
            "cluster_id": cluster_id,
            "scaled_center": center,
            "transformed_center": transformed,
            "representative_raw_scale": representative,
            "interpretation_note": "Representative values implied by transformed-space centroids; not raw cluster arithmetic means.",
        })
    return rows


def write_report(path: Path, summary: Dict[str, Any]) -> None:
    metrics = summary.get("metrics", {})
    lines = [
        "STAGE 15 — TRAIN & LOCK FINAL ENHANCED K-MEANS MODEL",
        "",
        "AUTHORITY: ENHANCED_KMEANS_FINAL",
        f"SELECTED_K: {summary.get('selected_k')}",
        f"READY_FOR_STAGE_16: {summary.get('ready_for_stage16')}",
        "",
        "FINAL MODEL: K=5, seed=42, euclidean, k-means||, maxIter=100, tol=1e-4",
        f"SILHOUETTE: {metrics.get('silhouette')}",
        f"WCSS: {metrics.get('wcss')}",
        f"NUM_ITER: {metrics.get('num_iter')}",
        f"FIT_RUNTIME_SEC: {metrics.get('fit_runtime_sec')}",
        "",
        "REPRESENTATIVE CENTER NOTE: Representative values implied by transformed-space centroids; not raw cluster arithmetic means.",
        "NO BUSINESS SEGMENT NAMES: True",
        "NO GMM: True",
        "NO MAIN-VS-ENHANCED CONCLUSION: True",
        "",
        f"MODEL_READBACK_PASS: {summary.get('model_readback', {}).get('pass')}",
        f"PREDICTION_READBACK_PASS: {summary.get('prediction_readback', {}).get('pass')}",
        f"REPRODUCIBILITY_PASS: {summary.get('reproducibility', {}).get('pass')}",
    ]
    if summary.get("warnings"):
        lines.extend(["", "WARNINGS:"] + [f"- {item}" for item in summary["warnings"]])
    if summary.get("blockers"):
        lines.extend(["", "BLOCKERS:"] + [f"- {item}" for item in summary["blockers"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "output" / "ml" / "improved_erfm_scaled"
    stage12_path = project_root / "output" / "analysis" / "stage12" / "transformation_summary.json"
    scaler_path = project_root / "output" / "ml" / "improved_erfm_scaler_metadata" / "scaler_metadata.json"
    stage13_path = project_root / "output" / "analysis" / "stage13" / "enhanced_kmeans_k_sweep.json"
    stage14_path = project_root / "output" / "analysis" / "stage14" / "enhanced_optimal_k_selection.json"
    analysis_dir = project_root / "output" / "analysis" / "stage15"
    models_dir = project_root / "output" / "models"
    final_dir = project_root / "output" / "final"
    model_path = models_dir / "enhanced_kmeans_final"
    predictions_path = final_dir / "enhanced_kmeans_predictions"
    metadata_path = models_dir / "enhanced_kmeans_final_metadata.json"
    summary_path = analysis_dir / "final_enhanced_model_summary.json"
    report_path = project_root / "reports" / "stage15_final_enhanced_kmeans.txt"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    blockers: List[str] = []
    warnings: List[str] = []
    spark: Optional[SparkSession] = None
    cached_df: Optional[DataFrame] = None

    try:
        stage12 = read_json(stage12_path)
        scaler = read_json(scaler_path)
        stage13 = read_json(stage13_path)
        stage14 = read_json(stage14_path)

        if not (stage12.get("stage") == 12 and stage12.get("authority") == "IMPROVED_ERFM_FEATURE_SPACE" and stage12.get("ready_for_stage13") is True):
            blockers.append("Stage 12 authority invalid.")
        if not (scaler.get("scaled_vector_col") == FEATURE_COL and scaler.get("scaled_vector_dimension") == VECTOR_DIMENSION and scaler.get("withMean") is True and scaler.get("withStd") is True):
            blockers.append("Stage 12 scaler metadata invalid.")
        scaler_order = [scaler.get("feature_order", {}).get(str(index)) for index in range(VECTOR_DIMENSION)]
        if scaler_order != FEATURE_ORDER:
            blockers.append("Stage 12 feature order mismatch.")
        experiments = {int(item.get("k")): item for item in stage13.get("experiments", [])}
        if not (stage13.get("stage") == 13 and stage13.get("authority") == "ENHANCED_KMEANS_K_SWEEP"):
            blockers.append("Stage 13 authority invalid.")
        if 5 not in experiments or experiments[5].get("status") != "PASS" or experiments[5].get("validation_pass") is not True:
            blockers.append("Stage 13 K=5 evidence invalid.")
        if not (stage14.get("stage") == 14 and stage14.get("authority") == "ENHANCED_KMEANS_OPTIMAL_K" and stage14.get("selected_k") == SELECTED_K and stage14.get("ready_for_stage15") is True):
            blockers.append("Stage 14 selected-K authority invalid.")
        if blockers:
            failed = {"stage": 15, "authority": "ENHANCED_KMEANS_FINAL", "ready_for_stage16": False, "blockers": blockers}
            write_json(summary_path, failed)
            report_path.write_text("STAGE 15 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_16: False\n", encoding="utf-8")
            return 3

        reference = experiments[SELECTED_K]
        configuration = {
            "k": SELECTED_K,
            "seed": SEED,
            "featuresCol": FEATURE_COL,
            "predictionCol": PREDICTION_COL,
            "distanceMeasure": DISTANCE_MEASURE,
            "initMode": INIT_MODE,
            "maxIter": MAX_ITER,
            "tol": TOL,
        }
        spark = (
            SparkSession.builder
            .master("local[4]")
            .appName("Stage15-Train-Final-Enhanced-KMeans")
            .config("spark.sql.shuffle.partitions", "64")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.local.dir", str(project_root.parent / "spark_tmp"))
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        enable_windows_java_filesystem_fallback(spark)

        source = spark.read.parquet(str(input_path))
        required = ["customer_id"] + RAW_FEATURES + FEATURE_ORDER + ["features_improved_raw", FEATURE_COL]
        missing = [column for column in required if column not in source.columns]
        if missing:
            raise ValueError(f"Missing Stage 12 columns: {missing}")
        cached_df = source.select(*required).persist(StorageLevel.MEMORY_AND_DISK)
        cache_start = time.perf_counter()
        cached_df.count()
        cache_runtime = time.perf_counter() - cache_start
        input_validation = validate_input(cached_df)
        print(f"INPUT_VALIDATION={input_validation}")
        if not input_validation["pass"]:
            raise ValueError("Stage 12 input validation failed.")

        model_input = cached_df.select(FEATURE_COL)
        model = KMeans(
            k=SELECTED_K,
            seed=SEED,
            featuresCol=FEATURE_COL,
            predictionCol=PREDICTION_COL,
            distanceMeasure=DISTANCE_MEASURE,
            initMode=INIT_MODE,
            maxIter=MAX_ITER,
            tol=TOL,
        )
        fit_start = time.perf_counter()
        final_model = model.fit(model_input)
        fit_runtime = time.perf_counter() - fit_start
        centers = centers_as_lists(final_model)
        center_validation = validate_centers(centers, SELECTED_K)
        if not center_validation["pass"]:
            raise ValueError("Final center validation failed.")

        transformed = final_model.transform(cached_df)
        prediction_df = transformed.select("customer_id", *RAW_FEATURES, PREDICTION_COL).persist(StorageLevel.MEMORY_AND_DISK)
        prediction_df.count()
        distribution = cluster_distribution(prediction_df, SELECTED_K)
        if not distribution["pass"]:
            raise ValueError("Final cluster distribution validation failed.")

        evaluator = ClusteringEvaluator(featuresCol=FEATURE_COL, predictionCol=PREDICTION_COL, metricName="silhouette", distanceMeasure="squaredEuclidean")
        silhouette = float(evaluator.evaluate(transformed.select(FEATURE_COL, PREDICTION_COL)))
        wcss = float(final_model.summary.trainingCost)
        num_iter = int(final_model.summary.numIter)
        if not finite(silhouette) or silhouette < -1.0 or silhouette > 1.0 or not finite(wcss) or not finite(num_iter):
            raise ValueError("Final metric validation failed.")

        reproducibility = {
            "silhouette": metric_close(silhouette, float(reference["silhouette"])),
            "wcss": metric_close(wcss, float(reference["wcss_trainingCost"])),
            "num_iter": {"final": num_iter, "reference": int(reference["num_iter"]), "pass": num_iter == int(reference["num_iter"])},
            "cluster_sizes": {"final": distribution["cluster_sizes"], "reference": reference["cluster_sizes"], "pass": distribution["cluster_sizes"] == reference["cluster_sizes"]},
            "centers_scaled": centers_close(centers, reference["centers_scaled_improved_5d"]),
        }
        reproducibility["pass"] = all([
            reproducibility["silhouette"]["pass"],
            reproducibility["wcss"]["pass"],
            reproducibility["num_iter"]["pass"],
            reproducibility["cluster_sizes"]["pass"],
            reproducibility["centers_scaled"]["pass"],
        ])
        if not reproducibility["pass"]:
            warnings.append("REPRODUCIBILITY_WARNING: final result materially differs from Stage 13 K=5.")
            raise ValueError("Stage 15 reproducibility check failed.")

        means = [float(value) for value in scaler["means"]]
        stds = [float(value) for value in scaler["stds"]]
        center_rows = transformed_and_representative_centers(centers, means, stds)
        transformed_centers = [{"cluster_id": row["cluster_id"], "values": row["transformed_center"]} for row in center_rows]
        representative_centers = [{"cluster_id": row["cluster_id"], "values": row["representative_raw_scale"] if "representative_raw_scale" in row else row["representative_raw_scale"]} for row in center_rows]

        prediction_df.write.mode("overwrite").parquet(str(predictions_path))
        final_model.write().overwrite().save(str(model_path))

        loaded_model = KMeansModel.load(str(model_path))
        loaded_centers = centers_as_lists(loaded_model)
        model_readback = {
            "center_validation": validate_centers(loaded_centers, SELECTED_K),
            "center_count": len(loaded_centers),
            "center_dimension": len(loaded_centers[0]) if loaded_centers else 0,
        }
        loaded_predictions = loaded_model.transform(cached_df).select("customer_id", col(PREDICTION_COL).alias("loaded_prediction"))
        mismatches = (
            prediction_df.select("customer_id", PREDICTION_COL)
            .join(loaded_predictions, on="customer_id", how="inner")
            .where(col(PREDICTION_COL) != col("loaded_prediction"))
            .count()
        )
        model_readback["prediction_mismatch_count"] = mismatches
        model_readback["pass"] = bool(model_readback["center_validation"]["pass"] and mismatches == 0)
        if not model_readback["pass"]:
            raise ValueError("Reloaded model validation failed.")

        readback_predictions = spark.read.parquet(str(predictions_path)).select("customer_id", *RAW_FEATURES, PREDICTION_COL)
        readback_distribution = cluster_distribution(readback_predictions, SELECTED_K)
        prediction_readback = {
            "row_count": readback_predictions.count(),
            "distinct_customer_id": readback_predictions.select("customer_id").distinct().count(),
            "duplicate_customer_id": readback_predictions.count() - readback_predictions.select("customer_id").distinct().count(),
            "distribution": readback_distribution,
            "counts_match_pre_write": readback_distribution["cluster_sizes"] == distribution["cluster_sizes"],
        }
        prediction_readback["pass"] = bool(
            prediction_readback["row_count"] == EXPECTED_ROW_COUNT
            and prediction_readback["distinct_customer_id"] == EXPECTED_ROW_COUNT
            and prediction_readback["duplicate_customer_id"] == 0
            and readback_distribution["pass"]
            and prediction_readback["counts_match_pre_write"]
        )
        if not prediction_readback["pass"]:
            raise ValueError("Prediction Parquet read-back validation failed.")

        centers_scaled_payload = {"feature_space": "standardized improved 5D", "feature_order": FEATURE_ORDER, "centers": [{"cluster_id": index, "values": center} for index, center in enumerate(centers)]}
        centers_transformed_payload = {"feature_order": ["Recency", "log1p(Frequency)", "log1p(Monetary)", "log1p(AvgBasketSize)", "log1p(ProductVariety)"], "scaler_means": means, "scaler_stds": stds, "centers": transformed_centers}
        centers_representative_payload = {"feature_order": RAW_FEATURES, "centers": [{"cluster_id": row["cluster_id"], "values": row["representative_raw_scale"], "interpretation_note": row["interpretation_note"]} for row in center_rows]}
        write_json(analysis_dir / "final_cluster_sizes.json", distribution)
        write_json(analysis_dir / "final_centers_scaled.json", centers_scaled_payload)
        write_json(analysis_dir / "final_centers_transformed.json", centers_transformed_payload)
        write_json(analysis_dir / "final_centers_representative_raw.json", centers_representative_payload)
        write_json(analysis_dir / "reproducibility_check.json", reproducibility)

        metadata = {
            "stage": 15,
            "authority": "ENHANCED_KMEANS_FINAL",
            "model": "Enhanced K-Means",
            "selected_k": SELECTED_K,
            "feature_authority": "IMPROVED_ERFM_FEATURE_SPACE",
            "feature_order": FEATURE_ORDER,
            "transformations": {"Recency": "identity", "Frequency": "log1p", "Monetary": "log1p", "AvgBasketSize": "log1p", "ProductVariety": "log1p"},
            "configuration": configuration,
            "training": {"cache_materialization_runtime_sec": cache_runtime, "fit_runtime_sec": fit_runtime, "silhouette": silhouette, "wcss": wcss, "num_iter": num_iter},
            "cluster_distribution": distribution["cluster_sizes"],
            "centers_scaled": centers,
            "reproducibility_vs_stage13_k5": reproducibility,
            "representative_center_note": "Representative values implied by transformed-space centroids; not raw cluster arithmetic means.",
        }
        write_json(metadata_path, metadata)

        summary: Dict[str, Any] = {
            "stage": 15,
            "authority": "ENHANCED_KMEANS_FINAL",
            "selected_k": SELECTED_K,
            "source_authorities": {"stage12": "IMPROVED_ERFM_FEATURE_SPACE", "stage13": "ENHANCED_KMEANS_K_SWEEP", "stage14": "ENHANCED_KMEANS_OPTIMAL_K"},
            "input_validation": input_validation,
            "cache_materialization_runtime_sec": cache_runtime,
            "configuration": configuration,
            "metrics": {"silhouette": silhouette, "wcss": wcss, "num_iter": num_iter, "fit_runtime_sec": fit_runtime},
            "cluster_distribution": distribution,
            "centers_scaled": centers_scaled_payload,
            "centers_transformed": centers_transformed_payload,
            "centers_representative_raw": centers_representative_payload,
            "reproducibility": reproducibility,
            "model_save_path": str(model_path),
            "prediction_save_path": str(predictions_path),
            "model_readback": model_readback,
            "prediction_readback": prediction_readback,
            "no_business_labels": True,
            "no_main_vs_enhanced_conclusion": True,
            "no_gmm": True,
            "warnings": warnings,
            "blockers": blockers,
            "ready_for_stage16": True,
        }
        artifact_paths = [
            metadata_path,
            predictions_path,
            model_path,
            analysis_dir / "final_enhanced_model_summary.json",
            analysis_dir / "final_cluster_sizes.json",
            analysis_dir / "final_centers_scaled.json",
            analysis_dir / "final_centers_transformed.json",
            analysis_dir / "final_centers_representative_raw.json",
            analysis_dir / "reproducibility_check.json",
            report_path,
        ]
        # Write the summary once before report generation so the summary itself
        # is included in the artifact read-back contract.
        write_json(summary_path, summary)
        write_report(report_path, summary)
        artifact_readback = {}
        for path in artifact_paths:
            if path.is_dir():
                artifact_readback[str(path.relative_to(project_root))] = any(path.iterdir())
            else:
                artifact_readback[str(path.relative_to(project_root))] = path.exists() and path.stat().st_size > 0
        summary["artifact_readback"] = artifact_readback
        summary["ready_for_stage16"] = all(artifact_readback.values())
        write_json(summary_path, summary)
        write_report(report_path, summary)
        if not summary["ready_for_stage16"]:
            raise ValueError("Stage 15 artifact read-back failed.")
        print("AUTHORITY_AUDIT_PASS=True")
        print(f"FINAL_SILHOUETTE={silhouette}")
        print(f"FINAL_WCSS={wcss}")
        print(f"FINAL_NUM_ITER={num_iter}")
        print("MODEL_RELOAD_PREDICTION_MISMATCH_COUNT=0")
        print("READY_FOR_STAGE_16=True")
        prediction_df.unpersist()
        return 0
    except Exception as error:
        blockers.append(str(error))
        failed = {"stage": 15, "authority": "ENHANCED_KMEANS_FINAL", "ready_for_stage16": False, "blockers": blockers, "warnings": warnings}
        write_json(summary_path, failed)
        report_path.write_text("STAGE 15 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_16: False\n", encoding="utf-8")
        print(f"STAGE 15 FAILED CLOSED: {error}")
        return 3
    finally:
        if cached_df is not None:
            cached_df.unpersist()
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    sys.exit(main())
