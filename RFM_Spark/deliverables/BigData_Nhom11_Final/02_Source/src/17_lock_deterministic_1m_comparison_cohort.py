"""Stage 17: lock the deterministic 1M comparison cohort and protocol.

This stage does not train or compare models.  It only audits the locked
authorities, materializes one customer-ID cohort, and records the future
three-model comparison contract.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import abs as spark_abs
from pyspark.sql.functions import col, concat, count, element_at, lit, max as spark_max, sha2, size
from pyspark.ml.functions import vector_to_array
from pyspark.storagelevel import StorageLevel


SOURCE_COUNT = 1_362_281
COHORT_COUNT = 1_000_000
SELECTION_SALT = "BIGDATA_NHOM11_STAGE17_COMPARISON_1M_V1"
MAIN_K = 4
ENHANCED_K = 5
HASH_HEX_LENGTH = 64
VECTOR_TOLERANCE = 1e-10
RAW_FEATURE_TOLERANCE = 1e-10
RAW_FEATURES = ["Recency", "Frequency", "Monetary", "AvgBasketSize", "ProductVariety"]
RFM_FEATURES = ["Recency", "Frequency", "Monetary"]
MAIN_FEATURE_ORDER = ["Recency", "Frequency", "Monetary"]
ENHANCED_FEATURE_ORDER = [
    "Recency_improved",
    "Frequency_log1p",
    "Monetary_log1p",
    "AvgBasketSize_log1p",
    "ProductVariety_log1p",
]


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, fields: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


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
        base = unsafe.staticFieldBase(windows_field)
        offset = unsafe.staticFieldOffset(windows_field)
        unsafe.putBoolean(base, offset, False)
    print("Windows Hadoop NativeIO fallback: enabled")


def validate_vector_column(df: DataFrame, vector_col: str, dimension: int) -> Dict[str, Any]:
    values = vector_to_array(col(vector_col))
    null_vector = df.where(col(vector_col).isNull()).count()
    wrong_dimension = df.where(col(vector_col).isNotNull() & (size(values) != dimension)).count()
    nan_total = 0
    infinity_total = 0
    for index in range(1, dimension + 1):
        value = element_at(values, index)
        nan_total += df.where(value != value).count()
        infinity_total += df.where((value == float("inf")) | (value == float("-inf"))).count()
    return {
        "column": vector_col,
        "dimension": dimension,
        "null_vector": null_vector,
        "wrong_dimension": wrong_dimension,
        "nan_total": nan_total,
        "infinity_total": infinity_total,
        "pass": null_vector == 0 and wrong_dimension == 0 and nan_total == 0 and infinity_total == 0,
    }


def audit_source(df: DataFrame, source_name: str, expected_prediction_ids: List[int] | None = None, vector_col: str | None = None, vector_dimension: int | None = None) -> Dict[str, Any]:
    row_count = df.count()
    distinct_customer_id = df.select("customer_id").distinct().count()
    result: Dict[str, Any] = {
        "source": source_name,
        "row_count": row_count,
        "distinct_customer_id": distinct_customer_id,
        "duplicate_customer_id": row_count - distinct_customer_id,
    }
    if expected_prediction_ids is not None:
        null_prediction = df.where(col("prediction").isNull()).count()
        invalid_prediction = df.where(~col("prediction").isin(expected_prediction_ids)).count()
        actual_ids = [int(row["prediction"]) for row in df.select("prediction").distinct().orderBy("prediction").collect()]
        result.update({"prediction_null": null_prediction, "invalid_prediction": invalid_prediction, "prediction_values": actual_ids, "prediction_pass": null_prediction == 0 and invalid_prediction == 0 and actual_ids == expected_prediction_ids})
    if vector_col is not None and vector_dimension is not None:
        result["vector_quality"] = validate_vector_column(df, vector_col, vector_dimension)
    result["pass"] = row_count == SOURCE_COUNT and distinct_customer_id == SOURCE_COUNT and result["duplicate_customer_id"] == 0 and result.get("prediction_pass", True) and result.get("vector_quality", {}).get("pass", True)
    return result


def distribution(df: DataFrame, prediction_col: str, expected_k: int, expected_total: int = COHORT_COUNT) -> Dict[str, Any]:
    rows = df.groupBy(prediction_col).count().orderBy(prediction_col).collect()
    total = sum(int(row["count"]) for row in rows)
    clusters = []
    for row in rows:
        cluster_id = int(row[prediction_col])
        count_value = int(row["count"])
        clusters.append({"cluster_id": cluster_id, "count": count_value, "ratio_pct": count_value / total * 100.0})
    return {"cluster_sizes": clusters, "total": total, "cluster_count": len(clusters), "prediction_values": [item["cluster_id"] for item in clusters], "pass": total == expected_total and len(clusters) == expected_k and [item["cluster_id"] for item in clusters] == list(range(expected_k))}


def canonical_fingerprint(df: DataFrame) -> str:
    digest = hashlib.sha256()
    for row in df.select("customer_id").orderBy("customer_id").toLocalIterator():
        digest.update(str(row["customer_id"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def source_id_summary(df: DataFrame, source_name: str) -> Dict[str, Any]:
    row_count = df.count()
    distinct_count = df.select("customer_id").distinct().count()
    return {"source": source_name, "row_count": row_count, "distinct_customer_id": distinct_count, "duplicate_customer_id": row_count - distinct_count}


def build_report(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "STAGE 17 — DETERMINISTIC 1M COMPARISON COHORT & THREE-MODEL PROTOCOL",
        "",
        f"COHORT_AUTHORITY: {summary.get('cohort_authority')}",
        f"PROTOCOL_AUTHORITY: {summary.get('protocol_authority')}",
        f"COHORT_COUNT: {summary.get('cohort_count')}",
        f"COHORT_CUSTOMER_ID_SHA256: {summary.get('cohort_customer_id_sha256')}",
        f"READY_FOR_STAGE_18: {summary.get('ready_for_stage18')}",
        "",
        "SELECTION: SHA256(salt || '||' || customer_id), ordered by selection_hash ASC then customer_id ASC.",
        f"SELECTION_SALT: {summary.get('selection_salt')}",
        "SELECTION_DEPENDS_ONLY_ON_CUSTOMER_ID: True",
        "MODEL_AWARE_SAMPLING: False",
        "",
        "SOURCE ALIGNMENT:",
        str(summary.get("source_alignment")),
        "",
        "COHORT DISTRIBUTION SANITY:",
        str(summary.get("cohort_distribution_sanity")),
        "",
        "RULE-BASED DEFINITION STATUS: " + str(summary.get("rule_based_definition_status")),
        "",
        "THREE REQUIRED MODELS: R1 Rule-based RFM, K1 Main K-Means, K2 Enhanced K-Means.",
        "COMMON EVALUATION SPACES: RFM_3D and ERFM_5D.",
        "NO FORMAL COMPARISON EXECUTED IN STAGE 17.",
        "NO GMM EXECUTED.",
    ]
    if summary.get("warnings"):
        lines.extend(["", "WARNINGS:"] + [f"- {item}" for item in summary["warnings"]])
    if summary.get("blockers"):
        lines.extend(["", "BLOCKERS:"] + [f"- {item}" for item in summary["blockers"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    main_feature_path = project_root / "output" / "ml" / "main_rfm_scaled"
    main_prediction_path = project_root / "output" / "final" / "main_kmeans_predictions"
    enhanced_feature_path = project_root / "output" / "ml" / "improved_erfm_scaled"
    enhanced_prediction_path = project_root / "output" / "final" / "enhanced_kmeans_predictions"
    stage7_summary_path = project_root / "output" / "analysis" / "stage7" / "main_rfm_pipeline_summary.json"
    stage7_scaler_path = project_root / "output" / "ml" / "main_rfm_scaler_metadata" / "scaler_metadata.json"
    stage10_summary_path = project_root / "output" / "analysis" / "stage10" / "final_model_summary.json"
    stage12_summary_path = project_root / "output" / "analysis" / "stage12" / "transformation_summary.json"
    stage12_scaler_path = project_root / "output" / "ml" / "improved_erfm_scaler_metadata" / "scaler_metadata.json"
    stage15_summary_path = project_root / "output" / "analysis" / "stage15" / "final_enhanced_model_summary.json"
    stage16_summary_path = project_root / "output" / "analysis" / "stage16" / "final_enhanced_cluster_profile_summary.json"
    output_dir = project_root / "output" / "comparison" / "stage17"
    ids_path = output_dir / "comparison_customer_ids_1m"
    cohort_path = output_dir / "comparison_cohort_1m"
    manifest_path = output_dir / "deterministic_1m_cohort_manifest.json"
    alignment_path = output_dir / "source_alignment_audit.json"
    sanity_path = output_dir / "cohort_distribution_sanity.json"
    protocol_path = output_dir / "three_model_comparison_protocol_v1.json"
    summary_path = output_dir / "stage17_final_summary.json"
    report_path = project_root / "reports" / "stage17_deterministic_1m_comparison_protocol.txt"
    run_log_path = project_root / "reports" / "stage17_run.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    blockers: List[str] = []
    warnings: List[str] = []
    spark = None
    persisted: List[DataFrame] = []

    try:
        stage7_summary = read_json(stage7_summary_path)
        stage7_scaler = read_json(stage7_scaler_path)
        stage10 = read_json(stage10_summary_path)
        stage12 = read_json(stage12_summary_path)
        stage12_scaler = read_json(stage12_scaler_path)
        stage15 = read_json(stage15_summary_path)
        stage16 = read_json(stage16_summary_path)

        if not (stage7_summary.get("stage") == 7 and stage7_summary.get("ready_for_stage8") is True and stage7_scaler.get("scaled_vector_col") == "features" if "scaled_vector_col" in stage7_scaler else stage7_scaler.get("scaled_vector_col") is None):
            # Stage 7 metadata names the fields vector_col/scaled_vector_col differently
            # across historical artifacts; the explicit checks below are authoritative.
            pass
        if not (stage7_summary.get("stage") == 7 and stage7_summary.get("ready_for_stage8") is True and stage7_summary.get("output", {}).get("vector_dimension") == 3 and stage7_summary.get("features") == MAIN_FEATURE_ORDER):
            blockers.append("Stage 7 Main feature authority invalid.")
        if not (stage7_scaler.get("scaled_vector_col") == "features" and stage7_scaler.get("withMean") is True and stage7_scaler.get("withStd") is True and [stage7_scaler.get("feature_order", {}).get(str(index)) for index in range(3)] == MAIN_FEATURE_ORDER):
            blockers.append("Stage 7 scaler metadata invalid.")
        if not (stage10.get("stage") == 10 and stage10.get("authority") == "MAIN_KMEANS_FINAL" and stage10.get("selected_k") == MAIN_K):
            blockers.append("Stage 10 Main final authority invalid.")
        if not (stage12.get("stage") == 12 and stage12.get("authority") == "IMPROVED_ERFM_FEATURE_SPACE" and stage12.get("ready_for_stage13") is True and stage12.get("feature_order") == ENHANCED_FEATURE_ORDER):
            blockers.append("Stage 12 Enhanced feature authority invalid.")
        if not (stage12_scaler.get("scaled_vector_col") == "features_improved" and stage12_scaler.get("scaled_vector_dimension") == 5 and stage12_scaler.get("withMean") is True and stage12_scaler.get("withStd") is True and [stage12_scaler.get("feature_order", {}).get(str(index)) for index in range(5)] == ENHANCED_FEATURE_ORDER):
            blockers.append("Stage 12 scaler metadata invalid.")
        if not (stage15.get("stage") == 15 and stage15.get("authority") == "ENHANCED_KMEANS_FINAL" and stage15.get("selected_k") == ENHANCED_K and stage15.get("ready_for_stage16") is True):
            blockers.append("Stage 15 Enhanced final authority invalid.")
        if not (stage16.get("stage") == 16 and stage16.get("authority") == "ENHANCED_KMEANS_FINAL_CLUSTER_PROFILE" and stage16.get("selected_k") == ENHANCED_K and stage16.get("ready_for_gmm_stage") is True):
            blockers.append("Stage 16 Enhanced profile authority invalid.")
        if blockers:
            failed = {"stage": 17, "cohort_authority": "DETERMINISTIC_1M_COMPARISON_COHORT", "protocol_authority": "THREE_MODEL_COMPARISON_PROTOCOL_V1", "ready_for_stage18": False, "blockers": blockers}
            write_json(summary_path, failed)
            report_path.write_text("STAGE 17 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_18: False\n", encoding="utf-8")
            return 3

        spark = (
            SparkSession.builder
            .master("local[4]")
            .appName("Stage17-Lock-Deterministic-1M-Cohort")
            .config("spark.sql.shuffle.partitions", "64")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.local.dir", str(project_root.parent / "spark_tmp"))
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        enable_windows_java_filesystem_fallback(spark)

        main_features = spark.read.parquet(str(main_feature_path)).select("customer_id", *RFM_FEATURES, "features").persist(StorageLevel.MEMORY_AND_DISK)
        main_predictions = spark.read.parquet(str(main_prediction_path)).select("customer_id", "prediction").persist(StorageLevel.MEMORY_AND_DISK)
        enhanced_features = spark.read.parquet(str(enhanced_feature_path)).select("customer_id", *RAW_FEATURES, "features_improved").persist(StorageLevel.MEMORY_AND_DISK)
        enhanced_predictions = spark.read.parquet(str(enhanced_prediction_path)).select("customer_id", "prediction").persist(StorageLevel.MEMORY_AND_DISK)
        persisted.extend([main_features, main_predictions, enhanced_features, enhanced_predictions])
        for df in persisted:
            df.count()

        source_audits = {
            "main_rfm_scaled": audit_source(main_features, "main_rfm_scaled", vector_col="features", vector_dimension=3),
            "main_kmeans_predictions": audit_source(main_predictions, "main_kmeans_predictions", expected_prediction_ids=list(range(MAIN_K))),
            "improved_erfm_scaled": audit_source(enhanced_features, "improved_erfm_scaled", vector_col="features_improved", vector_dimension=5),
            "enhanced_kmeans_predictions": audit_source(enhanced_predictions, "enhanced_kmeans_predictions", expected_prediction_ids=list(range(ENHANCED_K))),
        }
        if not all(item["pass"] for item in source_audits.values()):
            raise ValueError("One or more authoritative sources failed row/vector/prediction validation.")

        source_ids = {name: df.select("customer_id") for name, df in {"main_rfm_scaled": main_features, "main_kmeans_predictions": main_predictions, "improved_erfm_scaled": enhanced_features, "enhanced_kmeans_predictions": enhanced_predictions}.items()}
        anti_join_counts: Dict[str, int] = {}
        for left_name, right_name in combinations(source_ids.keys(), 2):
            anti_join_counts[f"{left_name}_minus_{right_name}"] = source_ids[left_name].join(source_ids[right_name], "customer_id", "left_anti").count()
            anti_join_counts[f"{right_name}_minus_{left_name}"] = source_ids[right_name].join(source_ids[left_name], "customer_id", "left_anti").count()

        joined_raw = main_features.select("customer_id", *RFM_FEATURES).alias("m").join(enhanced_features.select("customer_id", *RAW_FEATURES).alias("e"), "customer_id", "inner")
        raw_consistency: Dict[str, Any] = {}
        for feature in RFM_FEATURES:
            mismatch = joined_raw.where((col(f"m.{feature}").isNull() != col(f"e.{feature}").isNull()) | (spark_abs(col(f"m.{feature}") - col(f"e.{feature}")) > RAW_FEATURE_TOLERANCE)).count()
            max_diff = joined_raw.select(spark_max(spark_abs(col(f"m.{feature}") - col(f"e.{feature}"))).alias("max_diff")).first()["max_diff"]
            raw_consistency[feature] = {"mismatch_count": mismatch, "max_absolute_difference": float(max_diff or 0.0), "tolerance": RAW_FEATURE_TOLERANCE, "pass": mismatch == 0}

        source_alignment = {
            "sources": {name: {"path": str(path), **source_audits[name]} for name, path in {"main_rfm_scaled": main_feature_path, "main_kmeans_predictions": main_prediction_path, "improved_erfm_scaled": enhanced_feature_path, "enhanced_kmeans_predictions": enhanced_prediction_path}.items()},
            "anti_join_counts": anti_join_counts,
            "raw_feature_consistency_main_vs_enhanced": raw_consistency,
            "main_prediction_validation": source_audits["main_kmeans_predictions"],
            "enhanced_prediction_validation": source_audits["enhanced_kmeans_predictions"],
            "pass": all(value == 0 for value in anti_join_counts.values()) and all(item["pass"] for item in raw_consistency.values()) and all(item["pass"] for item in source_audits.values()),
        }
        if not source_alignment["pass"]:
            raise ValueError("Source universe alignment or raw feature consistency failed.")

        universe = main_features.select("customer_id").distinct()
        selection = universe.withColumn("selection_hash", sha2(concat(lit(SELECTION_SALT), lit("||"), col("customer_id").cast("string")), 256)).orderBy("selection_hash", "customer_id").limit(COHORT_COUNT).persist(StorageLevel.MEMORY_AND_DISK)
        persisted.append(selection)
        selection.count()
        selection_ids = selection.select("customer_id", "selection_hash")
        selection_hash_collision_groups = selection.groupBy("selection_hash").count().where(col("count") > 1).count()
        selection_hash_distinct = selection.select("selection_hash").distinct().count()

        recomputed_selection = universe.withColumn("selection_hash", sha2(concat(lit(SELECTION_SALT), lit("||"), col("customer_id").cast("string")), 256)).orderBy("selection_hash", "customer_id").limit(COHORT_COUNT)
        deterministic_rerun = {
            "recomputed_row_count": recomputed_selection.count(),
            "saved_minus_recomputed": selection_ids.select("customer_id").join(recomputed_selection.select("customer_id"), "customer_id", "left_anti").count(),
            "recomputed_minus_saved": recomputed_selection.select("customer_id").join(selection_ids.select("customer_id"), "customer_id", "left_anti").count(),
        }
        deterministic_rerun["pass"] = deterministic_rerun["recomputed_row_count"] == COHORT_COUNT and deterministic_rerun["saved_minus_recomputed"] == 0 and deterministic_rerun["recomputed_minus_saved"] == 0
        if not deterministic_rerun["pass"]:
            raise ValueError("Deterministic selection rerun mismatch.")

        fingerprint = canonical_fingerprint(selection_ids)
        selection_ids.write.mode("overwrite").parquet(str(ids_path))
        readback_ids = spark.read.parquet(str(ids_path)).select("customer_id", "selection_hash")
        ids_readback = {"row_count": readback_ids.count(), "distinct_customer_id": readback_ids.select("customer_id").distinct().count(), "duplicate_customer_id": readback_ids.count() - readback_ids.select("customer_id").distinct().count(), "selection_hash_null": readback_ids.where(col("selection_hash").isNull()).count()}
        ids_readback["pass"] = ids_readback["row_count"] == COHORT_COUNT and ids_readback["distinct_customer_id"] == COHORT_COUNT and ids_readback["duplicate_customer_id"] == 0 and ids_readback["selection_hash_null"] == 0
        if not ids_readback["pass"]:
            raise ValueError("Cohort ID authority read-back failed.")

        cohort = (
            selection_ids
            .join(enhanced_features.select("customer_id", "AvgBasketSize", "ProductVariety", "features_improved"), "customer_id", "inner")
            .join(main_features.select("customer_id", "Recency", "Frequency", "Monetary", "features"), "customer_id", "inner")
            .join(main_predictions.withColumnRenamed("prediction", "main_prediction"), "customer_id", "inner")
            .join(enhanced_predictions.withColumnRenamed("prediction", "enhanced_prediction"), "customer_id", "inner")
            .select("customer_id", "selection_hash", "Recency", "Frequency", "Monetary", "AvgBasketSize", "ProductVariety", col("main_prediction"), col("enhanced_prediction"), col("features").alias("eval_rfm3d_scaled"), col("features_improved").alias("eval_erfm5d_scaled"))
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        persisted.append(cohort)
        cohort.count()
        cohort_main = distribution(cohort, "main_prediction", MAIN_K)
        cohort_enhanced = distribution(cohort, "enhanced_prediction", ENHANCED_K)
        cohort_vector_3d = validate_vector_column(cohort, "eval_rfm3d_scaled", 3)
        cohort_vector_5d = validate_vector_column(cohort, "eval_erfm5d_scaled", 5)
        cohort_null_raw = sum(cohort.where(col(feature).isNull()).count() for feature in RAW_FEATURES)
        cohort_validation = {"row_count": cohort.count(), "distinct_customer_id": cohort.select("customer_id").distinct().count(), "duplicate_customer_id": cohort.count() - cohort.select("customer_id").distinct().count(), "missing_raw_feature_cells": cohort_null_raw, "main_prediction": cohort_main, "enhanced_prediction": cohort_enhanced, "eval_rfm3d_scaled": cohort_vector_3d, "eval_erfm5d_scaled": cohort_vector_5d}
        cohort_validation["pass"] = cohort_validation["row_count"] == COHORT_COUNT and cohort_validation["distinct_customer_id"] == COHORT_COUNT and cohort_validation["duplicate_customer_id"] == 0 and cohort_null_raw == 0 and cohort_main["pass"] and cohort_enhanced["pass"] and cohort_vector_3d["pass"] and cohort_vector_5d["pass"]
        if not cohort_validation["pass"]:
            raise ValueError("Self-contained comparison cohort validation failed.")
        cohort.write.mode("overwrite").parquet(str(cohort_path))
        readback_cohort = spark.read.parquet(str(cohort_path))
        expected_cohort_columns = ["customer_id", "selection_hash", "Recency", "Frequency", "Monetary", "AvgBasketSize", "ProductVariety", "main_prediction", "enhanced_prediction", "eval_rfm3d_scaled", "eval_erfm5d_scaled"]
        cohort_readback = {"columns_present": all(column in readback_cohort.columns for column in expected_cohort_columns), "row_count": readback_cohort.count(), "distinct_customer_id": readback_cohort.select("customer_id").distinct().count()}
        cohort_readback["duplicate_customer_id"] = cohort_readback["row_count"] - cohort_readback["distinct_customer_id"]
        cohort_readback["pass"] = cohort_readback["columns_present"] and cohort_readback["row_count"] == COHORT_COUNT and cohort_readback["distinct_customer_id"] == COHORT_COUNT and cohort_readback["duplicate_customer_id"] == 0
        if not cohort_readback["pass"]:
            raise ValueError("Comparison cohort Parquet read-back failed.")

        full_main = distribution(main_predictions, "prediction", MAIN_K, SOURCE_COUNT)
        full_enhanced = distribution(enhanced_predictions, "prediction", ENHANCED_K, SOURCE_COUNT)
        def ratio_sanity(full: Dict[str, Any], selected: Dict[str, Any]) -> Dict[str, Any]:
            full_map = {item["cluster_id"]: item["ratio_pct"] for item in full["cluster_sizes"]}
            selected_map = {item["cluster_id"]: item["ratio_pct"] for item in selected["cluster_sizes"]}
            rows = [{"cluster_id": cluster_id, "full_ratio_pct": full_map[cluster_id], "cohort_ratio_pct": selected_map[cluster_id], "difference_pp": selected_map[cluster_id] - full_map[cluster_id], "absolute_difference_pp": abs(selected_map[cluster_id] - full_map[cluster_id])} for cluster_id in sorted(full_map)]
            return {"full_data": full, "cohort_1m": selected, "ratio_difference": rows, "descriptive_only": True}
        distribution_sanity = {"main": ratio_sanity(full_main, cohort_main), "enhanced": ratio_sanity(full_enhanced, cohort_enhanced)}

        protocol = {
            "stage": 17,
            "authority": "THREE_MODEL_COMPARISON_PROTOCOL_V1",
            "cohort_authority": "DETERMINISTIC_1M_COMPARISON_COHORT",
            "cohort_size": COHORT_COUNT,
            "models": [
                {"id": "R1", "name": "Rule-based RFM", "type": "deterministic business-rule segmentation", "authority_status": "PENDING_STAGE18"},
                {"id": "K1", "name": "Main K-Means", "authority": "MAIN_KMEANS_FINAL", "k": MAIN_K, "training_feature_space": "standardized raw RFM 3D", "prediction_column": "main_prediction"},
                {"id": "K2", "name": "Enhanced K-Means", "authority": "ENHANCED_KMEANS_FINAL", "k": ENHANCED_K, "training_feature_space": "standardized transformed eRFM 5D", "prediction_column": "enhanced_prediction"},
            ],
            "common_evaluation_spaces": {"RFM_3D": {"column": "eval_rfm3d_scaled", "dimension": 3, "feature_order": MAIN_FEATURE_ORDER, "source_authority": "Stage 7 main_rfm_scaled"}, "ERFM_5D": {"column": "eval_erfm5d_scaled", "dimension": 5, "feature_order": ENHANCED_FEATURE_ORDER, "source_authority": "Stage 12 improved_erfm_scaled"}},
            "quality_metrics": ["silhouette_rfm3d", "silhouette_erfm5d", "cluster_balance", "normalized_cluster_entropy", "common_space_sse_per_customer", "pairwise_partition_agreement", "raw_profile_distinctness", "business_interpretability"],
            "agreement_metrics": ["ARI", "NMI"],
            "native_metric_cross_space_comparison_allowed": False,
            "rule_based_is_ground_truth": False,
            "classification_accuracy_allowed": False,
            "arbitrary_weighted_score_allowed": False,
            "gmm_in_mandatory_comparison": False,
            "runtime_track_separate": True,
            "ready_for_stage18": True,
        }
        protocol_validation = bool(protocol["cohort_size"] == COHORT_COUNT and len(protocol["models"]) == 3 and [item["id"] for item in protocol["models"]] == ["R1", "K1", "K2"] and protocol["gmm_in_mandatory_comparison"] is False and protocol["ready_for_stage18"] is True)

        source_alignment_payload = {**source_alignment, "authority_metadata": {"stage7": str(stage7_summary_path), "stage10": str(stage10_summary_path), "stage12": str(stage12_summary_path), "stage15": str(stage15_summary_path), "stage16": str(stage16_summary_path)}}
        manifest = {"stage": 17, "authority": "DETERMINISTIC_1M_COMPARISON_COHORT", "source_customer_count": SOURCE_COUNT, "cohort_customer_count": COHORT_COUNT, "selection_method": "SHA256_ORDER_TOP_N", "selection_salt": SELECTION_SALT, "ordering": ["selection_hash ASC", "customer_id ASC"], "selection_depends_only_on_customer_id": True, "model_aware_sampling": False, "cohort_customer_id_sha256": fingerprint, "fingerprint_algorithm": "SHA-256", "fingerprint_order": "customer_id ASC", "fingerprint_row_separator": "LF", "selection_hash_distinct_count": selection_hash_distinct, "selection_hash_collision_groups": selection_hash_collision_groups, "source_alignment_pass": source_alignment["pass"], "deterministic_rerun_pass": deterministic_rerun["pass"], "cohort_validation_pass": cohort_validation["pass"]}
        write_json(manifest_path, manifest)
        write_json(alignment_path, source_alignment_payload)
        write_json(sanity_path, distribution_sanity)
        write_json(protocol_path, protocol)

        summary: Dict[str, Any] = {
            "stage": 17,
            "cohort_authority": "DETERMINISTIC_1M_COMPARISON_COHORT",
            "protocol_authority": "THREE_MODEL_COMPARISON_PROTOCOL_V1",
            "source_customer_count": SOURCE_COUNT,
            "cohort_count": COHORT_COUNT,
            "cohort_customer_id_sha256": fingerprint,
            "selection_salt": SELECTION_SALT,
            "source_alignment": source_alignment,
            "source_alignment_pass": source_alignment["pass"],
            "deterministic_selection": {"method": "SHA256_ORDER_TOP_N", "ordering": ["selection_hash ASC", "customer_id ASC"], "selection_depends_only_on_customer_id": True, "model_aware_sampling": False, "collision_groups": selection_hash_collision_groups},
            "deterministic_rerun_validation": deterministic_rerun,
            "cohort_validation": cohort_validation,
            "cohort_readback": cohort_readback,
            "cohort_id_readback": ids_readback,
            "cohort_distribution_sanity": distribution_sanity,
            "protocol_validation_pass": protocol_validation,
            "rule_based_definition_status": "PENDING_STAGE18_AUTHORITY",
            "three_model_protocol": protocol,
            "quality_track": {"uses_locked_authoritative_assignments": True, "evaluation_spaces": ["RFM_3D", "ERFM_5D"], "formal_comparison_executed": False},
            "runtime_track": {"separate_from_quality_track": True, "controlled_1m_benchmark_executed": False},
            "warnings": warnings,
            "blockers": blockers,
            "ready_for_stage18": True,
        }
        write_json(summary_path, summary)
        build_report(report_path, summary)
        artifact_paths = [manifest_path, alignment_path, sanity_path, protocol_path, summary_path, report_path, run_log_path]
        artifact_readback: Dict[str, bool] = {}
        for path in artifact_paths:
            artifact_readback[str(path.relative_to(project_root))] = path.exists() and path.stat().st_size > 0
            if path.suffix == ".json" and artifact_readback[str(path.relative_to(project_root))]:
                try:
                    json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    artifact_readback[str(path.relative_to(project_root))] = False
        for path in [ids_path, cohort_path]:
            artifact_readback[str(path.relative_to(project_root))] = path.exists() and any(path.iterdir())
        summary["artifact_readback"] = artifact_readback
        summary["artifact_readback_pass"] = all(artifact_readback.values())
        summary["ready_for_stage18"] = bool(summary["ready_for_stage18"] and summary["artifact_readback_pass"])
        write_json(summary_path, summary)
        build_report(report_path, summary)
        if not summary["ready_for_stage18"]:
            raise ValueError("Stage 17 artifact read-back failed.")

        print("AUTHORITY_AUDIT_PASS=True")
        print(f"COHORT_FINGERPRINT={fingerprint}")
        print("SOURCE_ALIGNMENT_PASS=True")
        print("DETERMINISTIC_RERUN_PASS=True")
        print("COHORT_READBACK_PASS=True")
        print("PROTOCOL_VALIDATION_PASS=True")
        print("READY_FOR_STAGE_18=True")
        return 0
    except Exception as error:
        blockers.append(str(error))
        failed = {"stage": 17, "cohort_authority": "DETERMINISTIC_1M_COMPARISON_COHORT", "protocol_authority": "THREE_MODEL_COMPARISON_PROTOCOL_V1", "ready_for_stage18": False, "warnings": warnings, "blockers": blockers}
        write_json(summary_path, failed)
        report_path.write_text("STAGE 17 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_18: False\n", encoding="utf-8")
        print(f"STAGE 17 FAILED CLOSED: {error}")
        return 3
    finally:
        for df in persisted:
            try:
                df.unpersist()
            except Exception:
                pass
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    sys.exit(main())
