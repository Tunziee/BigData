"""Stage 18: build and lock the project-owned Rule-based RFM baseline."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import avg, col, count, lit, max as spark_max, min as spark_min, stddev_samp, when
from pyspark.storagelevel import StorageLevel


EXPECTED_COUNT = 1_000_000
EXPECTED_K = 5
COHORT_FINGERPRINT = "ca082ec1d828df9c34a10105ae11599c3cc0653dc5b4d2c11af18ec82b781cf5"
QUANTILE_PROBABILITIES = [0.2, 0.4, 0.6, 0.8]
QUANTILE_RELATIVE_ERROR = 0.0
PROFILE_RELATIVE_ERROR = 0.001
RAW_RFM = ["Recency", "Frequency", "Monetary"]
RAW_PROFILE_FEATURES = ["Recency", "Frequency", "Monetary", "AvgBasketSize", "ProductVariety"]
REQUIRED_INPUT_COLUMNS = [
    "customer_id", "selection_hash", "Recency", "Frequency", "Monetary", "AvgBasketSize", "ProductVariety",
    "main_prediction", "enhanced_prediction", "eval_rfm3d_scaled", "eval_erfm5d_scaled",
]
SEGMENT_NAMES = {
    0: {"name_vi": "RFM rất thấp", "name_en": "Very Low RFM", "score_range": "3-5"},
    1: {"name_vi": "RFM thấp", "name_en": "Low RFM", "score_range": "6-7"},
    2: {"name_vi": "RFM trung bình", "name_en": "Medium RFM", "score_range": "8-10"},
    3: {"name_vi": "RFM cao", "name_en": "High RFM", "score_range": "11-12"},
    4: {"name_vi": "RFM rất cao", "name_en": "Very High RFM", "score_range": "13-15"},
}


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


def input_validation(df: DataFrame) -> Dict[str, Any]:
    row_count = df.count()
    distinct_customer_id = df.select("customer_id").distinct().count()
    nulls = {feature: df.where(col(feature).isNull()).count() for feature in RAW_RFM}
    non_finite = {feature: df.where(~col(feature).cast("double").isNotNull()).count() for feature in RAW_RFM}
    negative_or_zero = {
        "Recency_lt_0": df.where(col("Recency") < 0).count(),
        "Frequency_le_0": df.where(col("Frequency") <= 0).count(),
        "Monetary_le_0": df.where(col("Monetary") <= 0).count(),
    }
    # Spark SQL's isNotNull does not identify NaN; explicitly use value != value.
    nan_counts = {feature: df.where(col(feature) != col(feature)).count() for feature in RAW_RFM}
    pass_value = (
        row_count == EXPECTED_COUNT
        and distinct_customer_id == EXPECTED_COUNT
        and row_count == distinct_customer_id
        and all(value == 0 for value in nulls.values())
        and all(value == 0 for value in non_finite.values())
        and all(value == 0 for value in nan_counts.values())
        and all(value == 0 for value in negative_or_zero.values())
    )
    return {"row_count": row_count, "distinct_customer_id": distinct_customer_id, "duplicate_customer_id": row_count - distinct_customer_id, "nulls": nulls, "nan_counts": nan_counts, "non_finite": non_finite, "invalid_domain_counts": negative_or_zero, "pass": pass_value}


def quantile_thresholds(df: DataFrame) -> Dict[str, List[float]]:
    return {feature: [float(value) for value in df.approxQuantile(feature, QUANTILE_PROBABILITIES, QUANTILE_RELATIVE_ERROR)] for feature in RAW_RFM}


def score_columns(df: DataFrame, thresholds: Dict[str, List[float]]) -> DataFrame:
    r20, r40, r60, r80 = thresholds["Recency"]
    f20, f40, f60, f80 = thresholds["Frequency"]
    m20, m40, m60, m80 = thresholds["Monetary"]
    r_score = when(col("Recency") <= lit(r20), 5).when(col("Recency") <= lit(r40), 4).when(col("Recency") <= lit(r60), 3).when(col("Recency") <= lit(r80), 2).otherwise(1)
    f_score = when(col("Frequency") <= lit(f20), 1).when(col("Frequency") <= lit(f40), 2).when(col("Frequency") <= lit(f60), 3).when(col("Frequency") <= lit(f80), 4).otherwise(5)
    m_score = when(col("Monetary") <= lit(m20), 1).when(col("Monetary") <= lit(m40), 2).when(col("Monetary") <= lit(m60), 3).when(col("Monetary") <= lit(m80), 4).otherwise(5)
    total = col("r_score") + col("f_score") + col("m_score")
    return (
        df.withColumn("r_score", r_score)
        .withColumn("f_score", f_score)
        .withColumn("m_score", m_score)
        .withColumn("rfm_total_score", col("r_score") + col("f_score") + col("m_score"))
        .withColumn("rule_segment_id", when(col("rfm_total_score") <= 5, 0).when(col("rfm_total_score") <= 7, 1).when(col("rfm_total_score") <= 10, 2).when(col("rfm_total_score") <= 12, 3).otherwise(4))
        .withColumn("rule_segment_name_vi", when(col("rule_segment_id") == 0, SEGMENT_NAMES[0]["name_vi"]).when(col("rule_segment_id") == 1, SEGMENT_NAMES[1]["name_vi"]).when(col("rule_segment_id") == 2, SEGMENT_NAMES[2]["name_vi"]).when(col("rule_segment_id") == 3, SEGMENT_NAMES[3]["name_vi"]).otherwise(SEGMENT_NAMES[4]["name_vi"]))
        .withColumn("rule_segment_name_en", when(col("rule_segment_id") == 0, SEGMENT_NAMES[0]["name_en"]).when(col("rule_segment_id") == 1, SEGMENT_NAMES[1]["name_en"]).when(col("rule_segment_id") == 2, SEGMENT_NAMES[2]["name_en"]).when(col("rule_segment_id") == 3, SEGMENT_NAMES[3]["name_en"]).otherwise(SEGMENT_NAMES[4]["name_en"]))
    )


def score_distribution(df: DataFrame, column: str) -> List[Dict[str, Any]]:
    rows = df.groupBy(column).count().orderBy(column).collect()
    return [{"value": int(row[column]), "count": int(row["count"]), "ratio_pct": int(row["count"]) / EXPECTED_COUNT * 100.0} for row in rows]


def segment_distribution(df: DataFrame) -> Dict[str, Any]:
    rows = df.groupBy("rule_segment_id").count().orderBy("rule_segment_id").collect()
    values = [{"segment_id": int(row["rule_segment_id"]), "count": int(row["count"]), "ratio_pct": int(row["count"]) / EXPECTED_COUNT * 100.0, **SEGMENT_NAMES[int(row["rule_segment_id"])]} for row in rows]
    missing = [segment_id for segment_id in range(EXPECTED_K) if segment_id not in {item["segment_id"] for item in values}]
    return {"segments": values, "missing_segment_ids": missing, "pass": sum(item["count"] for item in values) == EXPECTED_COUNT and not missing and [item["segment_id"] for item in values] == list(range(EXPECTED_K))}


def scalar_profile(df: DataFrame, feature: str) -> Dict[str, Any]:
    row = df.agg(count(feature).alias("count"), avg(feature).alias("mean"), stddev_samp(feature).alias("std"), spark_min(feature).alias("min"), spark_max(feature).alias("max")).first().asDict()
    quantiles = df.approxQuantile(feature, [0.05, 0.25, 0.50, 0.75, 0.95], PROFILE_RELATIVE_ERROR)
    return {"count": int(row["count"]), "mean": float(row["mean"]), "std": float(row["std"]) if row["std"] is not None else 0.0, "min": float(row["min"]), "p05": float(quantiles[0]), "p25": float(quantiles[1]), "median": float(quantiles[2]), "p75": float(quantiles[3]), "p95": float(quantiles[4]), "max": float(row["max"]), "quantile_relative_error": PROFILE_RELATIVE_ERROR}


def profile_segments(df: DataFrame) -> Dict[str, Any]:
    result = {}
    for segment_id in range(EXPECTED_K):
        segment_df = df.where(col("rule_segment_id") == segment_id)
        result[str(segment_id)] = {"segment_id": segment_id, "count": segment_df.count(), "ratio_pct": segment_df.count() / EXPECTED_COUNT * 100.0, "raw_features": {feature: scalar_profile(segment_df, feature) for feature in RAW_PROFILE_FEATURES}}
    return result


def prediction_validation(df: DataFrame) -> Dict[str, Any]:
    row_count = df.count()
    distinct_customer_id = df.select("customer_id").distinct().count()
    score_invalid = {column: df.where(~col(column).isin([1, 2, 3, 4, 5])).count() for column in ["r_score", "f_score", "m_score"]}
    total_invalid = df.where((col("rfm_total_score") < 3) | (col("rfm_total_score") > 15)).count()
    segment_invalid = df.where(~col("rule_segment_id").isin([0, 1, 2, 3, 4])).count()
    null_labels = sum(df.where(col(column).isNull()).count() for column in ["r_score", "f_score", "m_score", "rfm_total_score", "rule_segment_id", "rule_segment_name_vi", "rule_segment_name_en"])
    return {"row_count": row_count, "distinct_customer_id": distinct_customer_id, "duplicate_customer_id": row_count - distinct_customer_id, "score_invalid": score_invalid, "total_score_invalid": total_invalid, "segment_invalid": segment_invalid, "null_labels": null_labels, "pass": row_count == EXPECTED_COUNT and distinct_customer_id == EXPECTED_COUNT and row_count == distinct_customer_id and all(value == 0 for value in score_invalid.values()) and total_invalid == 0 and segment_invalid == 0 and null_labels == 0}


def prediction_fingerprint(df: DataFrame) -> str:
    digest = hashlib.sha256()
    for row in df.select("customer_id", "rule_segment_id").orderBy("customer_id").toLocalIterator():
        digest.update(str(row["customer_id"]).encode("utf-8"))
        digest.update(b"|")
        digest.update(str(int(row["rule_segment_id"])).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_report(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "STAGE 18 — RULE-BASED RFM BASELINE",
        "",
        "AUTHORITY: RULE_BASED_RFM_BASELINE_V1",
        "METHOD AUTHORITY: RULE_BASED_RFM_QUINTILE_V1",
        f"COHORT_COUNT: {summary.get('cohort_count')}",
        f"READY_FOR_STAGE_19: {summary.get('ready_for_stage19')}",
        "RULE-BASED RFM IS GROUND TRUTH: False",
        "FORMAL THREE-MODEL COMPARISON EXECUTED: False",
        "GMM EXECUTED: False",
        "",
        "THRESHOLDS:",
        str(summary.get("thresholds")),
        "",
        "SEGMENT DISTRIBUTION:",
        str(summary.get("segment_distribution")),
        "",
        "ORDINAL SANITY:",
        str(summary.get("ordinal_sanity")),
        "",
        "REPRODUCIBILITY:",
        str(summary.get("reproducibility")),
        "",
        "CUSTOMER UNIVERSE CHECK:",
        str(summary.get("customer_universe_match")),
    ]
    if summary.get("warnings"):
        lines.extend(["", "WARNINGS:"] + [f"- {item}" for item in summary["warnings"]])
    if summary.get("blockers"):
        lines.extend(["", "BLOCKERS:"] + [f"- {item}" for item in summary["blockers"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    stage17_summary_path = project_root / "output" / "comparison" / "stage17" / "stage17_final_summary.json"
    cohort_path = project_root / "output" / "comparison" / "stage17" / "comparison_cohort_1m"
    cohort_ids_path = project_root / "output" / "comparison" / "stage17" / "comparison_customer_ids_1m"
    output_dir = project_root / "output" / "comparison" / "stage18"
    prediction_path = output_dir / "rule_based_rfm_predictions_1m"
    thresholds_path = output_dir / "rule_based_rfm_thresholds_v1.json"
    definition_path = output_dir / "rule_based_rfm_definition_v1.json"
    score_distribution_path = output_dir / "score_distributions.json"
    segment_distribution_path = output_dir / "rule_segment_distribution.json"
    profiles_path = output_dir / "rule_segment_profiles.json"
    profiles_csv_path = output_dir / "rule_segment_profiles.csv"
    reproducibility_path = output_dir / "reproducibility_check.json"
    manifest_path = output_dir / "rule_prediction_manifest.json"
    summary_path = output_dir / "stage18_final_summary.json"
    report_path = project_root / "reports" / "stage18_rule_based_rfm_baseline.txt"
    run_log_path = project_root / "reports" / "stage18_run.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    blockers: List[str] = []
    warnings: List[str] = []
    spark = None
    cached = None
    predictions = None

    try:
        stage17 = read_json(stage17_summary_path)
        if not (
            stage17.get("stage") == 17
            and stage17.get("cohort_authority") == "DETERMINISTIC_1M_COMPARISON_COHORT"
            and stage17.get("protocol_authority") == "THREE_MODEL_COMPARISON_PROTOCOL_V1"
            and stage17.get("cohort_count") == EXPECTED_COUNT
            and stage17.get("cohort_customer_id_sha256") == COHORT_FINGERPRINT
            and stage17.get("source_alignment_pass") is True
            and stage17.get("deterministic_rerun_validation", {}).get("pass") is True
            and stage17.get("cohort_readback", {}).get("pass") is True
            and stage17.get("protocol_validation_pass") is True
            and stage17.get("ready_for_stage18") is True
            and stage17.get("rule_based_definition_status") == "PENDING_STAGE18_AUTHORITY"
        ):
            blockers.append("Stage 17 authority/fingerprint/gate mismatch.")
        if blockers:
            failed = {"stage": 18, "authority": "RULE_BASED_RFM_BASELINE_V1", "ready_for_stage19": False, "blockers": blockers}
            write_json(summary_path, failed)
            report_path.write_text("STAGE 18 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_19: False\n", encoding="utf-8")
            return 3

        spark = (
            SparkSession.builder
            .master("local[4]")
            .appName("Stage18-Build-Rule-Based-RFM-Baseline")
            .config("spark.sql.shuffle.partitions", "64")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.local.dir", str(project_root.parent / "spark_tmp"))
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        enable_windows_java_filesystem_fallback(spark)
        cohort = spark.read.parquet(str(cohort_path))
        missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in cohort.columns]
        if missing:
            raise ValueError(f"Missing Stage 17 cohort columns: {missing}")
        cached = cohort.select(*REQUIRED_INPUT_COLUMNS).persist(StorageLevel.MEMORY_AND_DISK)
        cached.count()
        validation = input_validation(cached)
        print(f"INPUT_VALIDATION={validation}")
        if not validation["pass"]:
            raise ValueError("Stage 17 cohort RFM input validation failed.")

        thresholds = quantile_thresholds(cached)
        duplicate_thresholds = {feature: sorted({str(value) for value in values}) for feature, values in thresholds.items() if len(set(values)) < len(values)}
        predictions = score_columns(cached.select("customer_id", "selection_hash", *RAW_PROFILE_FEATURES), thresholds).persist(StorageLevel.MEMORY_AND_DISK)
        predictions.count()
        pred_validation = prediction_validation(predictions)
        if not pred_validation["pass"]:
            raise ValueError("Rule-based prediction validation failed.")
        segments = segment_distribution(predictions)
        if not segments["pass"]:
            blockers.append("EMPTY_RULE_SEGMENT")
            raise ValueError("At least one rule-based segment is empty.")

        score_distributions = {column: score_distribution(predictions, column) for column in ["r_score", "f_score", "m_score", "rfm_total_score"]}
        segment_profiles = profile_segments(predictions)
        means_by_segment = {int(segment_id): {feature: profile["raw_features"][feature]["mean"] for feature in RAW_PROFILE_FEATURES} for segment_id, profile in segment_profiles.items()}
        medians_by_segment = {int(segment_id): {feature: profile["raw_features"][feature]["median"] for feature in RAW_PROFILE_FEATURES} for segment_id, profile in segment_profiles.items()}
        ordinal_sanity = {
            "recency_mean_segment0_to4": [means_by_segment[index]["Recency"] for index in range(EXPECTED_K)],
            "frequency_mean_segment0_to4": [means_by_segment[index]["Frequency"] for index in range(EXPECTED_K)],
            "monetary_mean_segment0_to4": [means_by_segment[index]["Monetary"] for index in range(EXPECTED_K)],
            "recency_endpoint_improves": means_by_segment[4]["Recency"] < means_by_segment[0]["Recency"],
            "frequency_endpoint_increases": means_by_segment[4]["Frequency"] > means_by_segment[0]["Frequency"],
            "monetary_endpoint_increases": means_by_segment[4]["Monetary"] > means_by_segment[0]["Monetary"],
            "interpretation": "Descriptive ordinal sanity only; individual feature means need not be strictly monotonic because total scores combine three dimensions.",
        }

        recomputed_thresholds = quantile_thresholds(cached)
        recomputed = score_columns(cached.select("customer_id", "selection_hash", *RAW_PROFILE_FEATURES), recomputed_thresholds).select("customer_id", "r_score", "f_score", "m_score", "rfm_total_score", "rule_segment_id")
        original_keys = predictions.select("customer_id", "r_score", "f_score", "m_score", "rfm_total_score", "rule_segment_id").alias("o")
        recomputed_keys = recomputed.alias("r")
        joined = original_keys.join(recomputed_keys, "customer_id", "inner")
        mismatch_counts = {}
        for column in ["r_score", "f_score", "m_score", "rfm_total_score", "rule_segment_id"]:
            mismatch_counts[f"{column}_mismatch_count"] = joined.where(col(f"o.{column}") != col(f"r.{column}")).count()
        threshold_differences = {feature: [abs(thresholds[feature][index] - recomputed_thresholds[feature][index]) for index in range(4)] for feature in RAW_RFM}
        reproducibility = {"thresholds": {"original": thresholds, "recomputed": recomputed_thresholds, "absolute_differences": threshold_differences, "pass": all(all(value <= 1e-12 for value in values) for values in threshold_differences.values())}, **mismatch_counts}
        reproducibility["pass"] = reproducibility["thresholds"]["pass"] and all(value == 0 for value in mismatch_counts.values())
        if not reproducibility["pass"]:
            raise ValueError("Rule-based reproducibility validation failed.")

        rule_fingerprint = prediction_fingerprint(predictions)
        predictions_to_write = predictions.select("customer_id", "selection_hash", "Recency", "Frequency", "Monetary", "r_score", "f_score", "m_score", "rfm_total_score", "rule_segment_id", "rule_segment_name_vi", "rule_segment_name_en")
        predictions_to_write.write.mode("overwrite").parquet(str(prediction_path))
        readback = spark.read.parquet(str(prediction_path)).persist(StorageLevel.MEMORY_AND_DISK)
        readback.count()
        readback_validation = prediction_validation(readback)
        readback_segments = segment_distribution(readback)
        readback_fingerprint = prediction_fingerprint(readback)
        readback_validation["segment_counts_match"] = readback_segments["segments"] == segments["segments"]
        readback_validation["fingerprint_match"] = readback_fingerprint == rule_fingerprint
        readback_validation["pass"] = readback_validation["pass"] and readback_segments["pass"] and readback_validation["segment_counts_match"] and readback_validation["fingerprint_match"]
        if not readback_validation["pass"]:
            raise ValueError("Rule prediction Parquet read-back failed.")

        stage17_ids = spark.read.parquet(str(cohort_ids_path)).select("customer_id")
        customer_universe_match = {"stage17_minus_rule_predictions": stage17_ids.join(readback.select("customer_id"), "customer_id", "left_anti").count(), "rule_predictions_minus_stage17": readback.select("customer_id").join(stage17_ids, "customer_id", "left_anti").count()}
        customer_universe_match["pass"] = customer_universe_match["stage17_minus_rule_predictions"] == 0 and customer_universe_match["rule_predictions_minus_stage17"] == 0
        if not customer_universe_match["pass"]:
            raise ValueError("Rule prediction customer universe mismatch.")

        thresholds_payload = {"stage": 18, "authority": "RULE_BASED_RFM_QUINTILE_V1", "cohort_authority": "DETERMINISTIC_1M_COMPARISON_COHORT", "cohort_size": EXPECTED_COUNT, "threshold_source": "locked Stage 17 1M cohort", "quantile_method": "Spark approxQuantile", "relative_error": QUANTILE_RELATIVE_ERROR, "probabilities": QUANTILE_PROBABILITIES, "thresholds": thresholds, "duplicate_thresholds": duplicate_thresholds, "tie_policy": "same raw value receives same score", "recency_direction": "lower_is_better", "frequency_direction": "higher_is_better", "monetary_direction": "higher_is_better"}
        definition = {"stage": 18, "authority": "RULE_BASED_RFM_BASELINE_V1", "method_authority": "RULE_BASED_RFM_QUINTILE_V1", "cohort_authority": "DETERMINISTIC_1M_COMPARISON_COHORT", "inputs_used_for_labels": RAW_RFM, "semantic_locks": {"Recency": "days since last purchase date; lower is better", "Frequency": "number of distinct purchase dates; higher is better", "Monetary": "SUM(price); higher is better"}, "scoring_rules": {"Recency": "R<=P20:5, R<=P40:4, R<=P60:3, R<=P80:2, else:1", "Frequency": "F<=P20:1, F<=P40:2, F<=P60:3, F<=P80:4, else:5", "Monetary": "M<=P20:1, M<=P40:2, M<=P60:3, M<=P80:4, else:5"}, "total_score_formula": "r_score + f_score + m_score", "total_score_range": [3, 15], "bands": {str(segment_id): SEGMENT_NAMES[segment_id] for segment_id in range(EXPECTED_K)}, "weighted": False, "ml_training": False, "optimization": False, "depends_on_kmeans_labels": False, "methodology_note": "Project-owned deterministic baseline V1; not the only valid RFM rule system, not ground truth, and defined before Stage 19 comparison."}
        profile_rows = []
        for segment_id, profile in segment_profiles.items():
            for feature, stats in profile["raw_features"].items():
                profile_rows.append({"SegmentId": int(segment_id), "SegmentNameVI": SEGMENT_NAMES[int(segment_id)]["name_vi"], "SegmentNameEN": SEGMENT_NAMES[int(segment_id)]["name_en"], "CustomerCount": profile["count"], "RatioPct": profile["ratio_pct"], "Feature": feature, **{key: value for key, value in stats.items() if key != "quantile_relative_error"}})
        write_json(thresholds_path, thresholds_payload)
        write_json(definition_path, definition)
        write_json(score_distribution_path, score_distributions)
        write_json(segment_distribution_path, segments)
        write_json(profiles_path, {"feature_order": RAW_PROFILE_FEATURES, "segments": segment_profiles, "ordinal_sanity": ordinal_sanity})
        write_csv(profiles_csv_path, ["SegmentId", "SegmentNameVI", "SegmentNameEN", "CustomerCount", "RatioPct", "Feature", "count", "mean", "std", "min", "p05", "p25", "median", "p75", "p95", "max"], profile_rows)
        write_json(reproducibility_path, reproducibility)

        manifest = {"stage": 18, "authority": "RULE_BASED_RFM_BASELINE_V1", "method_authority": "RULE_BASED_RFM_QUINTILE_V1", "cohort_authority": "DETERMINISTIC_1M_COMPARISON_COHORT", "cohort_customer_id_sha256": COHORT_FINGERPRINT, "row_count": EXPECTED_COUNT, "segment_count": EXPECTED_K, "rule_prediction_sha256": rule_fingerprint, "reproducibility_pass": reproducibility["pass"], "readback_pass": readback_validation["pass"]}
        write_json(manifest_path, manifest)
        summary = {"stage": 18, "authority": "RULE_BASED_RFM_BASELINE_V1", "method_authority": "RULE_BASED_RFM_QUINTILE_V1", "cohort_authority": "DETERMINISTIC_1M_COMPARISON_COHORT", "protocol_authority": "THREE_MODEL_COMPARISON_PROTOCOL_V1", "cohort_count": EXPECTED_COUNT, "cohort_customer_id_sha256": COHORT_FINGERPRINT, "thresholds": thresholds, "input_validation": validation, "prediction_validation": pred_validation, "score_distributions": score_distributions, "segment_distribution": segments, "segment_profiles": segment_profiles, "ordinal_sanity": ordinal_sanity, "customer_universe_match": customer_universe_match, "rule_prediction_sha256": rule_fingerprint, "reproducibility": reproducibility, "prediction_readback": readback_validation, "rule_based_is_ground_truth": False, "formal_three_model_comparison_executed": False, "gmm_executed": False, "warnings": warnings, "blockers": blockers, "ready_for_stage19": True}
        write_json(summary_path, summary)
        build_report(report_path, summary)

        artifact_paths = [thresholds_path, definition_path, score_distribution_path, segment_distribution_path, profiles_path, profiles_csv_path, reproducibility_path, manifest_path, summary_path, report_path, run_log_path]
        artifact_readback = {}
        for path in artifact_paths:
            artifact_readback[str(path.relative_to(project_root))] = path.exists() and path.stat().st_size > 0
            if path.suffix == ".json" and artifact_readback[str(path.relative_to(project_root))]:
                try:
                    json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    artifact_readback[str(path.relative_to(project_root))] = False
            if path.suffix == ".csv" and artifact_readback[str(path.relative_to(project_root))]:
                artifact_readback[str(path.relative_to(project_root))] = bool(path.read_text(encoding="utf-8").splitlines()[0].strip())
        artifact_readback[str(prediction_path.relative_to(project_root))] = prediction_path.exists() and any(prediction_path.iterdir())
        summary["artifact_readback"] = artifact_readback
        summary["artifact_readback_pass"] = all(artifact_readback.values())
        summary["ready_for_stage19"] = bool(summary["ready_for_stage19"] and summary["artifact_readback_pass"])
        write_json(summary_path, summary)
        build_report(report_path, summary)
        if not summary["ready_for_stage19"]:
            raise ValueError("Stage 18 artifact read-back failed.")

        print("AUTHORITY_AUDIT_PASS=True")
        print(f"RULE_PREDICTION_SHA256={rule_fingerprint}")
        print("PREDICTION_VALIDATION_PASS=True")
        print("REPRODUCIBILITY_PASS=True")
        print("CUSTOMER_UNIVERSE_MATCH_PASS=True")
        print("READY_FOR_STAGE_19=True")
        return 0
    except Exception as error:
        blockers.append(str(error))
        failed = {"stage": 18, "authority": "RULE_BASED_RFM_BASELINE_V1", "method_authority": "RULE_BASED_RFM_QUINTILE_V1", "ready_for_stage19": False, "warnings": warnings, "blockers": blockers}
        write_json(summary_path, failed)
        report_path.write_text("STAGE 18 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_19: False\n", encoding="utf-8")
        print(f"STAGE 18 FAILED CLOSED: {error}")
        return 3
    finally:
        if predictions is not None:
            predictions.unpersist()
        if cached is not None:
            cached.unpersist()
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    sys.exit(main())
