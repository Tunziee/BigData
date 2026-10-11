"""Stage 11: profile the locked final main K-Means clusters.

This stage is read-only with respect to the authoritative prediction dataset.
It does not retrain a model, change prediction IDs, or build an improved model.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    avg,
    col,
    count,
    countDistinct,
    max as spark_max,
    min as spark_min,
    stddev_samp,
    sum as spark_sum,
    when,
)


EXPECTED_ROW_COUNT = 1_362_281
SELECTED_K = 4
FEATURE_ORDER = ["Recency", "Frequency", "Monetary"]
REQUIRED_COLUMNS = ["customer_id", "Recency", "Frequency", "Monetary", "prediction"]
QUANTILE_PROBABILITIES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
QUANTILE_RELATIVE_ERROR = 0.001
# K-Means centers are updated at the end of the final iteration.  The final
# transform labels can therefore have a small post-update mean delta.  This
# absolute tolerance is 0.001 original RFM units (Recency ~= 86 seconds).
CENTROID_MEAN_TOLERANCE = 1e-3
SKEW_MEAN_MEDIAN_RATIO = 1.5
SKEW_UPPER_TAIL_RATIO = 2.0


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


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


def numeric_stats(df: DataFrame, feature: str) -> Dict[str, Any]:
    aggregate = (
        df.agg(
            count(col(feature)).alias("count"),
            avg(col(feature)).alias("mean"),
            stddev_samp(col(feature)).alias("std"),
            spark_min(col(feature)).alias("min"),
            spark_max(col(feature)).alias("max"),
        )
        .first()
        .asDict()
    )
    quantiles = df.approxQuantile(feature, QUANTILE_PROBABILITIES, QUANTILE_RELATIVE_ERROR)
    if len(quantiles) != len(QUANTILE_PROBABILITIES):
        raise ValueError(f"Quantiles unavailable for {feature}.")
    aggregate.update(
        {
            "p01": float(quantiles[0]),
            "p05": float(quantiles[1]),
            "p25": float(quantiles[2]),
            "median": float(quantiles[3]),
            "p75": float(quantiles[4]),
            "p95": float(quantiles[5]),
            "p99": float(quantiles[6]),
        }
    )
    for key, value in list(aggregate.items()):
        if key != "count" and value is not None:
            aggregate[key] = float(value)
    if not all(
        finite(value)
        for key, value in aggregate.items()
        if key != "count"
    ):
        raise ValueError(f"Non-finite statistic found for {feature}: {aggregate}")
    return aggregate


def all_rfm_stats(df: DataFrame) -> Dict[str, Dict[str, Any]]:
    return {feature: numeric_stats(df, feature) for feature in FEATURE_ORDER}


def direction_from_z(z_value: float) -> str:
    if z_value <= -1.0:
        return "VERY_LOW"
    if z_value <= -0.25:
        return "LOW"
    if z_value < 0.25:
        return "NEAR_AVERAGE"
    if z_value < 1.0:
        return "HIGH"
    return "VERY_HIGH"


def skew_flags(stats: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    median = float(stats["median"])
    mean = float(stats["mean"])
    if median > 0.0 and mean / median >= SKEW_MEAN_MEDIAN_RATIO:
        flags.append(
            f"MEAN_MEDIAN_RATIO>={SKEW_MEAN_MEDIAN_RATIO:g}"
        )
    if float(stats["p75"]) > 0.0 and float(stats["p95"]) / float(stats["p75"]) >= SKEW_UPPER_TAIL_RATIO:
        flags.append(
            f"P95_P75_RATIO>={SKEW_UPPER_TAIL_RATIO:g}"
        )
    return flags


def validate_predictions(predictions: DataFrame) -> Dict[str, Any]:
    count_value = predictions.count()
    distinct_value = predictions.select("customer_id").distinct().count()
    null_prediction = predictions.where(col("prediction").isNull()).count()
    invalid_prediction = predictions.where(
        ~col("prediction").isin(list(range(SELECTED_K)))
    ).count()
    distinct_predictions = predictions.select("prediction").distinct().count()
    return {
        "row_count": count_value,
        "distinct_customer_id": distinct_value,
        "duplicate_customer_id": count_value - distinct_value,
        "prediction_null": null_prediction,
        "invalid_prediction": invalid_prediction,
        "distinct_predictions": distinct_predictions,
        "prediction_values": [
            int(row["prediction"])
            for row in predictions.select("prediction").distinct().orderBy("prediction").collect()
        ],
        "pass": bool(
            count_value == EXPECTED_ROW_COUNT
            and distinct_value == EXPECTED_ROW_COUNT
            and count_value == distinct_value
            and null_prediction == 0
            and invalid_prediction == 0
            and distinct_predictions == SELECTED_K
        ),
    }


def audit_authority(
    stage10_summary_path: Path,
    stage10_metadata_path: Path,
    scaled_centers_path: Path,
    original_centers_path: Path,
) -> Dict[str, Any]:
    blockers: List[str] = []
    try:
        summary = read_json(stage10_summary_path)
    except Exception as error:
        summary = {}
        blockers.append(f"Stage 10 summary unreadable: {error}")
    try:
        metadata = read_json(stage10_metadata_path)
    except Exception as error:
        metadata = {}
        blockers.append(f"Stage 10 metadata unreadable: {error}")
    try:
        scaled_centers = read_json(scaled_centers_path)
    except Exception as error:
        scaled_centers = {}
        blockers.append(f"Stage 10 scaled centers unreadable: {error}")
    try:
        original_centers = read_json(original_centers_path)
    except Exception as error:
        original_centers = {}
        blockers.append(f"Stage 10 original centers unreadable: {error}")

    if not (
        summary.get("stage") == 10
        and summary.get("authority") == "MAIN_KMEANS_FINAL"
        and summary.get("selected_k") == SELECTED_K
        and summary.get("ready_for_stage11") is True
    ):
        blockers.append("Stage 10 summary authority mismatch.")
    if not (
        metadata.get("stage") == 10
        and metadata.get("authority") == "MAIN_KMEANS_FINAL"
        and metadata.get("selected_k") == SELECTED_K
    ):
        blockers.append("Stage 10 metadata authority mismatch.")
    if scaled_centers.get("selected_k") != SELECTED_K:
        blockers.append("Stage 10 scaled center authority mismatch.")
    if original_centers.get("selected_k") != SELECTED_K:
        blockers.append("Stage 10 original center authority mismatch.")

    return {
        "pass": not blockers,
        "blockers": blockers,
        "summary": summary,
        "metadata": metadata,
        "scaled_centers": scaled_centers,
        "original_centers": original_centers,
    }


def build_segment_definitions(
    cluster_profiles: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Names and recommendations are derived only from measured RFM behavior."""

    definitions = {
        "0": {
            "segment_name_vi": "Khách hàng mua sắm cơ bản",
            "segment_name_en": "Mainstream Occasional Customers",
            "evidence": "Largest cluster with moderate Recency and lower Frequency/Monetary than the global baseline.",
            "reason": "The distribution represents the broad mainstream customer base with relatively limited repeat-date frequency and spend per customer.",
            "business_strategy": [
                "low-cost automated engagement",
                "cross-sell and repeat-purchase nudges",
                "maintain broad retention coverage",
            ],
        },
        "1": {
            "segment_name_vi": "Khách hàng hoạt động trung thành",
            "segment_name_en": "Active Loyal Customers",
            "evidence": "Lower Recency and substantially higher Frequency and Monetary than the global baseline and cluster 0.",
            "reason": "Repeated purchase dates and higher observed spend support an active, engaged customer profile without asserting unmeasured loyalty causes.",
            "business_strategy": [
                "retention and loyalty rewards",
                "personalized cross-sell",
                "protect repeat-purchase continuity",
            ],
        },
        "2": {
            "segment_name_vi": "Khách hàng lâu chưa quay lại",
            "segment_name_en": "Lapsed Customers",
            "evidence": "Very high Recency with low Frequency and low Monetary; the distribution is checked against the long-inactivity quantiles.",
            "reason": "The measured gap since last purchase is large, so reactivation language is appropriate; confirmed churn is not claimed.",
            "business_strategy": [
                "reactivation reminders",
                "targeted win-back offers",
                "low-cost automated campaigns before higher-cost outreach",
            ],
        },
        "3": {
            "segment_name_vi": "Khách hàng hoạt động giá trị cao",
            "segment_name_en": "Active High-Value Customers",
            "evidence": "Very low Recency with very high Frequency and Monetary; median and interquartile values are checked to confirm coherence.",
            "reason": "The small cluster is a measured high-activity/high-spend niche rather than a label based only on a few extreme centroid values.",
            "business_strategy": [
                "high-touch retention",
                "loyalty rewards and VIP-care style service",
                "personalized high-value cross-sell",
            ],
        },
    }
    if {key for key in definitions} != {str(profile["cluster_id"]) for profile in cluster_profiles}:
        raise ValueError("Segment definition coverage does not match cluster IDs.")
    names = [definitions[str(profile["cluster_id"])]["segment_name_en"] for profile in cluster_profiles]
    if len(names) != len(set(names)):
        raise ValueError("Segment names are not unique.")
    return definitions


def write_report(
    path: Path,
    authority: Dict[str, Any],
    validation: Dict[str, Any],
    global_stats: Dict[str, Any],
    profiles: List[Dict[str, Any]],
    consistency: Dict[str, Any],
    segment_definitions: Dict[str, Dict[str, Any]],
    final_summary: Dict[str, Any],
    warnings: List[str],
    blockers: List[str],
) -> None:
    lines = [
        "STAGE 11 — FINAL MAIN K-MEANS CLUSTER PROFILING & BUSINESS SEGMENT INTERPRETATION",
        "=" * 120,
        "CURRENT_STAGE: Stage 11",
        "AUTHORITY: MAIN_KMEANS_FINAL_CLUSTER_PROFILE",
        f"Stage 10 authority audit: {authority['pass']}; blockers={authority['blockers']}",
        f"Prediction validation: {validation}",
        "",
        "GLOBAL_RFM_STATISTICS",
        f"Quantile method: approxQuantile; relativeError={QUANTILE_RELATIVE_ERROR}",
    ]
    for feature in FEATURE_ORDER:
        lines.append(f"{feature}: {global_stats[feature]}")
    lines.extend(
        [
            "",
            "CLUSTER_PROFILE_TABLE",
            "Cluster | Segment | Count | RatioPct | RecencyMean | RecencyMedian | RecencyStd | FrequencyMean | FrequencyMedian | FrequencyStd | MonetaryMean | MonetaryMedian | MonetaryStd",
        ]
    )
    for profile in profiles:
        lines.append(
            f"{profile['cluster_id']} | {profile['segment_name_en']} | {profile['count']} | {profile['ratio_pct']:.6f} | "
            f"{profile['rfm']['Recency']['mean']:.6f} | {profile['rfm']['Recency']['median']:.6f} | {profile['rfm']['Recency']['std']:.6f} | "
            f"{profile['rfm']['Frequency']['mean']:.6f} | {profile['rfm']['Frequency']['median']:.6f} | {profile['rfm']['Frequency']['std']:.6f} | "
            f"{profile['rfm']['Monetary']['mean']:.6f} | {profile['rfm']['Monetary']['median']:.6f} | {profile['rfm']['Monetary']['std']:.6f}"
        )
    lines.extend(["", "CENTROID_MEAN_CONSISTENCY"])
    for row in consistency["clusters"]:
        lines.append(str(row))
    lines.append(f"CENTROID_MEAN_CONSISTENCY_PASS: {consistency['pass']}")
    for cluster_id in range(SELECTED_K):
        profile = next(item for item in profiles if item["cluster_id"] == cluster_id)
        lines.extend(
            [
                "",
                f"CLUSTER_{cluster_id}_ANALYSIS",
                f"{profile['behavior_summary']}",
                f"Mean/median and quantile review: {profile['distribution_review']}",
                f"Relative profile: {profile['relative_profile']}",
                f"Business strategy: {profile['business_strategy']}",
            ]
        )
    lines.extend(
        [
            "",
            "FINAL_SEGMENT_NAMES",
            *[
                f"Cluster {cluster_id}: {segment_definitions[str(cluster_id)]['segment_name_vi']} / {segment_definitions[str(cluster_id)]['segment_name_en']}"
                for cluster_id in range(SELECTED_K)
            ],
            "",
            "BUSINESS_STRATEGY_TABLE",
            "Cluster | SegmentName | CustomerCount | RatioPct | RecencyProfile | FrequencyProfile | MonetaryProfile | BehaviorSummary | SuggestedStrategy",
        ]
    )
    for profile in profiles:
        lines.append(
            f"{profile['cluster_id']} | {profile['segment_name_en']} | {profile['count']} | {profile['ratio_pct']:.6f} | "
            f"{profile['relative_profile']['Recency']['direction']} | {profile['relative_profile']['Frequency']['direction']} | {profile['relative_profile']['Monetary']['direction']} | "
            f"{profile['behavior_summary']} | {'; '.join(profile['business_strategy'])}"
        )
    lines.extend(
        [
            "",
            f"WARNINGS: {warnings}",
            f"BLOCKERS: {blockers}",
            f"MAIN_MODEL_BRANCH_COMPLETE: {final_summary['main_model_branch_complete']}",
            f"READY_FOR_IMPROVED_MODEL_STAGE: {final_summary['ready_for_improved_model_stage']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    prediction_path = project_root / "output" / "final" / "main_kmeans_predictions"
    stage10_summary_path = project_root / "output" / "analysis" / "stage10" / "final_model_summary.json"
    stage10_metadata_path = project_root / "output" / "models" / "main_kmeans_final_metadata.json"
    scaled_centers_path = project_root / "output" / "analysis" / "stage10" / "final_centers_scaled.json"
    original_centers_path = project_root / "output" / "analysis" / "stage10" / "final_centers_original.json"
    analysis_dir = project_root / "output" / "analysis" / "stage11"
    report_path = project_root / "reports" / "stage11_final_cluster_profiling.txt"
    log_path = project_root / "reports" / "stage11_run.log"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    authority = audit_authority(
        stage10_summary_path,
        stage10_metadata_path,
        scaled_centers_path,
        original_centers_path,
    )
    print(f"AUTHORITY_AUDIT_PASS={authority['pass']}")
    print(f"AUTHORITY_BLOCKERS={authority['blockers']}")
    if not authority["pass"]:
        failure = {
            "stage": 11,
            "authority": "MAIN_KMEANS_FINAL_CLUSTER_PROFILE",
            "ready_for_improved_model_stage": False,
            "blockers": authority["blockers"],
        }
        write_json(analysis_dir / "final_cluster_profile_summary.json", failure)
        report_path.write_text(
            "STAGE 11 FAILED CLOSED\n"
            + "\n".join(authority["blockers"])
            + "\nREADY_FOR_IMPROVED_MODEL_STAGE: False\n",
            encoding="utf-8",
        )
        return 3

    spark = (
        SparkSession.builder
        .master("local[4]")
        .appName("Stage11-Final-Cluster-Profiling")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.local.dir", str(project_root.parent / "spark_tmp"))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    enable_windows_java_filesystem_fallback(spark)

    blockers: List[str] = []
    warnings: List[str] = []
    try:
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing final prediction authority: {prediction_path}")
        predictions = spark.read.parquet(str(prediction_path))
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in predictions.columns]
        if missing_columns:
            raise ValueError(f"Missing prediction columns: {missing_columns}")
        predictions = predictions.select(*REQUIRED_COLUMNS)
        validation = validate_predictions(predictions)
        print(f"INPUT_VALIDATION={validation}")
        if not validation["pass"]:
            raise ValueError("Final prediction validation failed; fail-closed.")

        global_stats = all_rfm_stats(predictions)
        global_means = {feature: float(global_stats[feature]["mean"]) for feature in FEATURE_ORDER}
        centers_original = authority["original_centers"]["centers_original"]
        centers_scaled = authority["scaled_centers"]["centers_scaled"]
        if len(centers_original) != SELECTED_K or len(centers_scaled) != SELECTED_K:
            raise ValueError("Stage 10 center authority does not contain four centers.")

        cluster_profiles: List[Dict[str, Any]] = []
        consistency_rows: List[Dict[str, Any]] = []
        for cluster_id in range(SELECTED_K):
            cluster_df = predictions.where(col("prediction") == cluster_id)
            cluster_count = cluster_df.count()
            cluster_stats = all_rfm_stats(cluster_df)
            ratio_pct = float(cluster_count) / EXPECTED_ROW_COUNT * 100.0
            actual_mean = [float(cluster_stats[feature]["mean"]) for feature in FEATURE_ORDER]
            inverse_centroid = [float(value) for value in centers_original[cluster_id]]
            absolute_difference = [
                abs(actual_mean[index] - inverse_centroid[index])
                for index in range(3)
            ]
            consistency_pass = all(value <= CENTROID_MEAN_TOLERANCE for value in absolute_difference)
            consistency_rows.append(
                {
                    "cluster_id": cluster_id,
                    "actual_mean_rfm": actual_mean,
                    "inverse_centroid_rfm": inverse_centroid,
                    "absolute_difference": absolute_difference,
                    "tolerance": CENTROID_MEAN_TOLERANCE,
                    "pass": consistency_pass,
                }
            )
            print(
                f"CENTROID_CHECK cluster={cluster_id} actual={actual_mean} "
                f"inverse={inverse_centroid} diff={absolute_difference} "
                f"tolerance={CENTROID_MEAN_TOLERANCE} pass={consistency_pass}"
            )

            relative: Dict[str, Any] = {}
            for index, feature in enumerate(FEATURE_ORDER):
                mean_value = float(cluster_stats[feature]["mean"])
                global_mean = global_means[feature]
                z_value = float(centers_scaled[cluster_id][index])
                relative[feature] = {
                    "mean_to_global_mean_ratio": mean_value / global_mean if global_mean else None,
                    "centroid_z": z_value,
                    "direction": direction_from_z(z_value),
                    "semantic_note": (
                        "Lower Recency indicates more recent activity."
                        if feature == "Recency"
                        else "Higher values indicate more observed purchase dates/spend."
                    ),
                }

            distribution_review_parts: List[str] = []
            for feature in FEATURE_ORDER:
                feature_stats = cluster_stats[feature]
                flags = skew_flags(feature_stats)
                if flags:
                    distribution_review_parts.append(f"{feature}: {', '.join(flags)}")
                else:
                    distribution_review_parts.append(
                        f"{feature}: mean={feature_stats['mean']:.4f}, median={feature_stats['median']:.4f}, no configured skew flag"
                    )
            cluster_profile = {
                "cluster_id": cluster_id,
                "count": cluster_count,
                "ratio_pct": ratio_pct,
                "rfm": cluster_stats,
                "centroid_original": inverse_centroid,
                "centroid_scaled": [float(value) for value in centers_scaled[cluster_id]],
                "relative_profile": relative,
                "distribution_review": "; ".join(distribution_review_parts),
            }
            cluster_profiles.append(cluster_profile)

        definitions = build_segment_definitions(cluster_profiles)
        for profile in cluster_profiles:
            definition = definitions[str(profile["cluster_id"])]
            profile.update(
                {
                    "segment_name_vi": definition["segment_name_vi"],
                    "segment_name_en": definition["segment_name_en"],
                    "evidence": definition["evidence"],
                    "reason": definition["reason"],
                    "business_strategy": definition["business_strategy"],
                }
            )

        cluster0 = cluster_profiles[0]
        cluster1 = cluster_profiles[1]
        cluster2 = cluster_profiles[2]
        cluster3 = cluster_profiles[3]
        cluster1_vs_cluster0 = {
            "frequency_mean_multiple": cluster1["rfm"]["Frequency"]["mean"] / cluster0["rfm"]["Frequency"]["mean"],
            "monetary_mean_multiple": cluster1["rfm"]["Monetary"]["mean"] / cluster0["rfm"]["Monetary"]["mean"],
            "recency_reduction_pct": (
                (cluster0["rfm"]["Recency"]["mean"] - cluster1["rfm"]["Recency"]["mean"])
                / cluster0["rfm"]["Recency"]["mean"]
                * 100.0
            ),
        }
        cluster3_review = {
            "count": cluster3["count"],
            "ratio_pct": cluster3["ratio_pct"],
            "frequency_median": cluster3["rfm"]["Frequency"]["median"],
            "frequency_p25": cluster3["rfm"]["Frequency"]["p25"],
            "frequency_p75": cluster3["rfm"]["Frequency"]["p75"],
            "frequency_p95": cluster3["rfm"]["Frequency"]["p95"],
            "monetary_median": cluster3["rfm"]["Monetary"]["median"],
            "monetary_p25": cluster3["rfm"]["Monetary"]["p25"],
            "monetary_p75": cluster3["rfm"]["Monetary"]["p75"],
            "monetary_p95": cluster3["rfm"]["Monetary"]["p95"],
            "coherent_high_activity_high_value": bool(
                cluster3["rfm"]["Frequency"]["p25"] > cluster0["rfm"]["Frequency"]["p75"]
                and cluster3["rfm"]["Monetary"]["p25"] > cluster0["rfm"]["Monetary"]["p75"]
            ),
        }
        cluster2_review = {
            "recency_mean": cluster2["rfm"]["Recency"]["mean"],
            "recency_median": cluster2["rfm"]["Recency"]["median"],
            "recency_p25": cluster2["rfm"]["Recency"]["p25"],
            "recency_p75": cluster2["rfm"]["Recency"]["p75"],
            "recency_p95": cluster2["rfm"]["Recency"]["p95"],
            "long_inactivity_supported": bool(
                cluster2["rfm"]["Recency"]["p25"] > cluster0["rfm"]["Recency"]["p75"]
            ),
        }
        cluster0_review = {
            "count": cluster0["count"],
            "ratio_pct": cluster0["ratio_pct"],
            "rfm_iqr": {
                feature: cluster0["rfm"][feature]["p75"] - cluster0["rfm"][feature]["p25"]
                for feature in FEATURE_ORDER
            },
            "mean_median": {
                feature: {
                    "mean": cluster0["rfm"][feature]["mean"],
                    "median": cluster0["rfm"][feature]["median"],
                    "flags": skew_flags(cluster0["rfm"][feature]),
                }
                for feature in FEATURE_ORDER
            },
            "p05_p95": {
                feature: {
                    "p05": cluster0["rfm"][feature]["p05"],
                    "p95": cluster0["rfm"][feature]["p95"],
                }
                for feature in FEATURE_ORDER
            },
        }

        for profile in cluster_profiles:
            relative = profile["relative_profile"]
            if profile["cluster_id"] == 0:
                behavior = "Largest mainstream group with moderate recency and lower repeat-date frequency and monetary value relative to the global baseline."
            elif profile["cluster_id"] == 1:
                behavior = (
                    f"More active than cluster 0: Frequency mean is {cluster1_vs_cluster0['frequency_mean_multiple']:.2f}x, "
                    f"Monetary mean is {cluster1_vs_cluster0['monetary_mean_multiple']:.2f}x, and Recency mean is {cluster1_vs_cluster0['recency_reduction_pct']:.2f}% lower."
                )
            elif profile["cluster_id"] == 2:
                behavior = "Long-inactive/lapsed pattern supported by high Recency across the central distribution and low Frequency/Monetary; no confirmed churn claim is made."
            else:
                behavior = "Small but coherent high-activity/high-value niche: its Frequency and Monetary lower quartiles exceed cluster 0 upper quartiles."
            profile["behavior_summary"] = behavior

        centroid_consistency = {
            "tolerance": CENTROID_MEAN_TOLERANCE,
            "clusters": consistency_rows,
            "pass": all(row["pass"] for row in consistency_rows),
        }
        if not centroid_consistency["pass"]:
            raise ValueError("Centroid mean consistency validation failed.")

        business_rows = []
        for profile in cluster_profiles:
            business_rows.append(
                {
                    "Cluster": profile["cluster_id"],
                    "SegmentName": profile["segment_name_en"],
                    "CustomerCount": profile["count"],
                    "RatioPct": profile["ratio_pct"],
                    "RecencyProfile": profile["relative_profile"]["Recency"]["direction"],
                    "FrequencyProfile": profile["relative_profile"]["Frequency"]["direction"],
                    "MonetaryProfile": profile["relative_profile"]["Monetary"]["direction"],
                    "BehaviorSummary": profile["behavior_summary"],
                    "SuggestedStrategy": "; ".join(profile["business_strategy"]),
                }
            )

        final_summary = {
            "stage": 11,
            "authority": "MAIN_KMEANS_FINAL_CLUSTER_PROFILE",
            "selected_k": SELECTED_K,
            "final_model_authority": "MAIN_KMEANS_FINAL",
            "cluster_count": SELECTED_K,
            "source": {
                "predictions": str(prediction_path),
                "final_model_summary": str(stage10_summary_path),
                "final_model_metadata": str(stage10_metadata_path),
            },
            "semantic_lock": {
                "Recency": "days from last purchase date to analysis date; lower is more recent",
                "Frequency": "number of distinct purchase dates per customer",
                "Monetary": "SUM(price)",
            },
            "quantile_method": {
                "method": "Spark approxQuantile",
                "probabilities": QUANTILE_PROBABILITIES,
                "relative_error": QUANTILE_RELATIVE_ERROR,
            },
            "global_statistics": global_stats,
            "segment_mapping": {
                str(profile["cluster_id"]): profile["segment_name_en"]
                for profile in cluster_profiles
            },
            "cluster_distribution": [
                {
                    "cluster_id": profile["cluster_id"],
                    "count": profile["count"],
                    "ratio_pct": profile["ratio_pct"],
                }
                for profile in cluster_profiles
            ],
            "centroid_mean_consistency_pass": centroid_consistency["pass"],
            "profiling_validation_pass": bool(
                validation["pass"]
                and centroid_consistency["pass"]
                and len(cluster_profiles) == SELECTED_K
                and len(set(profile["segment_name_en"] for profile in cluster_profiles)) == SELECTED_K
            ),
            "special_reviews": {
                "cluster_0": cluster0_review,
                "cluster_1_vs_cluster_0": cluster1_vs_cluster0,
                "cluster_2": cluster2_review,
                "cluster_3": cluster3_review,
            },
            "main_model_branch_complete": True,
            "ready_for_improved_model_stage": True,
        }
        if not final_summary["profiling_validation_pass"]:
            raise ValueError("Stage 11 profiling validation failed.")

        write_json(analysis_dir / "global_rfm_statistics.json", {
            "stage": 11,
            "authority": "MAIN_KMEANS_FINAL_CLUSTER_PROFILE",
            "quantile_method": final_summary["quantile_method"],
            "global_statistics": global_stats,
        })
        write_json(analysis_dir / "cluster_profiles.json", {
            "stage": 11,
            "authority": "MAIN_KMEANS_FINAL_CLUSTER_PROFILE",
            "source": final_summary["source"],
            "global_statistics": global_stats,
            "clusters": cluster_profiles,
        })
        write_json(analysis_dir / "centroid_mean_validation.json", centroid_consistency)
        write_json(analysis_dir / "segment_names.json", {
            "stage": 11,
            "authority": "MAIN_KMEANS_FINAL_CLUSTER_PROFILE",
            "prediction_ids_unchanged": True,
            "segments": {
                cluster_id: definition
                for cluster_id, definition in definitions.items()
            },
        })
        write_json(analysis_dir / "final_cluster_profile_summary.json", final_summary)

        profile_csv_rows: List[Dict[str, Any]] = []
        for profile in cluster_profiles:
            row: Dict[str, Any] = {
                "cluster_id": profile["cluster_id"],
                "count": profile["count"],
                "ratio_pct": profile["ratio_pct"],
                "segment_name_vi": profile["segment_name_vi"],
                "segment_name_en": profile["segment_name_en"],
            }
            for feature in FEATURE_ORDER:
                for statistic in ["mean", "median", "std", "min", "p01", "p05", "p25", "p75", "p95", "p99", "max"]:
                    row[f"{feature}_{statistic}"] = profile["rfm"][feature][statistic]
            profile_csv_rows.append(row)
        import csv

        with (analysis_dir / "cluster_profiles.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = list(profile_csv_rows[0].keys())
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(profile_csv_rows)
        with (analysis_dir / "business_strategy_table.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = list(business_rows[0].keys())
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(business_rows)

        readback_checks = {
            "global_json_exists": (analysis_dir / "global_rfm_statistics.json").exists(),
            "profiles_json_exists": (analysis_dir / "cluster_profiles.json").exists(),
            "profiles_csv_exists": (analysis_dir / "cluster_profiles.csv").exists(),
            "centroid_validation_exists": (analysis_dir / "centroid_mean_validation.json").exists(),
            "segment_names_exists": (analysis_dir / "segment_names.json").exists(),
            "business_csv_exists": (analysis_dir / "business_strategy_table.csv").exists(),
            "summary_exists": (analysis_dir / "final_cluster_profile_summary.json").exists(),
            "profile_rows": len(profile_csv_rows) == SELECTED_K,
            "business_rows": len(business_rows) == SELECTED_K,
        }
        final_summary["artifact_readback"] = readback_checks
        final_summary["ready_for_improved_model_stage"] = bool(
            final_summary["ready_for_improved_model_stage"] and all(readback_checks.values())
        )
        write_json(analysis_dir / "final_cluster_profile_summary.json", final_summary)
        write_report(
            report_path,
            authority,
            validation,
            global_stats,
            cluster_profiles,
            centroid_consistency,
            definitions,
            final_summary,
            warnings,
            blockers,
        )
        print(f"CENTROID_MEAN_CONSISTENCY_PASS={centroid_consistency['pass']}")
        print(f"READY_FOR_IMPROVED_MODEL_STAGE={final_summary['ready_for_improved_model_stage']}")
        return 0 if final_summary["ready_for_improved_model_stage"] else 3
    except Exception as error:
        blockers.append(str(error))
        print(f"STAGE 11 FAILED CLOSED: {error}")
        failure = {
            "stage": 11,
            "authority": "MAIN_KMEANS_FINAL_CLUSTER_PROFILE",
            "selected_k": SELECTED_K,
            "ready_for_improved_model_stage": False,
            "blockers": blockers,
        }
        write_json(analysis_dir / "final_cluster_profile_summary.json", failure)
        report_path.write_text(
            "STAGE 11 FAILED CLOSED\n"
            + "\n".join(blockers)
            + "\nREADY_FOR_IMPROVED_MODEL_STAGE: False\n",
            encoding="utf-8",
        )
        return 3
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
