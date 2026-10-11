"""Stage 16: profile the locked final Enhanced K-Means predictions.

No model fitting occurs here.  The Stage 15 prediction Parquet is the sole
cluster-membership authority for all raw-feature profiling.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import avg, col, count, lit, log, max as spark_max, min as spark_min, stddev_samp
from pyspark.storagelevel import StorageLevel


EXPECTED_ROW_COUNT = 1_362_281
EXPECTED_K = 5
RAW_FEATURES = ["Recency", "Frequency", "Monetary", "AvgBasketSize", "ProductVariety"]
TRANSFORMED_FEATURES = [
    "Recency_improved",
    "Frequency_log1p",
    "Monetary_log1p",
    "AvgBasketSize_log1p",
    "ProductVariety_log1p",
]
TRANSFORMED_CENTER_ORDER = ["Recency", "log1p(Frequency)", "log1p(Monetary)", "log1p(AvgBasketSize)", "log1p(ProductVariety)"]
QUANTILE_PROBABILITIES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
QUANTILE_RELATIVE_ERROR = 0.001
CENTROID_TOLERANCE = 1e-3
CENTROID_FALLBACK_TOLERANCE = 1e-2
MEAN_TOLERANCE = 1e-8


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


def write_csv(path: Path, fields: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


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


def scalar_stats(df: DataFrame, feature: str) -> Dict[str, Any]:
    row = df.agg(
        count(feature).alias("count"),
        avg(feature).alias("mean"),
        stddev_samp(feature).alias("std"),
        spark_min(feature).alias("min"),
        spark_max(feature).alias("max"),
    ).first().asDict()
    quantiles = df.approxQuantile(feature, QUANTILE_PROBABILITIES, QUANTILE_RELATIVE_ERROR)
    labels = ["p01", "p05", "p25", "p50", "p75", "p95", "p99"]
    result: Dict[str, Any] = {
        "count": int(row["count"]),
        "mean": float(row["mean"]),
        "std": float(row["std"]) if row["std"] is not None else 0.0,
        "min": float(row["min"]),
        "max": float(row["max"]),
        "quantile_relative_error": QUANTILE_RELATIVE_ERROR,
    }
    result.update({label: float(value) for label, value in zip(labels, quantiles)})
    return result


def validate_distribution(df: DataFrame, expected_counts: Dict[int, int]) -> Dict[str, Any]:
    row_count = df.count()
    distinct_customer_id = df.select("customer_id").distinct().count()
    null_prediction = df.where(col("prediction").isNull()).count()
    rows = df.groupBy("prediction").count().orderBy("prediction").collect()
    counts = [{"cluster_id": int(row["prediction"]), "count": int(row["count"])} for row in rows]
    total = sum(item["count"] for item in counts)
    for item in counts:
        item["ratio_pct"] = item["count"] / total * 100.0
    actual_ids = [item["cluster_id"] for item in counts]
    expected_ids = list(range(EXPECTED_K))
    expected_rows = [{"cluster_id": key, "count": value} for key, value in sorted(expected_counts.items())]
    return {
        "row_count": row_count,
        "distinct_customer_id": distinct_customer_id,
        "duplicate_customer_id": row_count - distinct_customer_id,
        "prediction_null_count": null_prediction,
        "prediction_values": actual_ids,
        "cluster_count": len(counts),
        "cluster_sizes": counts,
        "expected_stage15_cluster_sizes": expected_rows,
        "pass": row_count == EXPECTED_ROW_COUNT and distinct_customer_id == EXPECTED_ROW_COUNT and row_count == distinct_customer_id and null_prediction == 0 and actual_ids == expected_ids and [{"cluster_id": item["cluster_id"], "count": item["count"]} for item in counts] == expected_rows,
    }


def profile_clusters(df: DataFrame) -> Dict[str, Any]:
    profiles: Dict[str, Any] = {}
    for cluster_id in range(EXPECTED_K):
        cluster_df = df.where(col("prediction") == cluster_id)
        count_value = cluster_df.count()
        profiles[str(cluster_id)] = {
            "cluster_id": cluster_id,
            "count": count_value,
            "ratio_pct": count_value / EXPECTED_ROW_COUNT * 100.0,
            "raw_features": {feature: scalar_stats(cluster_df, feature) for feature in RAW_FEATURES},
        }
    return profiles


def reconstruct_transformed(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("Recency_improved", col("Recency").cast("double"))
        .withColumn("Frequency_log1p", log(col("Frequency").cast("double") + lit(1.0)))
        .withColumn("Monetary_log1p", log(col("Monetary").cast("double") + lit(1.0)))
        .withColumn("AvgBasketSize_log1p", log(col("AvgBasketSize").cast("double") + lit(1.0)))
        .withColumn("ProductVariety_log1p", log(col("ProductVariety").cast("double") + lit(1.0)))
    )


def transformed_centroid_validation(df: DataFrame, stage15_centers: Dict[int, List[float]]) -> Dict[str, Any]:
    expressions = [avg(column).alias(column) for column in TRANSFORMED_FEATURES]
    rows = df.groupBy("prediction").agg(*expressions).orderBy("prediction").collect()
    clusters = []
    for row in rows:
        cluster_id = int(row["prediction"])
        actual = [float(row[column]) for column in TRANSFORMED_FEATURES]
        reference = stage15_centers[cluster_id]
        feature_rows = []
        for index, feature in enumerate(TRANSFORMED_CENTER_ORDER):
            difference = abs(actual[index] - reference[index])
            feature_rows.append({
                "feature": feature,
                "actual_transformed_mean": actual[index],
                "stage15_transformed_center": reference[index],
                "absolute_difference": difference,
                "tolerance": CENTROID_TOLERANCE,
                "pass": difference <= CENTROID_TOLERANCE,
            })
        clusters.append({"cluster_id": cluster_id, "features": feature_rows, "max_absolute_difference": max(item["absolute_difference"] for item in feature_rows), "pass": all(item["pass"] for item in feature_rows)})
    max_difference = max(item["max_absolute_difference"] for item in clusters)
    fallback_used = False
    if not all(item["pass"] for item in clusters) and max_difference <= CENTROID_FALLBACK_TOLERANCE:
        fallback_used = True
        for cluster in clusters:
            for item in cluster["features"]:
                item["tolerance"] = CENTROID_FALLBACK_TOLERANCE
                item["pass"] = item["absolute_difference"] <= CENTROID_FALLBACK_TOLERANCE
            cluster["pass"] = all(item["pass"] for item in cluster["features"])
    return {
        "feature_order": TRANSFORMED_CENTER_ORDER,
        "preferred_tolerance": CENTROID_TOLERANCE,
        "fallback_tolerance": CENTROID_FALLBACK_TOLERANCE,
        "fallback_used": fallback_used,
        "clusters": clusters,
        "max_absolute_difference": max_difference,
        "pass": all(item["pass"] for item in clusters),
    }


def raw_mean_comparison(profiles: Dict[str, Any], representative: Dict[int, List[float]]) -> List[Dict[str, Any]]:
    rows = []
    for cluster_id in range(EXPECTED_K):
        for feature_index, feature in enumerate(RAW_FEATURES):
            actual = profiles[str(cluster_id)]["raw_features"][feature]["mean"]
            derived = representative[cluster_id][feature_index]
            rows.append({
                "cluster_id": cluster_id,
                "feature": feature,
                "actual_raw_arithmetic_mean": actual,
                "representative_centroid_derived_raw_value": derived,
                "absolute_difference": abs(actual - derived),
                "note": "Representative values implied by transformed-space centroids; not raw cluster arithmetic means." if feature != "Recency" else "Recency is identity-transformed; arithmetic mean is expected to be close to the transformed centroid.",
            })
    return rows


def pairwise_distinctness(profiles: Dict[str, Any], global_stats: Dict[str, Any]) -> Dict[str, Any]:
    iqr = {feature: global_stats[feature]["p75"] - global_stats[feature]["p25"] for feature in RAW_FEATURES}
    safe_iqr = {feature: value if value > 0 else 1.0 for feature, value in iqr.items()}
    pairs = []
    for left, right in combinations(range(EXPECTED_K), 2):
        feature_rows = []
        squared_sum = 0.0
        for feature in RAW_FEATURES:
            left_median = profiles[str(left)]["raw_features"][feature]["p50"]
            right_median = profiles[str(right)]["raw_features"][feature]["p50"]
            normalized = abs(left_median - right_median) / safe_iqr[feature]
            squared_sum += normalized * normalized
            feature_rows.append({"feature": feature, "left_median": left_median, "right_median": right_median, "global_iqr": iqr[feature], "normalized_absolute_difference": normalized})
        pairs.append({"cluster_a": left, "cluster_b": right, "feature_differences": feature_rows, "normalized_median_distance": math.sqrt(squared_sum)})
    closest = min(pairs, key=lambda item: item["normalized_median_distance"])
    farthest = max(pairs, key=lambda item: item["normalized_median_distance"])
    return {"method": "Euclidean distance between cluster median differences normalized by global raw-feature IQR; descriptive only.", "global_iqr": iqr, "pairs": pairs, "closest_behavioral_pair": closest, "farthest_behavioral_pair": farthest}


def assign_segment_names(profiles: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    c0 = profiles["0"]["raw_features"]
    c1 = profiles["1"]["raw_features"]
    c2 = profiles["2"]["raw_features"]
    c3 = profiles["3"]["raw_features"]
    c4 = profiles["4"]["raw_features"]
    result = {
        "0": {"name_vi": "Khách hàng mua ít nhưng giỏ lớn", "name_en": "Infrequent Large-Basket Customers", "evidence_supported": c0["Frequency"]["p50"] <= 3 and c0["AvgBasketSize"]["p50"] > c3["AvgBasketSize"]["p50"]},
        "1": {"name_vi": "Khách hàng hoạt động giá trị cao", "name_en": "Active High-Value Customers", "evidence_supported": c1["Recency"]["p50"] < c3["Recency"]["p50"] and c1["Frequency"]["p25"] > c4["Frequency"]["p50"] and c1["Monetary"]["p25"] > c4["Monetary"]["p50"] and c1["ProductVariety"]["p25"] > c4["ProductVariety"]["p50"]},
        "2": {"name_vi": "Khách hàng lâu chưa quay lại", "name_en": "Lapsed Customers", "evidence_supported": c2["Recency"]["p50"] > c3["Recency"]["p50"] and c2["Frequency"]["p50"] <= 1 and c2["Monetary"]["p50"] < c3["Monetary"]["p50"]},
        "3": {"name_vi": "Khách hàng ít tương tác", "name_en": "Low-Engagement Customers", "evidence_supported": c3["Recency"]["p50"] < c2["Recency"]["p50"] and c3["Frequency"]["p50"] <= 2 and c3["Monetary"]["p50"] < c4["Monetary"]["p50"]},
        "4": {"name_vi": "Khách hàng hoạt động thường xuyên", "name_en": "Active Regular Customers", "evidence_supported": c4["Recency"]["p50"] < c3["Recency"]["p50"] and c4["Frequency"]["p50"] > c3["Frequency"]["p50"] and c4["Frequency"]["p50"] < c1["Frequency"]["p50"]},
    }
    for item in result.values():
        if not item["evidence_supported"]:
            item["name_vi"] = item["name_vi"] + " (cần xem xét)"
            item["name_en"] = item["name_en"] + " (review)"
    return result


def build_strategy_table(profiles: Dict[str, Any], names: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    strategies = {
        "0": "basket-growth và repeat-purchase nudges; personalization theo nhóm sản phẩm; theo dõi khả năng tăng tần suất mua.",
        "1": "high-value care, loyalty rewards và personalization; cross-sell có chọn lọc dựa trên lịch sử mua.",
        "2": "reactivation và win-back bằng low-cost automated campaigns; không diễn giải thành churn chắc chắn.",
        "3": "low-cost automated campaigns, repeat-purchase nudges và personalization đơn giản để khuyến khích quay lại.",
        "4": "loyalty rewards, cross-sell và repeat-purchase nudges để duy trì hoạt động thường xuyên.",
    }
    rows = []
    for cluster_id in range(EXPECTED_K):
        profile = profiles[str(cluster_id)]
        stats = profile["raw_features"]
        rows.append({
            "Cluster": cluster_id,
            "SegmentNameVI": names[str(cluster_id)]["name_vi"],
            "SegmentNameEN": names[str(cluster_id)]["name_en"],
            "CustomerCount": profile["count"],
            "RatioPct": profile["ratio_pct"],
            "RecencyProfile": f"median={stats['Recency']['p50']:.4f}, P25={stats['Recency']['p25']:.4f}, P75={stats['Recency']['p75']:.4f}",
            "FrequencyProfile": f"median={stats['Frequency']['p50']:.4f}, P25={stats['Frequency']['p25']:.4f}, P75={stats['Frequency']['p75']:.4f}",
            "MonetaryProfile": f"median={stats['Monetary']['p50']:.4f}, P25={stats['Monetary']['p25']:.4f}, P75={stats['Monetary']['p75']:.4f}",
            "AvgBasketProfile": f"median={stats['AvgBasketSize']['p50']:.4f}, P25={stats['AvgBasketSize']['p25']:.4f}, P75={stats['AvgBasketSize']['p75']:.4f}",
            "ProductVarietyProfile": f"median={stats['ProductVariety']['p50']:.4f}, P25={stats['ProductVariety']['p25']:.4f}, P75={stats['ProductVariety']['p75']:.4f}",
            "BehaviorSummary": f"Recency median {stats['Recency']['p50']:.4f}; Frequency median {stats['Frequency']['p50']:.4f}; Monetary median {stats['Monetary']['p50']:.4f}; AvgBasketSize median {stats['AvgBasketSize']['p50']:.4f}; ProductVariety median {stats['ProductVariety']['p50']:.4f}.",
            "SuggestedStrategy": strategies[str(cluster_id)],
        })
    return rows


def write_report(path: Path, summary: Dict[str, Any], strategy_rows: List[Dict[str, Any]]) -> None:
    lines = [
        "STAGE 16 — FINAL ENHANCED K-MEANS CLUSTER PROFILING",
        "",
        "AUTHORITY: ENHANCED_KMEANS_FINAL_CLUSTER_PROFILE",
        "FINAL MODEL AUTHORITY: ENHANCED_KMEANS_FINAL",
        f"SELECTED_K: {summary.get('selected_k')}",
        f"ENHANCED_MODEL_BRANCH_COMPLETE: {summary.get('enhanced_model_branch_complete')}",
        f"READY_FOR_GMM_STAGE: {summary.get('ready_for_gmm_stage')}",
        "NO GMM EXECUTED: True",
        "NO MAIN-VS-ENHANCED WINNER CONCLUSION: True",
        "",
        f"INPUT_VALIDATION: {summary.get('input_validation')}",
        "",
        "GLOBAL eRFM STATISTICS:",
    ]
    for feature, stats in summary.get("global_erfm_statistics", {}).get("statistics", {}).items():
        lines.append(
            f"  {feature}: count={stats['count']}, mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
            f"min={stats['min']:.6f}, P01={stats['p01']:.6f}, P05={stats['p05']:.6f}, P25={stats['p25']:.6f}, "
            f"P50={stats['p50']:.6f}, P75={stats['p75']:.6f}, P95={stats['p95']:.6f}, P99={stats['p99']:.6f}, max={stats['max']:.6f}"
        )
    lines.extend([
        "",
        "CLUSTER PROFILE TABLE:",
    ])
    for row in strategy_rows:
        lines.append(f"  C{row['Cluster']}: n={row['CustomerCount']}, ratio={row['RatioPct']:.6f}%, {row['BehaviorSummary']}")
    lines.extend(["", "CLUSTER ANALYSES:"])
    for cluster_id, profile in summary.get("cluster_profiles", {}).get("clusters", {}).items():
        stats = profile["raw_features"]
        lines.append(
            f"  CLUSTER_{cluster_id}_ANALYSIS: Recency mean/median={stats['Recency']['mean']:.6f}/{stats['Recency']['p50']:.6f}; "
            f"Frequency mean/median={stats['Frequency']['mean']:.6f}/{stats['Frequency']['p50']:.6f}; "
            f"Monetary mean/median={stats['Monetary']['mean']:.6f}/{stats['Monetary']['p50']:.6f}; "
            f"AvgBasketSize mean/median={stats['AvgBasketSize']['mean']:.6f}/{stats['AvgBasketSize']['p50']:.6f}; "
            f"ProductVariety mean/median={stats['ProductVariety']['mean']:.6f}/{stats['ProductVariety']['p50']:.6f}."
        )
    lines.extend(["", "TRANSFORMED CENTROID CONSISTENCY:", str(summary.get("transformed_centroid_validation")), "", "RAW MEAN VS REPRESENTATIVE CENTER:", "Representative values implied by transformed-space centroids; not raw cluster arithmetic means.", "", "PAIRWISE DISTINCTNESS:"])
    pairwise = summary.get("pairwise_cluster_distinctness", {})
    lines.append(f"  closest={pairwise.get('closest_behavioral_pair')}")
    lines.append(f"  farthest={pairwise.get('farthest_behavioral_pair')}")
    lines.extend(["", "FINAL SEGMENT NAMES:"])
    for cluster_id, item in summary.get("segment_mapping", {}).items():
        lines.append(f"  C{cluster_id}: {item['name_vi']} / {item['name_en']} (evidence_supported={item['evidence_supported']})")
    lines.extend(["", "BUSINESS STRATEGY:"])
    for row in strategy_rows:
        lines.append(f"  C{row['Cluster']}: {row['SuggestedStrategy']}")
    lines.extend(["", "STAGE 14 PROFILE CONSISTENCY:", str(summary.get("stage14_profile_consistency"))])
    if summary.get("warnings"):
        lines.extend(["", "WARNINGS:"] + [f"- {item}" for item in summary["warnings"]])
    if summary.get("blockers"):
        lines.extend(["", "BLOCKERS:"] + [f"- {item}" for item in summary["blockers"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    stage15_summary_path = project_root / "output" / "analysis" / "stage15" / "final_enhanced_model_summary.json"
    stage15_metadata_path = project_root / "output" / "models" / "enhanced_kmeans_final_metadata.json"
    stage15_transformed_path = project_root / "output" / "analysis" / "stage15" / "final_centers_transformed.json"
    stage15_representative_path = project_root / "output" / "analysis" / "stage15" / "final_centers_representative_raw.json"
    stage14_profiles_path = project_root / "output" / "analysis" / "stage14" / "candidate_raw_profiles.json"
    prediction_path = project_root / "output" / "final" / "enhanced_kmeans_predictions"
    output_dir = project_root / "output" / "analysis" / "stage16"
    report_path = project_root / "reports" / "stage16_final_enhanced_cluster_profiling.txt"
    run_log_path = project_root / "reports" / "stage16_run.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    blockers: List[str] = []
    warnings: List[str] = []
    spark = None
    cached_predictions = None

    try:
        stage15 = read_json(stage15_summary_path)
        metadata = read_json(stage15_metadata_path)
        transformed_payload = read_json(stage15_transformed_path)
        representative_payload = read_json(stage15_representative_path)
        if not (stage15.get("stage") == 15 and stage15.get("authority") == "ENHANCED_KMEANS_FINAL" and stage15.get("selected_k") == EXPECTED_K and stage15.get("ready_for_stage16") is True):
            blockers.append("Stage 15 authority mismatch.")
        if stage15.get("feature_authority") not in (None, "IMPROVED_ERFM_FEATURE_SPACE"):
            blockers.append("Stage 15 feature authority mismatch.")
        if metadata.get("feature_order") != TRANSFORMED_FEATURES:
            blockers.append("Stage 15 feature order mismatch.")
        if metadata.get("transformations") != {"Recency": "identity", "Frequency": "log1p", "Monetary": "log1p", "AvgBasketSize": "log1p", "ProductVariety": "log1p"}:
            blockers.append("Stage 15 transformation metadata mismatch.")
        if blockers:
            failed = {"stage": 16, "authority": "ENHANCED_KMEANS_FINAL_CLUSTER_PROFILE", "ready_for_gmm_stage": False, "blockers": blockers}
            write_json(output_dir / "final_enhanced_cluster_profile_summary.json", failed)
            report_path.write_text("STAGE 16 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_GMM_STAGE: False\n", encoding="utf-8")
            return 3

        expected_counts = {int(item["cluster_id"]): int(item["count"]) for item in stage15["cluster_distribution"]["cluster_sizes"]}
        stage15_centers_transformed = {int(item["cluster_id"]): [float(value) for value in item["values"]] for item in transformed_payload["centers"]}
        stage15_centers_representative = {int(item["cluster_id"]): [float(value) for value in item["values"]] for item in representative_payload["centers"]}

        spark = (
            SparkSession.builder
            .master("local[4]")
            .appName("Stage16-Profile-Final-Enhanced-KMeans")
            .config("spark.sql.shuffle.partitions", "64")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.local.dir", str(project_root.parent / "spark_tmp"))
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        enable_windows_java_filesystem_fallback(spark)
        source = spark.read.parquet(str(prediction_path))
        required = ["customer_id"] + RAW_FEATURES + ["prediction"]
        missing = [column for column in required if column not in source.columns]
        if missing:
            raise ValueError(f"Missing final prediction columns: {missing}")
        cached_predictions = source.select(*required).persist(StorageLevel.MEMORY_AND_DISK)
        cached_predictions.count()
        input_validation = validate_distribution(cached_predictions, expected_counts)
        print(f"INPUT_VALIDATION={input_validation}")
        if not input_validation["pass"]:
            raise ValueError("Final prediction validation failed.")

        global_stats = {feature: scalar_stats(cached_predictions, feature) for feature in RAW_FEATURES}
        profiles = profile_clusters(cached_predictions)
        raw_statistics_valid = all(
            finite(value)
            for feature_stats in global_stats.values()
            for key, value in feature_stats.items()
            if key != "quantile_relative_error"
        ) and all(
            finite(value)
            for profile in profiles.values()
            for feature_stats in profile["raw_features"].values()
            for key, value in feature_stats.items()
            if key != "quantile_relative_error"
        )
        if not raw_statistics_valid:
            raise ValueError("Raw statistics contain non-finite values.")

        reconstructed = reconstruct_transformed(cached_predictions)
        centroid_validation = transformed_centroid_validation(reconstructed, stage15_centers_transformed)
        if centroid_validation["fallback_used"]:
            warnings.append(
                "Centroid validation used the allowed 1e-2 fallback tolerance: all finite differences remained below 1e-2, with maximum difference "
                f"{centroid_validation['max_absolute_difference']:.12f}; this is reported and is not treated as a material discrepancy."
            )
        if not centroid_validation["pass"]:
            raise ValueError("Transformed centroid mean validation failed.")

        raw_comparison_rows = raw_mean_comparison(profiles, stage15_centers_representative)
        pairwise = pairwise_distinctness(profiles, global_stats)
        names = assign_segment_names(profiles)
        if not all(item["evidence_supported"] for item in names.values()):
            warnings.append("At least one candidate segment name required review against the observed profile evidence.")
        strategy_rows = build_strategy_table(profiles, names)

        stage14_consistency: Dict[str, Any] = {"available": stage14_profiles_path.exists(), "pass": True}
        if stage14_profiles_path.exists():
            stage14 = read_json(stage14_profiles_path)
            candidate = stage14.get("candidates", {}).get("5")
            if candidate is None:
                stage14_consistency = {"available": True, "pass": False, "reason": "Stage 14 K=5 profile missing."}
            else:
                checks = []
                for cluster_id in range(EXPECTED_K):
                    actual_profile = profiles[str(cluster_id)]
                    candidate_profile = next(item for item in candidate["clusters"] if int(item["cluster_id"]) == cluster_id)
                    count_pass = actual_profile["count"] == int(candidate_profile["count"])
                    mean_checks = {}
                    for feature in RAW_FEATURES:
                        actual_mean = actual_profile["raw_features"][feature]["mean"]
                        candidate_mean = float(candidate_profile["raw_features"][feature]["mean"])
                        difference = abs(actual_mean - candidate_mean)
                        allowed = MEAN_TOLERANCE * max(1.0, abs(candidate_mean))
                        mean_checks[feature] = {"actual": actual_mean, "stage14": candidate_mean, "absolute_difference": difference, "allowed_difference": allowed, "pass": difference <= allowed}
                    checks.append({"cluster_id": cluster_id, "count_pass": count_pass, "mean_checks": mean_checks, "pass": count_pass and all(item["pass"] for item in mean_checks.values())})
                stage14_consistency = {"available": True, "clusters": checks, "quantiles_exact_required": False, "pass": all(item["pass"] for item in checks)}
        if not stage14_consistency["pass"]:
            raise ValueError("Stage 14 K=5 profile consistency failed.")

        global_payload = {"feature_order": RAW_FEATURES, "statistics": global_stats}
        profiles_payload = {"feature_order": RAW_FEATURES, "cluster_count": EXPECTED_K, "clusters": profiles, "semantic_lock": {"Recency": "days from last purchase date to analysis date", "Frequency": "number of distinct purchase dates", "Monetary": "SUM(price)", "AvgBasketSize": "TotalItems / Frequency where Frequency is distinct purchase dates", "ProductVariety": "number of distinct article_id purchased"}}
        write_json(output_dir / "global_erfm_statistics.json", global_payload)
        write_json(output_dir / "cluster_profiles.json", profiles_payload)
        profile_rows = []
        for cluster_id, profile in profiles.items():
            for feature, stats in profile["raw_features"].items():
                profile_rows.append({"Cluster": int(cluster_id), "CustomerCount": profile["count"], "RatioPct": profile["ratio_pct"], "Feature": feature, **{key: value for key, value in stats.items() if key != "quantile_relative_error"}})
        write_csv(output_dir / "cluster_profiles.csv", ["Cluster", "CustomerCount", "RatioPct", "Feature", "count", "mean", "std", "min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"], profile_rows)
        write_json(output_dir / "transformed_centroid_mean_validation.json", centroid_validation)
        write_json(output_dir / "raw_mean_vs_representative_center.json", {"note": "Representative values implied by transformed-space centroids; not raw cluster arithmetic means.", "rows": raw_comparison_rows})
        write_csv(output_dir / "raw_mean_vs_representative_center.csv", list(raw_comparison_rows[0].keys()), raw_comparison_rows)
        write_json(output_dir / "pairwise_cluster_distinctness.json", pairwise)
        write_json(output_dir / "segment_names.json", {"segments": names})
        write_csv(output_dir / "business_strategy_table.csv", list(strategy_rows[0].keys()), strategy_rows)

        summary: Dict[str, Any] = {
            "stage": 16,
            "authority": "ENHANCED_KMEANS_FINAL_CLUSTER_PROFILE",
            "final_model_authority": "ENHANCED_KMEANS_FINAL",
            "selected_k": EXPECTED_K,
            "customer_count": EXPECTED_ROW_COUNT,
            "cluster_count": EXPECTED_K,
            "segment_mapping": names,
            "cluster_distribution": input_validation["cluster_sizes"],
            "input_validation": input_validation,
            "global_erfm_statistics": global_payload,
            "cluster_profiles": profiles_payload,
            "transformed_centroid_validation": centroid_validation,
            "transformed_centroid_validation_pass": centroid_validation["pass"],
            "raw_mean_vs_representative_center": {"note": "Representative values implied by transformed-space centroids; not raw cluster arithmetic means.", "rows": raw_comparison_rows},
            "pairwise_cluster_distinctness": pairwise,
            "stage14_profile_consistency": stage14_consistency,
            "profiling_validation_pass": bool(input_validation["pass"] and raw_statistics_valid and centroid_validation["pass"] and stage14_consistency["pass"]),
            "enhanced_model_branch_complete": True,
            "ready_for_gmm_stage": True,
            "no_main_vs_enhanced_winner": True,
            "no_gmm_executed": True,
            "warnings": warnings,
            "blockers": blockers,
        }
        summary_path = output_dir / "final_enhanced_cluster_profile_summary.json"
        write_json(summary_path, summary)
        write_report(report_path, summary, strategy_rows)

        artifact_paths = [
            output_dir / "global_erfm_statistics.json",
            output_dir / "cluster_profiles.json",
            output_dir / "cluster_profiles.csv",
            output_dir / "transformed_centroid_mean_validation.json",
            output_dir / "raw_mean_vs_representative_center.json",
            output_dir / "raw_mean_vs_representative_center.csv",
            output_dir / "pairwise_cluster_distinctness.json",
            output_dir / "segment_names.json",
            output_dir / "business_strategy_table.csv",
            summary_path,
            report_path,
            run_log_path,
        ]
        artifact_readback: Dict[str, bool] = {}
        for path in artifact_paths:
            artifact_readback[str(path.relative_to(project_root))] = path.exists() and path.stat().st_size > 0
            if path.suffix == ".json" and artifact_readback[str(path.relative_to(project_root))]:
                try:
                    json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    artifact_readback[str(path.relative_to(project_root))] = False
            if path.suffix == ".csv" and artifact_readback[str(path.relative_to(project_root))]:
                artifact_readback[str(path.relative_to(project_root))] = bool(path.read_text(encoding="utf-8").splitlines()[0].strip())
        summary["artifact_readback"] = artifact_readback
        summary["ready_for_gmm_stage"] = all(artifact_readback.values())
        summary["enhanced_model_branch_complete"] = summary["ready_for_gmm_stage"]
        write_json(summary_path, summary)
        write_report(report_path, summary, strategy_rows)
        if not summary["ready_for_gmm_stage"]:
            raise ValueError("Stage 16 artifact read-back failed.")

        print("AUTHORITY_AUDIT_PASS=True")
        print("TRANSFORMED_CENTROID_VALIDATION_PASS=True")
        print("STAGE14_PROFILE_CONSISTENCY_PASS=True")
        print("ENHANCED_MODEL_BRANCH_COMPLETE=True")
        print("READY_FOR_GMM_STAGE=True")
        return 0
    except Exception as error:
        blockers.append(str(error))
        failed = {"stage": 16, "authority": "ENHANCED_KMEANS_FINAL_CLUSTER_PROFILE", "enhanced_model_branch_complete": False, "ready_for_gmm_stage": False, "warnings": warnings, "blockers": blockers}
        write_json(project_root / "output" / "analysis" / "stage16" / "final_enhanced_cluster_profile_summary.json", failed)
        report_path.write_text("STAGE 16 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_GMM_STAGE: False\n", encoding="utf-8")
        print(f"STAGE 16 FAILED CLOSED: {error}")
        return 3
    finally:
        if cached_predictions is not None:
            cached_predictions.unpersist()
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    sys.exit(main())
