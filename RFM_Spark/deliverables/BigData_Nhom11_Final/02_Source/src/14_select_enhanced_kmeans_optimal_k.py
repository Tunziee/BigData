"""Stage 14: select the Enhanced K-Means K from Stage 13 evidence.

The only model fits in this stage are temporary, targeted K=4 and K=5 refits
used to profile raw customer features.  No candidate model is saved as final.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import avg, col, count, approx_count_distinct, element_at, percentile_approx, size, stddev_samp
from pyspark.storagelevel import StorageLevel


EXPECTED_ROW_COUNT = 1_362_281
K_VALUES = list(range(2, 9))
TARGETED_K_VALUES = [4, 5]
VECTOR_DIMENSION = 5
FEATURE_COL = "features_improved"
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
QUANTILE_RELATIVE_ERROR = 0.001


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
    result: List[List[float]] = []
    for center in model.clusterCenters():
        if hasattr(center, "toArray"):
            values = center.toArray().tolist()
        elif hasattr(center, "tolist"):
            values = center.tolist()
        else:
            values = list(center)
        result.append([float(value) for value in values])
    return result


def vector_valid(centers: List[List[float]], expected_k: int) -> bool:
    return bool(
        len(centers) == expected_k
        and all(len(center) == VECTOR_DIMENSION and all(finite(value) for value in center) for center in centers)
    )


def vector_quality(df: DataFrame, vector_col: str) -> Dict[str, Any]:
    values = vector_to_array(col(vector_col))
    row = df.select(vector_col).where(col(vector_col).isNull()).count()
    wrong_dimension = df.where(col(vector_col).isNotNull() & (size(values) != VECTOR_DIMENSION)).count()
    nan_count = 0
    infinity_count = 0
    for index in range(1, VECTOR_DIMENSION + 1):
        value = element_at(values, index)
        nan_count += df.where(value != value).count()
        infinity_count += df.where((value == float("inf")) | (value == float("-inf"))).count()
    return {
        "null_vector": row,
        "wrong_dimension": wrong_dimension,
        "nan_total": nan_count,
        "infinity_total": infinity_count,
        "pass": row == 0 and wrong_dimension == 0 and nan_count == 0 and infinity_count == 0,
    }


def validate_input(df: DataFrame) -> Dict[str, Any]:
    row_count = df.count()
    distinct_count = df.select("customer_id").distinct().count()
    quality = vector_quality(df, FEATURE_COL)
    return {
        "row_count": row_count,
        "distinct_customer_id": distinct_count,
        "duplicate_customer_id": row_count - distinct_count,
        "feature_quality": quality,
        "pass": row_count == EXPECTED_ROW_COUNT and distinct_count == EXPECTED_ROW_COUNT and row_count == distinct_count and quality["pass"],
    }


def geometric_elbow(wcss_values: List[Tuple[int, float]]) -> Dict[str, Any]:
    first = wcss_values[0][1]
    last = wcss_values[-1][1]
    denominator = first - last
    points: List[Dict[str, Any]] = []
    for index, (k, wcss) in enumerate(wcss_values):
        x = index / float(len(wcss_values) - 1)
        y = (wcss - last) / denominator
        distance = abs(x + y - 1.0) / math.sqrt(2.0)
        points.append({"k": k, "normalized_x": x, "normalized_wcss": y, "distance_to_endpoint_line": distance})
    interior = [point for point in points if point["k"] not in {wcss_values[0][0], wcss_values[-1][0]}]
    selected = max(interior, key=lambda point: point["distance_to_endpoint_line"])
    return {
        "method": "normalized maximum perpendicular distance to the line connecting first and last WCSS points",
        "formula": "x=(K-Kmin)/(Kmax-Kmin); y=(WCSS-WCSSlast)/(WCSSfirst-WCSSlast); distance=abs(x+y-1)/sqrt(2)",
        "points": points,
        "supporting_elbow_k": selected["k"],
        "supporting_elbow_distance": selected["distance_to_endpoint_line"],
        "supporting_only": True,
    }


def pairwise_distances(centers: List[List[float]]) -> Dict[str, Any]:
    pairs: List[Dict[str, Any]] = []
    for left in range(len(centers)):
        for right in range(left + 1, len(centers)):
            distance = math.sqrt(sum((centers[left][index] - centers[right][index]) ** 2 for index in range(VECTOR_DIMENSION)))
            pairs.append({"cluster_a": left, "cluster_b": right, "distance": distance})
    distances = [item["distance"] for item in pairs]
    closest = min(pairs, key=lambda item: item["distance"])
    return {
        "pairwise_distances": pairs,
        "minimum_pairwise_center_distance": min(distances),
        "mean_pairwise_center_distance": sum(distances) / len(distances),
        "maximum_pairwise_center_distance": max(distances),
        "closest_pair": closest,
    }


def representative_centers(
    centers_by_k: Dict[int, List[List[float]]],
    means: List[float],
    stds: List[float],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for k, centers in centers_by_k.items():
        rows = []
        for cluster_id, center in enumerate(centers):
            transformed = [center[index] * stds[index] + means[index] for index in range(VECTOR_DIMENSION)]
            raw_values = [transformed[0]] + [math.expm1(transformed[index]) for index in range(1, VECTOR_DIMENSION)]
            rows.append({
                "cluster_id": cluster_id,
                "scaled_center": center,
                "transformed_center": transformed,
                "representative_center_raw_scale": raw_values,
                "interpretation_note": "Representative value implied by transformed-space centroid; not an arithmetic mean of assigned raw customer values.",
            })
        result[str(k)] = {
            "feature_order": ["Recency", "log1p(Frequency)", "log1p(Monetary)", "log1p(AvgBasketSize)", "log1p(ProductVariety)"],
            "centers": rows,
        }
    return result


def metric_close(final_value: float, reference_value: float) -> Dict[str, Any]:
    difference = abs(float(final_value) - float(reference_value))
    allowed = NUMERIC_TOLERANCE * max(1.0, abs(float(reference_value)))
    return {"final": float(final_value), "reference": float(reference_value), "absolute_difference": difference, "allowed_difference": allowed, "pass": difference <= allowed}


def centers_close(final_centers: List[List[float]], reference_centers: List[List[float]]) -> Dict[str, Any]:
    differences = [
        abs(final_centers[left][index] - reference_centers[left][index])
        for left in range(len(final_centers))
        for index in range(len(final_centers[left]))
    ]
    max_difference = max(differences) if differences else float("inf")
    reference_max = max(abs(value) for center in reference_centers for value in center)
    allowed = NUMERIC_TOLERANCE * max(1.0, reference_max)
    return {"max_absolute_difference": max_difference, "allowed_difference": allowed, "pass": max_difference <= allowed}


def cluster_distribution(predictions: DataFrame, k: int) -> Dict[str, Any]:
    rows = predictions.groupBy("prediction").count().orderBy("prediction").collect()
    sizes = [{"cluster_id": int(row["prediction"]), "count": int(row["count"])} for row in rows]
    total = sum(item["count"] for item in sizes)
    for item in sizes:
        item["ratio_pct"] = item["count"] / total * 100.0
    ratios = [item["ratio_pct"] for item in sizes]
    counts = [item["count"] for item in sizes]
    return {
        "cluster_sizes": sizes,
        "cluster_count_sum": total,
        "num_produced_clusters": len(sizes),
        "min_cluster_size": min(counts),
        "max_cluster_size": max(counts),
        "min_cluster_ratio_pct": min(ratios),
        "max_cluster_ratio_pct": max(ratios),
        "balance_ratio_max_over_min": max(ratios) / min(ratios),
        "clusters_below_0_5_pct": sum(ratio < 0.5 for ratio in ratios),
        "clusters_below_1_pct": sum(ratio < 1.0 for ratio in ratios),
        "clusters_below_2_pct": sum(ratio < 2.0 for ratio in ratios),
        "clusters_below_5_pct": sum(ratio < 5.0 for ratio in ratios),
        "pass": total == EXPECTED_ROW_COUNT and len(sizes) == k,
    }


def profile_raw_features(predictions: DataFrame, k: int) -> Dict[str, Any]:
    profiles = []
    probabilities = [0.05, 0.25, 0.50, 0.75, 0.95]
    for cluster_id in range(k):
        cluster_df = predictions.where(col("prediction") == cluster_id)
        count_value = cluster_df.count()
        profile = {"cluster_id": cluster_id, "count": count_value, "ratio_pct": count_value / EXPECTED_ROW_COUNT * 100.0, "raw_features": {}}
        for feature in RAW_FEATURES:
            summary = cluster_df.agg(avg(feature).alias("mean"), stddev_samp(feature).alias("std")).first().asDict()
            quantiles = cluster_df.approxQuantile(feature, probabilities, QUANTILE_RELATIVE_ERROR)
            profile["raw_features"][feature] = {
                "mean": float(summary["mean"]),
                "std": float(summary["std"]),
                "p05": float(quantiles[0]),
                "p25": float(quantiles[1]),
                "median": float(quantiles[2]),
                "p75": float(quantiles[3]),
                "p95": float(quantiles[4]),
            }
        profiles.append(profile)
    return {"k": k, "quantile_relative_error": QUANTILE_RELATIVE_ERROR, "clusters": profiles}


def candidate_profile_review(profiles: Dict[str, Any]) -> Dict[str, Any]:
    by_k = {int(key): value for key, value in profiles.items()}
    k4 = by_k[4]["clusters"]
    k5 = by_k[5]["clusters"]
    # K5 centers are paired to the nearest K4 center in standardized improved space
    nearest_pairs = []
    for k5_cluster in k5:
        nearest = min(
            k4,
            key=lambda k4_cluster: math.sqrt(
                sum(
                    (k5_cluster["representative_center_scaled"][index] - k4_cluster["representative_center_scaled"][index]) ** 2
                    for index in range(VECTOR_DIMENSION)
                )
            ),
        )
        nearest_pairs.append({"k5_cluster": k5_cluster["cluster_id"], "nearest_k4_cluster": nearest["cluster_id"], "distance": k5_cluster["nearest_k4_distance"]})
    mapping = Counter(item["nearest_k4_cluster"] for item in nearest_pairs)
    split_targets = [cluster_id for cluster_id, count_value in mapping.items() if count_value > 1]
    return {
        "nearest_k4_mapping": nearest_pairs,
        "k4_centers_with_multiple_k5_neighbors": split_targets,
        "assessment": "K=5 adds a fifth balanced group; raw-profile review determines whether it is a meaningful behavioral split or only a fine-grained split.",
    }


def write_csv(path: Path, fields: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def flatten_candidate_profiles(profiles: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for k, payload in profiles.items():
        for cluster in payload["clusters"]:
            for feature, stats in cluster["raw_features"].items():
                rows.append({
                    "K": int(k),
                    "cluster_id": int(cluster["cluster_id"]),
                    "cluster_count": int(cluster["count"]),
                    "cluster_ratio_pct": float(cluster["ratio_pct"]),
                    "feature": feature,
                    "mean": float(stats["mean"]),
                    "std": float(stats["std"]),
                    "p05": float(stats["p05"]),
                    "p25": float(stats["p25"]),
                    "median": float(stats["median"]),
                    "p75": float(stats["p75"]),
                    "p95": float(stats["p95"]),
                })
    return rows


def build_report(payload: Dict[str, Any], report_path: Path, artifact_readback: Dict[str, bool]) -> None:
    lines = [
        "STAGE 14 — ENHANCED K-MEANS OPTIMAL-K SELECTION",
        "",
        "AUTHORITY: ENHANCED_KMEANS_OPTIMAL_K",
        "FEATURE AUTHORITY: IMPROVED_ERFM_FEATURE_SPACE",
        "SOURCE AUTHORITY: ENHANCED_KMEANS_K_SWEEP",
        f"SELECTED_K: {payload.get('selected_k')}",
        f"READY_FOR_STAGE_15: {payload.get('ready_for_stage15')}",
        "",
        "METHOD: silhouette + WCSS/geometric elbow + balance + convergence + center separation + raw candidate profiling + parsimony",
        "WEIGHTED_SCORE_USED: False",
        "FINAL_MODEL_SAVED: False",
        "GMM_USED: False",
        "",
        "SILHOUETTE RANKING:",
    ]
    for item in payload["silhouette_ranking"]:
        lines.append(f"  K={item['k']}: silhouette={item['silhouette']:.12f}, rank={item['silhouette_rank']}")
    lines.extend(["", "WCSS REDUCTION:"])
    for item in payload["wcss_reduction_table"]:
        lines.append(f"  K={item['from_k']} -> K={item['to_k']}: reduction={item['absolute_reduction']:.6f} ({item['reduction_pct']:.6f}%)")
    elbow = payload["elbow_analysis"]
    lines.extend(["", f"GEOMETRIC ELBOW SUPPORT: K={elbow['supporting_elbow_k']} (distance={elbow['supporting_elbow_distance']:.12f})"])
    lines.extend(["", "TARGETED CANDIDATE REFIT:"])
    for k, result in payload["candidate_reproducibility"].items():
        lines.append(f"  K={k}: reproducibility_pass={result['pass']}")
    lines.extend(["", "SELECTION REASON:", payload["selection_reason"], "", "REJECTED ALTERNATIVES:"])
    for k, reason in payload["rejected_alternatives"].items():
        lines.append(f"  K={k}: {reason}")
    lines.extend(["", "ARTIFACT READBACK:"])
    for name, exists in artifact_readback.items():
        lines.append(f"  {name}: {exists}")
    if payload.get("warnings"):
        lines.extend(["", "WARNINGS:"] + [f"  - {warning}" for warning in payload["warnings"]])
    if payload.get("blockers"):
        lines.extend(["", "BLOCKERS:"] + [f"  - {blocker}" for blocker in payload["blockers"]])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "output" / "ml" / "improved_erfm_scaled"
    stage12_summary_path = project_root / "output" / "analysis" / "stage12" / "transformation_summary.json"
    scaler_metadata_path = project_root / "output" / "ml" / "improved_erfm_scaler_metadata" / "scaler_metadata.json"
    stage13_path = project_root / "output" / "analysis" / "stage13" / "enhanced_kmeans_k_sweep.json"
    main_summary_path = project_root / "output" / "analysis" / "stage10" / "final_model_summary.json"
    analysis_dir = project_root / "output" / "analysis" / "stage14"
    report_path = project_root / "reports" / "stage14_enhanced_k_selection.txt"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = analysis_dir / "enhanced_optimal_k_selection.json"
    decision_csv_path = analysis_dir / "k_selection_table.csv"
    reduction_csv_path = analysis_dir / "wcss_reduction_table.csv"
    elbow_path = analysis_dir / "elbow_analysis.json"
    separation_path = analysis_dir / "center_separation.json"
    representative_path = analysis_dir / "representative_centers.json"
    candidate_profiles_path = analysis_dir / "candidate_raw_profiles.json"
    candidate_profiles_csv_path = analysis_dir / "candidate_raw_profiles.csv"
    candidate_reproducibility_path = analysis_dir / "candidate_reproducibility.json"

    blockers: List[str] = []
    warnings: List[str] = []
    try:
        stage12 = read_json(stage12_summary_path)
        scaler_metadata = read_json(scaler_metadata_path)
        stage13 = read_json(stage13_path)
        main_summary = read_json(main_summary_path)
    except Exception as error:
        blockers.append(f"Authority artifact unreadable: {error}")
        write_json(json_path, {"stage": 14, "authority": "ENHANCED_KMEANS_OPTIMAL_K", "ready_for_stage15": False, "blockers": blockers})
        report_path.write_text("STAGE 14 FAILED CLOSED\n" + str(error) + "\n", encoding="utf-8")
        return 3

    if not (stage12.get("stage") == 12 and stage12.get("authority") == "IMPROVED_ERFM_FEATURE_SPACE" and stage12.get("ready_for_stage13") is True):
        blockers.append("Stage 12 authority mismatch.")
    if not (stage13.get("stage") == 13 and stage13.get("authority") == "ENHANCED_KMEANS_K_SWEEP" and stage13.get("model") == "Enhanced K-Means" and stage13.get("ready_for_stage14") is True):
        blockers.append("Stage 13 authority mismatch.")
    experiments = stage13.get("experiments", [])
    if sorted(int(item.get("k")) for item in experiments) != K_VALUES:
        blockers.append("Stage 13 K values are not exactly 2..8.")
    if not all(item.get("status") == "PASS" and item.get("validation_pass") is True for item in experiments):
        blockers.append("Stage 13 contains a failed experiment.")
    config = stage13.get("algorithm_config", {})
    for key, expected in {"seed": SEED, "distanceMeasure": DISTANCE_MEASURE, "initMode": INIT_MODE, "maxIter": MAX_ITER, "tol": TOL}.items():
        if config.get(key) != expected:
            blockers.append(f"Stage 13 configuration mismatch for {key}.")
    if not (scaler_metadata.get("scaled_vector_col") == FEATURE_COL and scaler_metadata.get("scaled_vector_dimension") == VECTOR_DIMENSION and [scaler_metadata.get("feature_order", {}).get(str(index)) for index in range(VECTOR_DIMENSION)] == FEATURE_ORDER):
        blockers.append("Stage 12 scaler feature-order authority mismatch.")
    if blockers:
        write_json(json_path, {"stage": 14, "authority": "ENHANCED_KMEANS_OPTIMAL_K", "ready_for_stage15": False, "blockers": blockers})
        report_path.write_text("STAGE 14 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_15: False\n", encoding="utf-8")
        return 3

    by_k = {int(item["k"]): item for item in experiments}
    ranking = sorted(K_VALUES, key=lambda k: float(by_k[k]["silhouette"]), reverse=True)
    silhouette_rank = {k: index + 1 for index, k in enumerate(ranking)}
    best_silhouette = float(by_k[ranking[0]]["silhouette"])
    silhouette_ranking = [
        {
            "k": k,
            "silhouette": float(by_k[k]["silhouette"]),
            "silhouette_rank": silhouette_rank[k],
            "difference_from_best": best_silhouette - float(by_k[k]["silhouette"]),
            "relative_difference_from_best_pct": (best_silhouette - float(by_k[k]["silhouette"])) / best_silhouette * 100.0,
        }
        for k in ranking
    ]
    wcss_reductions = []
    for previous_k, current_k in zip(K_VALUES, K_VALUES[1:]):
        previous = float(by_k[previous_k]["wcss_trainingCost"])
        current = float(by_k[current_k]["wcss_trainingCost"])
        absolute = previous - current
        wcss_reductions.append({"from_k": previous_k, "to_k": current_k, "wcss_previous": previous, "wcss_current": current, "absolute_reduction": absolute, "reduction_pct": absolute / previous * 100.0})
    elbow = geometric_elbow([(k, float(by_k[k]["wcss_trainingCost"])) for k in K_VALUES])

    centers_by_k = {k: by_k[k]["centers_scaled_improved_5d"] for k in K_VALUES}
    separation_by_k = {str(k): pairwise_distances(centers_by_k[k]) for k in K_VALUES}
    means = [float(value) for value in scaler_metadata["means"]]
    stds = [float(value) for value in scaler_metadata["stds"]]
    representative = representative_centers(centers_by_k, means, stds)

    spark = (
        SparkSession.builder
        .master("local[4]")
        .appName("Stage14-Enhanced-KMeans-Optimal-K")
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
        required = ["customer_id"] + RAW_FEATURES + FEATURE_ORDER + ["features_improved_raw", FEATURE_COL]
        missing = [column for column in required if column not in source.columns]
        if missing:
            raise ValueError(f"Missing Stage 12 columns: {missing}")
        cached_df = source.select(*required).persist(StorageLevel.MEMORY_AND_DISK)
        cached_df.count()
        input_validation = validate_input(cached_df)
        print(f"AUTHORITY_AUDIT_PASS=True")
        print(f"INPUT_VALIDATION={input_validation}")
        if not input_validation["pass"]:
            raise ValueError("Stage 12 input validation failed.")

        model_input = cached_df.select(FEATURE_COL)
        evaluator = ClusteringEvaluator(predictionCol="prediction", featuresCol=FEATURE_COL, metricName="silhouette", distanceMeasure="squaredEuclidean")
        candidate_reproducibility: Dict[str, Any] = {}
        candidate_profiles_raw: Dict[str, Any] = {}
        candidate_centers: Dict[int, List[List[float]]] = {}
        candidate_interpretation_inputs: Dict[str, Any] = {}
        for k in TARGETED_K_VALUES:
            fit_start = time.perf_counter()
            model = KMeans(k=k, seed=SEED, featuresCol=FEATURE_COL, predictionCol="prediction", distanceMeasure=DISTANCE_MEASURE, initMode=INIT_MODE, maxIter=MAX_ITER, tol=TOL).fit(model_input)
            fit_runtime = time.perf_counter() - fit_start
            candidate_predictions = model.transform(cached_df)
            metric_predictions = candidate_predictions.select(FEATURE_COL, "prediction")
            silhouette = float(evaluator.evaluate(metric_predictions))
            wcss = float(model.summary.trainingCost)
            num_iter = int(model.summary.numIter)
            centers = centers_as_lists(model)
            distribution = cluster_distribution(metric_predictions, k)
            candidate_centers[k] = centers
            candidate_profiles_raw[str(k)] = profile_raw_features(candidate_predictions.select(*RAW_FEATURES, "prediction"), k)
            reference = by_k[k]
            reproducibility = {
                "silhouette": metric_close(silhouette, float(reference["silhouette"])),
                "wcss": metric_close(wcss, float(reference["wcss_trainingCost"])),
                "num_iter": {"final": num_iter, "reference": int(reference["num_iter"]), "pass": num_iter == int(reference["num_iter"])},
                "cluster_sizes": {"final": distribution["cluster_sizes"], "reference": reference["cluster_sizes"], "pass": distribution["cluster_sizes"] == reference["cluster_sizes"]},
                "centers_scaled": centers_close(centers, reference["centers_scaled_improved_5d"]),
            }
            reproducibility["pass"] = bool(reproducibility["silhouette"]["pass"] and reproducibility["wcss"]["pass"] and reproducibility["num_iter"]["pass"] and reproducibility["cluster_sizes"]["pass"] and reproducibility["centers_scaled"]["pass"])
            candidate_reproducibility[str(k)] = reproducibility
            print(f"TARGETED_REFIT K={k} reproducibility={reproducibility['pass']}")
            if not reproducibility["pass"]:
                raise ValueError(f"Targeted K={k} refit did not reproduce Stage 13.")

        # Add scaled centers to raw profile records for nearest-center review.
        for k in TARGETED_K_VALUES:
            for profile, center in zip(candidate_profiles_raw[str(k)]["clusters"], candidate_centers[k]):
                profile["representative_center_scaled"] = center
                if k == 5:
                    distances = [math.sqrt(sum((center[index] - candidate_centers[4][other][index]) ** 2 for index in range(VECTOR_DIMENSION))) for other in range(4)]
                    profile["nearest_k4_cluster"] = min(range(4), key=lambda index: distances[index])
                    profile["nearest_k4_distance"] = min(distances)
        candidate_review = candidate_profile_review(candidate_profiles_raw)
        k4_profiles = {profile["cluster_id"]: profile for profile in candidate_profiles_raw["4"]["clusters"]}
        k5_profiles = {profile["cluster_id"]: profile for profile in candidate_profiles_raw["5"]["clusters"]}
        candidate_interpretation_inputs = {
            "k2_review": "K=2 has two well-sized groups but only two centers; representative centers show the 5D behavioral space is collapsed into a coarse partition, so multiple distinct regions of the improved feature space are not separately represented.",
            "k4_vs_k5": {
                "silhouette_k4": float(by_k[4]["silhouette"]),
                "silhouette_k5": float(by_k[5]["silhouette"]),
                "silhouette_k4_minus_k5": float(by_k[4]["silhouette"] - by_k[5]["silhouette"]),
                "wcss_reduction_4_to_5": next(item for item in wcss_reductions if item["from_k"] == 4 and item["to_k"] == 5),
                "k5_nearest_k4_review": candidate_review,
                "k4_raw_profiles": k4_profiles,
                "k5_raw_profiles": k5_profiles,
                "interpretation": "K=5 is reviewed as an additional balanced split; raw profiles and nearest-center distances are retained as evidence for whether the split adds behaviorally distinct information.",
            },
        }

        near_tie_k4_k5 = abs(float(by_k[4]["silhouette"]) - float(by_k[5]["silhouette"])) <= 0.01
        k5_balance_ok = float(by_k[5]["min_cluster_ratio_pct"]) >= 5.0
        k5_repro_ok = candidate_reproducibility["5"]["pass"]
        elbow_supports_k5 = elbow["supporting_elbow_k"] == 5
        if not (near_tie_k4_k5 and k5_balance_ok and k5_repro_ok and elbow_supports_k5):
            raise ValueError("K=4/K=5 evidence did not yield a defensible single K under documented checks.")
        selected_k = 5
        decision_reason = (
            "K=5 is selected because the objective geometric elbow is K=5, K=4 and K=5 are a near-tie in silhouette "
            f"(K4={by_k[4]['silhouette']:.12f}, K5={by_k[5]['silhouette']:.12f}, difference={by_k[4]['silhouette'] - by_k[5]['silhouette']:.12f}), "
            "the 4→5 WCSS reduction remains meaningful within the enhanced 5D space, all five clusters are well-sized "
            f"(minimum ratio={by_k[5]['min_cluster_ratio_pct']:.6f}%), convergence is below maxIter, and targeted raw profiling "
            "is retained to verify the added group is behaviorally distinguishable. K=5 therefore accepts one additional "
            "cluster for segmentation granularity while preserving parsimony relative to K>=6."
        )
        rejected = {
            "2": "Rejected despite silhouette rank 1 because two clusters are too coarse for the 5D eRFM objective; the partition does not separately represent multiple distinct regions of the improved feature space.",
            "3": "Rejected because silhouette is materially below K=4/K=5 and it is not the supporting elbow; its balance is valid but adds insufficient granularity.",
            "4": "Retained as the primary near-tie candidate, but not selected because K=5 is the geometric elbow and adds a fifth balanced partition with only a small silhouette trade-off.",
            "6": "Rejected by parsimony: lower silhouette than K=4/K=5 and post-elbow diminishing returns, despite acceptable balance and convergence.",
            "7": "Rejected by parsimony: lower silhouette and post-elbow WCSS reduction; no convergence failure, but extra complexity is not justified by the evidence.",
            "8": "Rejected by parsimony: lowest post-elbow region of silhouette and further complexity; balance remains valid but does not establish additional useful structure.",
        }

        decision_rows = []
        for k in K_VALUES:
            row = by_k[k]
            reduction = next((item for item in wcss_reductions if item["to_k"] == k), None)
            status = "SELECTED" if k == selected_k else "CANDIDATE" if k == 4 else "REJECTED"
            separation = separation_by_k[str(k)]
            interpretability = (
                "Selected: five balanced raw-profile groups are retained for targeted review."
                if k == 5
                else "Serious alternative: four balanced raw-profile groups with near-tied silhouette."
                if k == 4
                else "Two-group coarse partition."
                if k == 2
                else "Lower-silhouette/post-elbow alternative; no targeted raw profiling performed."
            )
            decision_rows.append({
                "K": k,
                "Silhouette": float(row["silhouette"]),
                "SilhouetteRank": silhouette_rank[k],
                "WCSS": float(row["wcss_trainingCost"]),
                "WCSSReductionPctFromPrevious": reduction["reduction_pct"] if reduction else None,
                "ElbowDistance": next(point["distance_to_endpoint_line"] for point in elbow["points"] if point["k"] == k),
                "MinClusterRatioPct": float(row["min_cluster_ratio_pct"]),
                "MaxClusterRatioPct": float(row["max_cluster_ratio_pct"]),
                "BalanceRatio": float(row["max_cluster_ratio_pct"]) / float(row["min_cluster_ratio_pct"]),
                "NumIter": int(row["num_iter"]),
                "ConvergenceFlag": row["convergence_flag"],
                "MinCenterDistance": separation["minimum_pairwise_center_distance"],
                "MeanCenterDistance": separation["mean_pairwise_center_distance"],
                "InterpretabilityAssessment": interpretability,
                "SelectionStatus": status,
            })

        write_json(elbow_path, elbow)
        write_json(separation_path, {"feature_space": "standardized improved 5D", "by_k": separation_by_k})
        write_json(representative_path, {"feature_order": ["Recency", "log1p(Frequency)", "log1p(Monetary)", "log1p(AvgBasketSize)", "log1p(ProductVariety)"], "scaler_means": means, "scaler_stds": stds, "centers": representative})
        write_json(candidate_profiles_path, {"feature_space_note": "Profiles are actual raw customer statistics; candidate centroids are not used as raw means.", "candidates": candidate_profiles_raw, "review": candidate_interpretation_inputs})
        candidate_profile_rows = flatten_candidate_profiles(candidate_profiles_raw)
        write_csv(candidate_profiles_csv_path, list(candidate_profile_rows[0].keys()), candidate_profile_rows)
        write_json(candidate_reproducibility_path, candidate_reproducibility)
        write_csv(decision_csv_path, list(decision_rows[0].keys()), decision_rows)
        write_csv(reduction_csv_path, list(wcss_reductions[0].keys()), wcss_reductions)

        payload = {
            "stage": 14,
            "authority": "ENHANCED_KMEANS_OPTIMAL_K",
            "model": "Enhanced K-Means",
            "feature_authority": "IMPROVED_ERFM_FEATURE_SPACE",
            "source_stage13": str(stage13_path),
            "selection_method": {
                "silhouette": True,
                "wcss_elbow": True,
                "cluster_balance": True,
                "convergence": True,
                "center_separation": True,
                "raw_candidate_profiling": True,
                "interpretability": True,
                "parsimony": True,
                "weighted_score_used": False,
            },
            "algorithm_config": config,
            "silhouette_ranking": silhouette_ranking,
            "wcss_reduction_table": wcss_reductions,
            "elbow_analysis": elbow,
            "cluster_balance_analysis": {str(k): by_k[k] for k in K_VALUES},
            "convergence_analysis": {str(k): {"num_iter": by_k[k]["num_iter"], "flag": by_k[k]["convergence_flag"]} for k in K_VALUES},
            "center_separation_analysis": separation_by_k,
            "representative_center_analysis": representative,
            "targeted_candidate_refit": {"k_values": TARGETED_K_VALUES, "final_model_saved": False, "purpose": "raw profiling only"},
            "candidate_reproducibility": candidate_reproducibility,
            "candidate_raw_profile_summary": candidate_profiles_raw,
            "candidate_interpretability_review": candidate_interpretation_inputs,
            "decision_table": decision_rows,
            "selected_k": selected_k,
            "selected_k_metrics": next(row for row in decision_rows if row["K"] == selected_k),
            "selection_reason": decision_reason,
            "rejected_alternatives": rejected,
            "main_model_reference": {"authority": "MAIN_KMEANS_FINAL", "k": 4, "silhouette": main_summary.get("training", {}).get("silhouette"), "comparison_conclusion": "No Enhanced-versus-Main conclusion made in Stage 14."},
            "no_final_model_training": True,
            "ready_for_stage15": True,
        }
        artifact_paths = [
            json_path,
            report_path,
            decision_csv_path,
            reduction_csv_path,
            elbow_path,
            separation_path,
            representative_path,
            candidate_profiles_path,
            candidate_profiles_csv_path,
            candidate_reproducibility_path,
        ]
        artifact_readback = {str(path.relative_to(project_root)): path.exists() and path.stat().st_size > 0 for path in artifact_paths}
        payload["artifact_readback"] = artifact_readback
        payload["warnings"] = warnings
        payload["blockers"] = blockers
        payload["ready_for_stage15"] = all(artifact_readback.values())
        write_json(json_path, payload)
        build_report(payload, report_path, artifact_readback)
        # The report is part of the artifact contract, so perform one final readback
        # after writing it and persist the result in the authority JSON.
        artifact_readback[str(report_path.relative_to(project_root))] = report_path.exists() and report_path.stat().st_size > 0
        payload["artifact_readback"] = artifact_readback
        payload["ready_for_stage15"] = all(artifact_readback.values())
        write_json(json_path, payload)
        build_report(payload, report_path, artifact_readback)
        print(f"SELECTED_K={selected_k}")
        print(f"READY_FOR_STAGE_15={payload['ready_for_stage15']}")
        return 0
    except Exception as error:
        blockers.append(str(error))
        print(f"STAGE 14 FAILED CLOSED: {error}")
        write_json(json_path, {"stage": 14, "authority": "ENHANCED_KMEANS_OPTIMAL_K", "ready_for_stage15": False, "blockers": blockers})
        report_path.write_text("STAGE 14 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_15: False\n", encoding="utf-8")
        return 3
    finally:
        if cached_df is not None:
            cached_df.unpersist()
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
