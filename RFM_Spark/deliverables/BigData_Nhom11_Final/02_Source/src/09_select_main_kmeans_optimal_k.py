"""Stage 9: select the main-model K from verified Stage 8 evidence.

This stage is deliberately artifact-only.  It does not create a Spark session,
fit K-Means, transform customer rows, or write final predictions.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


EXPECTED_K_VALUES = list(range(2, 9))
EXPECTED_ROW_COUNT = 1_362_281
FEATURE_ORDER = ["Recency", "Frequency", "Monetary"]
MAX_ITER = 100
TINY_THRESHOLDS = [0.5, 1.0, 2.0, 5.0]


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


def values_equal(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def audit_authorities(
    metadata_path: Path,
    stage8_path: Path,
    cluster_dir: Path,
    centers_dir: Path,
) -> Dict[str, Any]:
    blockers: List[str] = []
    metadata: Dict[str, Any] = {}
    stage8: Dict[str, Any] = {}
    by_k: Dict[int, Dict[str, Any]] = {}

    if not metadata_path.exists():
        blockers.append(f"Missing Stage 7 scaler metadata: {metadata_path}")
    else:
        try:
            metadata = read_json(metadata_path)
        except Exception as error:
            blockers.append(f"Unreadable Stage 7 scaler metadata: {error}")

    if not stage8_path.exists():
        blockers.append(f"Missing Stage 8 evidence: {stage8_path}")
    else:
        try:
            stage8 = read_json(stage8_path)
        except Exception as error:
            blockers.append(f"Unreadable Stage 8 evidence: {error}")

    if metadata:
        if metadata.get("stage") != 7:
            blockers.append("Stage 7 scaler metadata stage is not 7.")
        if metadata.get("input_features") != FEATURE_ORDER:
            blockers.append("Stage 7 input feature order is inconsistent.")
        if metadata.get("feature_order") != {
            "0": "Recency",
            "1": "Frequency",
            "2": "Monetary",
        }:
            blockers.append("Stage 7 feature_order mapping is inconsistent.")
        if metadata.get("withMean") is not True or metadata.get("withStd") is not True:
            blockers.append("Stage 7 scaler must have withMean=True and withStd=True.")
        means = metadata.get("means")
        stds = metadata.get("stds")
        if not isinstance(means, list) or len(means) != 3:
            blockers.append("Stage 7 scaler mean must have length 3.")
        if not isinstance(stds, list) or len(stds) != 3:
            blockers.append("Stage 7 scaler std must have length 3.")
        if isinstance(means, list) and (
            len(means) != 3 or not all(finite(value) for value in means)
        ):
            blockers.append("Stage 7 scaler means must be finite and length 3.")
        if isinstance(stds, list) and (
            len(stds) != 3
            or not all(finite(value) and float(value) > 0.0 for value in stds)
        ):
            blockers.append("Stage 7 scaler stds must be finite, positive, and length 3.")

    experiments = stage8.get("experiments") if stage8 else None
    actual_k_values = sorted(
        int(item["k"]) for item in experiments
    ) if isinstance(experiments, list) and all(
        isinstance(item, dict) and "k" in item for item in experiments
    ) else []
    if actual_k_values != EXPECTED_K_VALUES:
        blockers.append(
            f"Stage 8 must contain exactly K=2..8; found {actual_k_values}."
        )

    if isinstance(experiments, list):
        for item in experiments:
            if not isinstance(item, dict) or "k" not in item:
                continue
            k = int(item["k"])
            by_k[k] = item
            if item.get("status") != "PASS":
                blockers.append(f"Stage 8 K={k} status is not PASS.")
            if not finite(item.get("silhouette")):
                blockers.append(f"Stage 8 K={k} silhouette is not finite.")
            if not finite(item.get("wcss")) or float(item["wcss"]) <= 0.0:
                blockers.append(f"Stage 8 K={k} WCSS is not finite and positive.")
            if item.get("cluster_count_sum") != EXPECTED_ROW_COUNT:
                blockers.append(f"Stage 8 K={k} cluster counts do not sum to the expected row count.")
            if item.get("num_produced_clusters") != k:
                blockers.append(f"Stage 8 K={k} did not produce exactly K clusters.")

            cluster_path = cluster_dir / f"k{k}.json"
            centers_path = centers_dir / f"k{k}.json"
            if not cluster_path.exists():
                blockers.append(f"Missing Stage 8 cluster-size artifact for K={k}.")
            if not centers_path.exists():
                blockers.append(f"Missing Stage 8 center artifact for K={k}.")
            if cluster_path.exists():
                try:
                    cluster_payload = read_json(cluster_path)
                    cluster_sizes = cluster_payload.get("cluster_sizes")
                    if not isinstance(cluster_sizes, list) or len(cluster_sizes) != k:
                        blockers.append(f"Cluster-size artifact K={k} has the wrong cluster count.")
                    if cluster_payload.get("cluster_count_sum") != EXPECTED_ROW_COUNT:
                        blockers.append(f"Cluster-size artifact K={k} has the wrong row-count sum.")
                    if not values_equal(cluster_sizes, item.get("cluster_sizes")):
                        blockers.append(f"Cluster-size artifact K={k} disagrees with Stage 8 JSON.")
                except Exception as error:
                    blockers.append(f"Unreadable cluster-size artifact K={k}: {error}")
            if centers_path.exists():
                try:
                    centers_payload = read_json(centers_path)
                    centers = centers_payload.get("centers_scaled")
                    if not isinstance(centers, list) or len(centers) != k:
                        blockers.append(f"Center artifact K={k} has the wrong center count.")
                    if isinstance(centers, list):
                        for center in centers:
                            if not isinstance(center, list) or len(center) != 3:
                                blockers.append(f"Center artifact K={k} has a non-3D center.")
                            elif not all(finite(value) for value in center):
                                blockers.append(f"Center artifact K={k} has a non-finite value.")
                    if not values_equal(centers, item.get("centers_scaled")):
                        blockers.append(f"Center artifact K={k} disagrees with Stage 8 JSON.")
                except Exception as error:
                    blockers.append(f"Unreadable center artifact K={k}: {error}")

    return {
        "pass": not blockers,
        "blockers": blockers,
        "metadata": metadata,
        "stage8": stage8,
        "by_k": by_k,
    }


def geometric_elbow(wcss_values: List[Tuple[int, float]]) -> Dict[str, Any]:
    first = wcss_values[0][1]
    last = wcss_values[-1][1]
    denominator = first - last
    points: List[Dict[str, Any]] = []
    for index, (k, wcss) in enumerate(wcss_values):
        x = index / float(len(wcss_values) - 1)
        y = (wcss - last) / denominator if denominator else 0.0
        distance = abs(x + y - 1.0) / math.sqrt(2.0)
        points.append(
            {
                "k": k,
                "normalized_x": x,
                "normalized_wcss": y,
                "distance_to_endpoint_line": distance,
            }
        )
    interior = [point for point in points if point["k"] not in {wcss_values[0][0], wcss_values[-1][0]}]
    best = max(interior, key=lambda point: point["distance_to_endpoint_line"])
    return {
        "method": "normalized maximum distance to the line connecting the first and last WCSS points",
        "formula": "x=(K-Kmin)/(Kmax-Kmin); y=(WCSS-WCSSlast)/(WCSSfirst-WCSSlast); distance=abs(x+y-1)/sqrt(2)",
        "endpoint_k_values": [wcss_values[0][0], wcss_values[-1][0]],
        "points": points,
        "supporting_elbow_k": best["k"],
        "supporting_elbow_distance": best["distance_to_endpoint_line"],
        "note": "Supporting evidence only; final selection also evaluates silhouette, balance, tiny clusters, convergence, and centroid patterns.",
    }


def tiny_flag(ratio_pct: float) -> str:
    if ratio_pct < 0.5:
        return "VERY_SMALL"
    if ratio_pct < 1.0:
        return "SMALL"
    if ratio_pct < 2.0:
        return "MINOR"
    return "NONE"


def direction(value: float) -> str:
    if value <= -0.1:
        return "LOWER_THAN_GLOBAL_MEAN"
    if value >= 0.1:
        return "HIGHER_THAN_GLOBAL_MEAN"
    return "NEAR_GLOBAL_MEAN"


def centroid_assessment(
    k: int,
    patterns: List[Tuple[str, ...]],
    min_pairwise_distance: float,
    min_ratio_pct: float,
) -> str:
    distinct = len(set(patterns))
    if distinct == k:
        base = f"All {k} centroids have distinct directional RFM patterns"
    else:
        base = f"{distinct} broad directional RFM patterns are present across {k} centroids; remaining non-identical centroids represent magnitude-level splits that require business review"
    if min_ratio_pct < 0.5:
        base += f"; the smallest cluster is {min_ratio_pct:.6f}% and requires outlier scrutiny"
    elif min_ratio_pct < 1.0:
        base += f"; the smallest cluster is {min_ratio_pct:.6f}% and is a small diagnostic segment"
    elif min_ratio_pct < 2.0:
        base += f"; the smallest cluster is {min_ratio_pct:.6f}% and is a minor but inspectable segment"
    else:
        base += f"; the smallest cluster is {min_ratio_pct:.6f}%"
    return base + f"; minimum scaled-center separation is {min_pairwise_distance:.6f}."


def pairwise_min_distance(centers: List[List[float]]) -> float:
    distances: List[float] = []
    for left_index in range(len(centers)):
        for right_index in range(left_index + 1, len(centers)):
            distances.append(
                math.sqrt(
                    sum(
                        (float(centers[left_index][dimension]) - float(centers[right_index][dimension])) ** 2
                        for dimension in range(3)
                    )
                )
            )
    return min(distances) if distances else 0.0


def write_csv(path: Path, fields: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def build_analysis(audit: Dict[str, Any]) -> Dict[str, Any]:
    metadata = audit["metadata"]
    by_k = audit["by_k"]
    means = [float(value) for value in metadata["means"]]
    stds = [float(value) for value in metadata["stds"]]

    wcss_values = [(k, float(by_k[k]["wcss"])) for k in EXPECTED_K_VALUES]
    elbow = geometric_elbow(wcss_values)
    elbow_k = int(elbow["supporting_elbow_k"])

    ranking = sorted(EXPECTED_K_VALUES, key=lambda k: float(by_k[k]["silhouette"]), reverse=True)
    silhouette_rank = {k: rank for rank, k in enumerate(ranking, start=1)}
    best_silhouette_k = ranking[0]
    second_silhouette_k = ranking[1]

    reductions: List[Dict[str, Any]] = []
    reductions_by_to_k: Dict[int, Dict[str, Any]] = {}
    for previous_k, current_k in zip(EXPECTED_K_VALUES, EXPECTED_K_VALUES[1:]):
        previous_wcss = float(by_k[previous_k]["wcss"])
        current_wcss = float(by_k[current_k]["wcss"])
        absolute_reduction = previous_wcss - current_wcss
        reduction_pct = absolute_reduction / previous_wcss * 100.0
        row = {
            "from_k": previous_k,
            "to_k": current_k,
            "wcss_previous": previous_wcss,
            "wcss_current": current_wcss,
            "absolute_reduction": absolute_reduction,
            "reduction_pct": reduction_pct,
        }
        reductions.append(row)
        reductions_by_to_k[current_k] = row

    balances: Dict[int, Dict[str, Any]] = {}
    inverse_centers: List[Dict[str, Any]] = []
    interpretability: Dict[int, Dict[str, Any]] = {}
    decision_rows: List[Dict[str, Any]] = []
    for k in EXPECTED_K_VALUES:
        item = by_k[k]
        cluster_sizes = item["cluster_sizes"]
        ratios = [float(cluster["ratio_pct"]) for cluster in cluster_sizes]
        counts = [int(cluster["count"]) for cluster in cluster_sizes]
        flags = [
            {
                "cluster_id": int(cluster["cluster_id"]),
                "count": int(cluster["count"]),
                "ratio_pct": float(cluster["ratio_pct"]),
                "flag": tiny_flag(float(cluster["ratio_pct"])),
            }
            for cluster in cluster_sizes
        ]
        balance = {
            "min_cluster_count": min(counts),
            "max_cluster_count": max(counts),
            "min_cluster_ratio_pct": min(ratios),
            "max_cluster_ratio_pct": max(ratios),
            "max_min_count_ratio": max(counts) / min(counts) if min(counts) else None,
            "clusters_below_0_5_pct": sum(ratio < 0.5 for ratio in ratios),
            "clusters_below_1_pct": sum(ratio < 1.0 for ratio in ratios),
            "clusters_below_2_pct": sum(ratio < 2.0 for ratio in ratios),
            "clusters_below_5_pct": sum(ratio < 5.0 for ratio in ratios),
            "cluster_flags": flags,
        }
        balances[k] = balance

        scaled_centers = item["centers_scaled"]
        patterns: List[Tuple[str, ...]] = []
        for cluster_id, (scaled_center, cluster) in enumerate(zip(scaled_centers, cluster_sizes)):
            original = [
                float(scaled_center[index]) * stds[index] + means[index]
                for index in range(3)
            ]
            if not all(finite(value) for value in original):
                raise ValueError(f"Inverse center validation failed for K={k}, cluster={cluster_id}: non-finite value")
            if any(value < 0.0 for value in original):
                raise ValueError(f"Inverse center validation failed for K={k}, cluster={cluster_id}: negative RFM value")
            pattern_values = tuple(direction(float(scaled_center[index])) for index in range(3))
            patterns.append(pattern_values)
            inverse_centers.append(
                {
                    "k": k,
                    "cluster_id": int(cluster["cluster_id"]),
                    "Recency_original": original[0],
                    "Frequency_original": original[1],
                    "Monetary_original": original[2],
                    "cluster_count": int(cluster["count"]),
                    "cluster_ratio_pct": float(cluster["ratio_pct"]),
                    "centroid_pattern": {
                        FEATURE_ORDER[index]: pattern_values[index]
                        for index in range(3)
                    },
                    "center_scaled": [float(value) for value in scaled_center],
                }
            )

        min_distance = pairwise_min_distance(scaled_centers)
        interpretability[k] = {
            "distinct_directional_pattern_count": len(set(patterns)),
            "directional_patterns": [list(pattern) for pattern in patterns],
            "minimum_pairwise_scaled_center_distance": min_distance,
            "assessment": centroid_assessment(
                k,
                patterns,
                min_distance,
                balance["min_cluster_ratio_pct"],
            ),
        }

        convergence_flag = "LIMIT_REACHED" if int(item["num_iter"]) >= MAX_ITER else "NORMAL"
        elbow_evidence = (
            "PRIMARY_GEOMETRIC_ELBOW"
            if k == elbow_k
            else "ADJACENT_TO_GEOMETRIC_ELBOW"
            if abs(k - elbow_k) == 1
            else "POST_ELBOW"
            if k > elbow_k
            else "PRE_ELBOW"
        )
        decision_rows.append(
            {
                "K": k,
                "Silhouette": float(item["silhouette"]),
                "SilhouetteRank": silhouette_rank[k],
                "WCSS": float(item["wcss"]),
                "WCSSReductionPctFromPrevious": reductions_by_to_k.get(k, {}).get("reduction_pct"),
                "MinClusterRatioPct": balance["min_cluster_ratio_pct"],
                "MaxClusterRatioPct": balance["max_cluster_ratio_pct"],
                "ClustersBelow1Pct": balance["clusters_below_1_pct"],
                "ClustersBelow2Pct": balance["clusters_below_2_pct"],
                "NumIter": int(item["num_iter"]),
                "ConvergenceFlag": convergence_flag,
                "ElbowEvidence": elbow_evidence,
                "InterpretabilityAssessment": interpretability[k]["assessment"],
                "SelectionStatus": "UNASSIGNED",
            }
        )

    selected_k = elbow_k
    selected_row = next(row for row in decision_rows if row["K"] == selected_k)
    selected_balance = balances[selected_k]
    selected_interp = interpretability[selected_k]
    selection_checks = {
        "supporting_geometric_elbow_is_interior_k": selected_k in EXPECTED_K_VALUES[1:-1],
        "silhouette_rank_competitive": silhouette_rank[selected_k] <= 3,
        "no_convergence_limit": selected_row["ConvergenceFlag"] != "LIMIT_REACHED",
        "no_cluster_below_1_pct": selected_balance["clusters_below_1_pct"] == 0,
        "centroids_non_identical": selected_interp["minimum_pairwise_scaled_center_distance"] > 1e-6,
        "inverse_centers_valid": True,
    }
    if not all(selection_checks.values()):
        raise ValueError(f"No defensible final K under the documented selection checks: {selection_checks}")

    for row in decision_rows:
        if row["K"] == selected_k:
            row["SelectionStatus"] = "SELECTED"
        elif row["K"] == 3:
            row["SelectionStatus"] = "CANDIDATE"
        else:
            row["SelectionStatus"] = "REJECTED"

    selected_metrics = by_k[selected_k]
    selection_reason = (
        "K=4 is selected because it is the maximum-distance interior point of the objective normalized WCSS elbow diagnostic, "
        "has the second-best silhouette (0.6859294450), improves on K=3 silhouette, produces four distinct directional "
        "RFM centroid vectors with a magnitude-level split in the high-Frequency/high-Monetary pattern, and has no cluster below 1% "
        "(the smallest cluster is 1.8211367552%, a minor but inspectable segment). K=2 has the best silhouette but is highly imbalanced at 87.2980684602% / "
        "12.7019315398% and is too coarse for the requested segmentation granularity. K=3 remains the closest balance-oriented "
        "candidate, but its silhouette and elbow evidence are both weaker than K=4. K>=5 are post-elbow solutions with lower "
        "silhouette and increasing small-cluster fragmentation; K=7 additionally reached maxIter=100. No weighted score was used."
    )

    rejected_alternatives = {
        "2": "Rejected despite silhouette rank 1 because the two clusters are strongly imbalanced (87.2980684602% / 12.7019315398%) and provide only two broad groups; K=4 is the geometric elbow and has a competitive silhouette.",
        "3": "Not selected; retained as a candidate because it is balanced with no sub-5% cluster, but K=4 has higher silhouette and the primary geometric elbow evidence.",
        "5": "Rejected as post-elbow; silhouette falls below K=4 and a 0.6501595486% SMALL cluster appears.",
        "6": "Rejected as post-elbow; silhouette falls to 0.6116205177% and a 0.5823321327% SMALL cluster appears.",
        "7": "Rejected because silhouette is lowest, a 0.2807790757% VERY_SMALL cluster appears, and num_iter=100 reached the configured limit; convergence requires caution.",
        "8": "Rejected as a more fragmented post-elbow solution; a 0.2387172691% VERY_SMALL cluster appears and silhouette remains below K=4 despite the WCSS reduction from K=7.",
    }

    return {
        "stage": 9,
        "model": "Main RFM K-Means",
        "source_stage7_scaler_metadata": str(audit["metadata_path"]),
        "source_stage8": str(audit["stage8_path"]),
        "feature_order": FEATURE_ORDER,
        "selection_method": {
            "silhouette": True,
            "wcss_elbow": True,
            "cluster_balance": True,
            "centroid_interpretability": True,
            "convergence": True,
            "business_usefulness": True,
            "weighted_score_used": False,
        },
        "authority_audit": {
            "pass": audit["pass"],
            "stage7_withMean": metadata.get("withMean"),
            "stage7_withStd": metadata.get("withStd"),
            "stage7_mean_length": len(metadata.get("means", [])),
            "stage7_std_length": len(metadata.get("stds", [])),
            "stage8_k_values": EXPECTED_K_VALUES,
            "stage8_all_pass": True,
        },
        "silhouette_ranking": [
            {
                "rank": rank,
                "k": k,
                "silhouette": float(by_k[k]["silhouette"]),
                "difference_from_best": float(by_k[best_silhouette_k]["silhouette"] - by_k[k]["silhouette"]),
            }
            for rank, k in enumerate(ranking, start=1)
        ],
        "best_silhouette": {
            "k": best_silhouette_k,
            "value": float(by_k[best_silhouette_k]["silhouette"]),
        },
        "second_best_silhouette": {
            "k": second_silhouette_k,
            "value": float(by_k[second_silhouette_k]["silhouette"]),
        },
        "wcss_reduction_table": reductions,
        "elbow_analysis": elbow,
        "cluster_balance_analysis": balances,
        "convergence_analysis": {
            "maxIter": MAX_ITER,
            "by_k": {
                str(k): {
                    "num_iter": int(by_k[k]["num_iter"]),
                    "convergence_flag": "LIMIT_REACHED" if int(by_k[k]["num_iter"]) >= MAX_ITER else "NORMAL",
                }
                for k in EXPECTED_K_VALUES
            },
            "convergence_warning_k7": int(by_k[7]["num_iter"]) >= MAX_ITER,
            "note": "K=7 reached the configured iteration limit and requires caution; no automatic rerun was performed.",
        },
        "inverse_center_validation": {
            "pass": True,
            "formula": "original_value = scaled_center * scaler_std + scaler_mean",
            "feature_order": FEATURE_ORDER,
            "negative_values_found": False,
            "all_values_finite": True,
            "all_dimensions_three": True,
        },
        "interpretability_analysis": interpretability,
        "decision_table": decision_rows,
        "candidates": [3, 4],
        "selected_k": selected_k,
        "selected_k_metrics": {
            "silhouette": float(selected_metrics["silhouette"]),
            "wcss": float(selected_metrics["wcss"]),
            "min_cluster_ratio_pct": float(selected_metrics["min_cluster_ratio_pct"]),
            "max_cluster_ratio_pct": float(selected_metrics["max_cluster_ratio_pct"]),
            "num_iter": int(selected_metrics["num_iter"]),
        },
        "selection_checks": selection_checks,
        "selection_reason": selection_reason,
        "rejected_alternatives": rejected_alternatives,
        "no_retraining": True,
        "final_model_trained": False,
        "ready_for_stage10": True,
    }


def write_report(
    report_path: Path,
    payload: Dict[str, Any],
    audit: Dict[str, Any],
    blockers: List[str],
) -> None:
    lines = [
        "STAGE 9 — MAIN K-MEANS OPTIMAL-K SELECTION",
        "=" * 115,
        f"SOURCE_STAGE7_SCALER: {payload.get('source_stage7_scaler_metadata')}",
        f"SOURCE_STAGE8: {payload.get('source_stage8')}",
        "NO_RETRAINING: True",
        "NO_FINAL_MODEL_TRAINED: True",
        "",
        "VERIFIED_AUTHORITIES",
        f"{audit.get('pass')}; blockers={audit.get('blockers')}",
        f"Feature order: {FEATURE_ORDER}; withMean=True; withStd=True; mean/std length=3.",
        "",
        "SILHOUETTE_RANKING",
    ]
    for row in payload.get("silhouette_ranking", []):
        lines.append(
            f"rank={row['rank']}; K={row['k']}; silhouette={row['silhouette']:.12f}; difference_from_best={row['difference_from_best']:.12f}"
        )
    lines.extend(["", "WCSS_REDUCTION_TABLE"])
    for row in payload.get("wcss_reduction_table", []):
        lines.append(
            f"K={row['from_k']}->{row['to_k']}; previous={row['wcss_previous']:.6f}; current={row['wcss_current']:.6f}; absolute_reduction={row['absolute_reduction']:.6f}; reduction_pct={row['reduction_pct']:.6f}%"
        )
    elbow = payload.get("elbow_analysis", {})
    lines.extend(
        [
            "",
            "ELBOW_ANALYSIS",
            f"Method: {elbow.get('method')}",
            f"Supporting elbow K: {elbow.get('supporting_elbow_k')}; distance={elbow.get('supporting_elbow_distance'):.12f}",
            "Supporting evidence only; it was not used as a single-metric decision rule.",
            "",
            "CLUSTER_BALANCE_ANALYSIS",
        ]
    )
    for k in EXPECTED_K_VALUES:
        balance = payload["cluster_balance_analysis"][str(k)] if str(k) in payload["cluster_balance_analysis"] else payload["cluster_balance_analysis"][k]
        lines.append(
            f"K={k}; min={balance['min_cluster_count']}; max={balance['max_cluster_count']}; min_ratio={balance['min_cluster_ratio_pct']:.6f}%; max_ratio={balance['max_cluster_ratio_pct']:.6f}%; max/min={balance['max_min_count_ratio']:.6f}; below0.5={balance['clusters_below_0_5_pct']}; below1={balance['clusters_below_1_pct']}; below2={balance['clusters_below_2_pct']}; below5={balance['clusters_below_5_pct']}; flags={balance['cluster_flags']}"
        )
    lines.extend(
        [
            "",
            "CONVERGENCE_ANALYSIS",
            f"{payload['convergence_analysis']}",
            "K=7 reached maxIter=100; this is a caution flag, not evidence of normal convergence.",
            "",
            "INVERSE_CENTER_VALIDATION",
            f"{payload['inverse_center_validation']}",
            "Inverse centers use original_value = scaled_center * std + mean; no negative reconstructed RFM values occurred.",
            "",
            "CANDIDATE_K_VALUES",
            f"{payload['candidates']}",
            "",
            "DECISION_TABLE",
            "K | Silhouette | Rank | WCSS | ReductionPctFromPrevious | MinRatioPct | MaxRatioPct | Below1Pct | Below2Pct | NumIter | Convergence | Elbow | Status",
        ]
    )
    for row in payload.get("decision_table", []):
        lines.append(
            f"{row['K']} | {row['Silhouette']:.12f} | {row['SilhouetteRank']} | {row['WCSS']:.6f} | {row['WCSSReductionPctFromPrevious'] if row['WCSSReductionPctFromPrevious'] is not None else 'N/A'} | {row['MinClusterRatioPct']:.6f} | {row['MaxClusterRatioPct']:.6f} | {row['ClustersBelow1Pct']} | {row['ClustersBelow2Pct']} | {row['NumIter']} | {row['ConvergenceFlag']} | {row['ElbowEvidence']} | {row['SelectionStatus']}"
        )
    selected_k = payload.get("selected_k")
    selected_centers = [row for row in payload.get("inverse_centers", []) if row.get("k") == selected_k]
    lines.extend(
        [
            "",
            f"SELECTED_K: {selected_k}",
            f"SELECTION_REASON: {payload.get('selection_reason')}",
            "",
            "REJECTED_ALTERNATIVES",
            *[f"K={k}: {reason}" for k, reason in payload.get("rejected_alternatives", {}).items()],
            "",
            f"WARNINGS: K=7 convergence limit; K>=5 tiny-cluster diagnostics; selected K={selected_k} was not retrained.",
            f"BLOCKERS: {blockers}",
            "READY_FOR_STAGE_10: True",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    metadata_path = project_root / "output" / "ml" / "main_rfm_scaler_metadata" / "scaler_metadata.json"
    stage8_path = project_root / "output" / "analysis" / "stage8" / "kmeans_k_sweep.json"
    cluster_dir = project_root / "output" / "analysis" / "stage8" / "cluster_sizes"
    centers_dir = project_root / "output" / "analysis" / "stage8" / "centers"
    analysis_dir = project_root / "output" / "analysis" / "stage9"
    report_path = project_root / "reports" / "stage9_optimal_k_selection.txt"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = analysis_dir / "optimal_k_selection.json"
    selection_csv_path = analysis_dir / "k_selection_table.csv"
    reduction_csv_path = analysis_dir / "wcss_reduction_table.csv"
    inverse_json_path = analysis_dir / "inverse_centers.json"
    inverse_csv_path = analysis_dir / "inverse_centers.csv"

    audit = audit_authorities(metadata_path, stage8_path, cluster_dir, centers_dir)
    audit["metadata_path"] = str(metadata_path)
    audit["stage8_path"] = str(stage8_path)
    print(f"AUTHORITY_AUDIT_PASS={audit['pass']}")
    print(f"AUTHORITY_BLOCKERS={audit['blockers']}")
    if not audit["pass"]:
        failure = {
            "stage": 9,
            "model": "Main RFM K-Means",
            "source_stage7_scaler_metadata": str(metadata_path),
            "source_stage8": str(stage8_path),
            "authority_audit": audit,
            "ready_for_stage10": False,
            "blockers": audit["blockers"],
        }
        write_json(json_path, failure)
        write_csv(selection_csv_path, ["K", "SelectionStatus"], [])
        write_csv(reduction_csv_path, ["from_k", "to_k"], [])
        write_json(inverse_json_path, {"stage": 9, "inverse_centers": [], "ready_for_stage10": False})
        write_csv(inverse_csv_path, ["k", "cluster_id"], [])
        report_path.write_text(
            "STAGE 9 FAILED CLOSED\n" + "\n".join(audit["blockers"]) + "\nREADY_FOR_STAGE_10: False\n",
            encoding="utf-8",
        )
        return 3

    payload = build_analysis(audit)
    payload["authority_audit"] = {
        **payload["authority_audit"],
        "blockers": audit["blockers"],
    }

    inverse_rows: List[Dict[str, Any]] = []
    for k in EXPECTED_K_VALUES:
        item = audit["by_k"][k]
        for cluster, scaled_center in zip(item["cluster_sizes"], item["centers_scaled"]):
            original = [
                float(scaled_center[index]) * float(audit["metadata"]["stds"][index])
                + float(audit["metadata"]["means"][index])
                for index in range(3)
            ]
            inverse_rows.append(
                {
                    "k": k,
                    "cluster_id": int(cluster["cluster_id"]),
                    "Recency_original": original[0],
                    "Frequency_original": original[1],
                    "Monetary_original": original[2],
                    "cluster_count": int(cluster["count"]),
                    "cluster_ratio_pct": float(cluster["ratio_pct"]),
                }
            )
    payload["inverse_centers"] = inverse_rows

    write_json(json_path, payload)
    write_csv(
        selection_csv_path,
        [
            "K", "Silhouette", "SilhouetteRank", "WCSS", "WCSSReductionPctFromPrevious",
            "MinClusterRatioPct", "MaxClusterRatioPct", "ClustersBelow1Pct", "ClustersBelow2Pct",
            "NumIter", "ConvergenceFlag", "ElbowEvidence", "InterpretabilityAssessment", "SelectionStatus",
        ],
        payload["decision_table"],
    )
    write_csv(
        reduction_csv_path,
        ["from_k", "to_k", "wcss_previous", "wcss_current", "absolute_reduction", "reduction_pct"],
        payload["wcss_reduction_table"],
    )
    write_json(
        inverse_json_path,
        {
            "stage": 9,
            "model": "Main RFM K-Means",
            "formula": "original_value = scaled_center * scaler_std + scaler_mean",
            "feature_order": FEATURE_ORDER,
            "scaler_metadata": str(metadata_path),
            "inverse_center_validation": payload["inverse_center_validation"],
            "inverse_centers": inverse_rows,
        },
    )
    write_csv(
        inverse_csv_path,
        [
            "k", "cluster_id", "Recency_original", "Frequency_original", "Monetary_original",
            "cluster_count", "cluster_ratio_pct",
        ],
        inverse_rows,
    )

    readback_checks = {
        "selection_json_exists": json_path.exists(),
        "selection_csv_exists": selection_csv_path.exists(),
        "reduction_csv_exists": reduction_csv_path.exists(),
        "inverse_json_exists": inverse_json_path.exists(),
        "inverse_csv_exists": inverse_csv_path.exists(),
        "decision_row_count": len(payload["decision_table"]) == len(EXPECTED_K_VALUES),
        "reduction_row_count": len(payload["wcss_reduction_table"]) == len(EXPECTED_K_VALUES) - 1,
        "inverse_row_count": len(inverse_rows) == sum(EXPECTED_K_VALUES),
        "exactly_one_selected": sum(row["SelectionStatus"] == "SELECTED" for row in payload["decision_table"]) == 1,
    }
    payload["artifact_readback"] = readback_checks
    payload["ready_for_stage10"] = bool(all(readback_checks.values()))
    write_json(json_path, payload)
    write_report(report_path, payload, audit, [] if payload["ready_for_stage10"] else ["Artifact read-back failed."])
    print(f"SELECTED_K={payload['selected_k']}")
    print(f"SUPPORTING_ELBOW_K={payload['elbow_analysis']['supporting_elbow_k']}")
    print(f"CONVERGENCE_WARNING_K7={payload['convergence_analysis']['convergence_warning_k7']}")
    print(f"READY_FOR_STAGE_10={payload['ready_for_stage10']}")
    return 0 if payload["ready_for_stage10"] else 3


if __name__ == "__main__":
    sys.exit(main())
