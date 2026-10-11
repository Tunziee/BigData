"""Stage 12: build the common improved eRFM feature space.

This stage creates transformed vectors and one shared StandardScaler for later
Enhanced K-Means and GMM work.  It deliberately does not train either model.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    abs as spark_abs,
    avg,
    col,
    count,
    countDistinct,
    element_at,
    isnan,
    log1p,
    max as spark_max,
    min as spark_min,
    skewness,
    size,
    stddev_samp,
    sum as spark_sum,
    when,
)
from pyspark.storagelevel import StorageLevel


EXPECTED_ROW_COUNT = 1_362_281
RAW_FEATURES = ["Recency", "Frequency", "Monetary", "AvgBasketSize", "ProductVariety"]
TRANSFORMED_FEATURES = [
    "Recency_improved",
    "Frequency_log1p",
    "Monetary_log1p",
    "AvgBasketSize_log1p",
    "ProductVariety_log1p",
]
VECTOR_DIMENSION = 5
FEATURE_ORDER = TRANSFORMED_FEATURES
QUANTILE_RELATIVE_ERROR = 0.001
SCALAR_TOLERANCE = 1e-10
STANDARDIZATION_TOLERANCE = 1e-6
SKEWNESS_COMPARISON_TOLERANCE = 0.1
RAW_REQUIRED_COLUMNS = ["customer_id"] + RAW_FEATURES
OUTPUT_REQUIRED_COLUMNS = RAW_REQUIRED_COLUMNS + TRANSFORMED_FEATURES + [
    "features_improved_raw",
    "features_improved",
]


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


def scalar_quality(df: DataFrame, columns: List[str]) -> Dict[str, Any]:
    expressions = []
    for feature in columns:
        expressions.extend(
            [
                spark_sum(when(col(feature).isNull(), 1).otherwise(0)).alias(
                    f"{feature}_null"
                ),
                spark_sum(when(isnan(col(feature)), 1).otherwise(0)).alias(
                    f"{feature}_nan"
                ),
                spark_sum(
                    when(
                        (col(feature) == float("inf"))
                        | (col(feature) == float("-inf")),
                        1,
                    ).otherwise(0)
                ).alias(f"{feature}_infinity"),
            ]
        )
    row = df.agg(*expressions).first().asDict()
    by_feature: Dict[str, Any] = {}
    for feature in columns:
        by_feature[feature] = {
            "null": int(row[f"{feature}_null"] or 0),
            "nan": int(row[f"{feature}_nan"] or 0),
            "infinity": int(row[f"{feature}_infinity"] or 0),
        }
        by_feature[feature]["pass"] = all(
            by_feature[feature][key] == 0 for key in ["null", "nan", "infinity"]
        )
    return {
        "by_feature": by_feature,
        "pass": all(item["pass"] for item in by_feature.values()),
    }


def vector_quality(df: DataFrame, columns: List[str]) -> Dict[str, Any]:
    by_vector: Dict[str, Any] = {}
    for vector_col in columns:
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
                            (value == float("inf")) | (value == float("-inf")),
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
        by_vector[vector_col] = {
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
    return {"by_vector": by_vector, "pass": all(item["pass"] for item in by_vector.values())}


def row_identity_validation(df: DataFrame) -> Dict[str, Any]:
    row = df.agg(
        count("*").alias("row_count"),
        countDistinct("customer_id").alias("distinct_customer_id"),
        spark_sum(when(col("customer_id").isNull(), 1).otherwise(0)).alias(
            "null_customer_id"
        ),
    ).first().asDict()
    row["duplicate_customer_id"] = int(row["row_count"] - row["distinct_customer_id"])
    row["pass"] = bool(
        row["row_count"] == EXPECTED_ROW_COUNT
        and row["distinct_customer_id"] == EXPECTED_ROW_COUNT
        and row["duplicate_customer_id"] == 0
        and row["null_customer_id"] == 0
    )
    return row


def positivity_validation(df: DataFrame) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for feature in RAW_FEATURES:
        row = df.agg(
            spark_sum(when(col(feature) <= 0, 1).otherwise(0)).alias("non_positive"),
            spark_min(col(feature)).alias("min"),
        ).first().asDict()
        result[feature] = {
            "non_positive": int(row["non_positive"] or 0),
            "min": float(row["min"]),
            "pass": int(row["non_positive"] or 0) == 0,
        }
    return {"by_feature": result, "pass": all(item["pass"] for item in result.values())}


def skewness_values(df: DataFrame, columns: List[str]) -> Dict[str, float]:
    row = df.agg(*[skewness(col(feature)).alias(feature) for feature in columns]).first().asDict()
    values = {feature: float(row[feature]) for feature in columns}
    if not all(finite(value) for value in values.values()):
        raise ValueError(f"Non-finite skewness values: {values}")
    return values


def standardization_validation(df: DataFrame, vector_col: str) -> Dict[str, Any]:
    values = vector_to_array(col(vector_col))
    means: List[float] = []
    stds: List[float] = []
    dimensions: List[int] = []
    for index in range(1, VECTOR_DIMENSION + 1):
        value = element_at(values, index)
        row = df.agg(
            avg(value).alias("mean"),
            stddev_samp(value).alias("std"),
        ).first().asDict()
        means.append(float(row["mean"]))
        stds.append(float(row["std"]))
        dimensions.append(VECTOR_DIMENSION)
    pass_value = bool(
        all(abs(value) <= STANDARDIZATION_TOLERANCE for value in means)
        and all(abs(value - 1.0) <= STANDARDIZATION_TOLERANCE for value in stds)
    )
    return {
        "vector_col": vector_col,
        "dimension": VECTOR_DIMENSION,
        "means": means,
        "stds": stds,
        "mean_tolerance": STANDARDIZATION_TOLERANCE,
        "std_tolerance": STANDARDIZATION_TOLERANCE,
        "pass": pass_value,
    }


def transformation_validation(df: DataFrame) -> Dict[str, Any]:
    expressions = [
        spark_max(spark_abs(col("Recency_improved") - col("Recency"))).alias(
            "recency_max_abs_error"
        ),
        spark_max(
            spark_abs(col("Frequency_log1p") - log1p(col("Frequency")))
        ).alias("frequency_max_abs_error"),
        spark_max(
            spark_abs(col("Monetary_log1p") - log1p(col("Monetary")))
        ).alias("monetary_max_abs_error"),
        spark_max(
            spark_abs(col("AvgBasketSize_log1p") - log1p(col("AvgBasketSize")))
        ).alias("avg_basket_size_max_abs_error"),
        spark_max(
            spark_abs(col("ProductVariety_log1p") - log1p(col("ProductVariety")))
        ).alias("product_variety_max_abs_error"),
    ]
    row = df.agg(*expressions).first().asDict()
    errors = {key: float(value) for key, value in row.items()}
    return {
        "max_abs_errors": errors,
        "tolerance": SCALAR_TOLERANCE,
        "pass": all(value <= SCALAR_TOLERANCE for value in errors.values()),
    }


def write_failure(
    summary_path: Path,
    report_path: Path,
    blockers: List[str],
    source_path: Path,
) -> None:
    write_json(
        summary_path,
        {
            "stage": 12,
            "authority": "IMPROVED_ERFM_FEATURE_SPACE",
            "source": str(source_path),
            "ready_for_stage13": False,
            "blockers": blockers,
        },
    )
    report_path.write_text(
        "STAGE 12 FAILED CLOSED\n"
        + "\n".join(blockers)
        + "\nREADY_FOR_STAGE_13: False\n",
        encoding="utf-8",
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    source_path = project_root / "output" / "features" / "erfm"
    stage6_summary_path = project_root / "output" / "analysis" / "erfm_distribution" / "summary.json"
    stage11_summary_path = project_root / "output" / "analysis" / "stage11" / "final_cluster_profile_summary.json"
    output_path = project_root / "output" / "ml" / "improved_erfm_scaled"
    scaler_metadata_dir = project_root / "output" / "ml" / "improved_erfm_scaler_metadata"
    scaler_metadata_path = scaler_metadata_dir / "scaler_metadata.json"
    analysis_dir = project_root / "output" / "analysis" / "stage12"
    summary_path = analysis_dir / "transformation_summary.json"
    skewness_path = analysis_dir / "skewness_comparison.json"
    report_path = project_root / "reports" / "stage12_improved_feature_pipeline.txt"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    scaler_metadata_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    blockers: List[str] = []
    warnings: List[str] = []
    stage6 = {}
    try:
        stage6 = read_json(stage6_summary_path)
    except Exception as error:
        blockers.append(f"Stage 6 evidence unreadable: {error}")
    try:
        stage11 = read_json(stage11_summary_path)
    except Exception as error:
        stage11 = {}
        blockers.append(f"Stage 11 gate unreadable: {error}")

    if not (
        stage11.get("stage") == 11
        and stage11.get("authority") == "MAIN_KMEANS_FINAL_CLUSTER_PROFILE"
        and stage11.get("selected_k") == 4
        and stage11.get("ready_for_improved_model_stage") is True
    ):
        blockers.append("Stage 11 main-model completion gate is not valid.")
    if not source_path.exists():
        blockers.append(f"Missing eRFM source: {source_path}")
    if not stage6 or stage6.get("row_count") != EXPECTED_ROW_COUNT:
        blockers.append("Stage 6 eRFM evidence is missing or has the wrong row count.")
    if blockers:
        write_failure(summary_path, report_path, blockers, source_path)
        return 3

    spark = (
        SparkSession.builder
        .master("local[4]")
        .appName("Stage12-Improved-eRFM-Feature-Pipeline")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.local.dir", str(project_root.parent / "spark_tmp"))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    enable_windows_java_filesystem_fallback(spark)
    source_df: Optional[DataFrame] = None
    try:
        source_df = spark.read.parquet(str(source_path)).select(*RAW_REQUIRED_COLUMNS)
        identity = row_identity_validation(source_df)
        scalar_raw = scalar_quality(source_df, RAW_FEATURES)
        positivity = positivity_validation(source_df)
        input_validation = {
            "identity": identity,
            "scalar_quality": scalar_raw,
            "positivity": positivity,
            "pass": bool(identity["pass"] and scalar_raw["pass"] and positivity["pass"]),
        }
        print(f"INPUT_VALIDATION={input_validation}")
        if not input_validation["pass"]:
            raise ValueError("eRFM input validation failed; fail-closed.")

        source_df = source_df.persist(StorageLevel.MEMORY_AND_DISK)
        source_df.count()
        transformed = (
            source_df
            .withColumn("Recency_improved", col("Recency"))
            .withColumn("Frequency_log1p", log1p(col("Frequency")))
            .withColumn("Monetary_log1p", log1p(col("Monetary")))
            .withColumn("AvgBasketSize_log1p", log1p(col("AvgBasketSize")))
            .withColumn("ProductVariety_log1p", log1p(col("ProductVariety")))
        )
        transformed_scalar = scalar_quality(transformed, TRANSFORMED_FEATURES)
        transformation_check = transformation_validation(transformed)
        if not transformed_scalar["pass"] or not transformation_check["pass"]:
            raise ValueError("Transformed scalar validation failed.")

        raw_skew = skewness_values(source_df, RAW_FEATURES)
        transformed_skew = skewness_values(transformed, TRANSFORMED_FEATURES)
        stage6_by_feature = {
            item["Feature"]: item for item in stage6.get("features", [])
        }
        skewness_comparison: Dict[str, Any] = {}
        for raw_feature, transformed_feature in zip(RAW_FEATURES, TRANSFORMED_FEATURES):
            stage6_item = stage6_by_feature.get(raw_feature, {})
            raw_value = raw_skew[raw_feature]
            transformed_value = transformed_skew[transformed_feature]
            entry = {
                "raw_feature": raw_feature,
                "transformed_feature": transformed_feature,
                "stage6_raw_skewness": stage6_item.get("Skewness"),
                "stage6_log1p_skewness": stage6_item.get("Log1pSkewness"),
                "stage12_raw_skewness": raw_value,
                "stage12_transformed_skewness": transformed_value,
                "raw_difference_vs_stage6": (
                    abs(raw_value - float(stage6_item["Skewness"]))
                    if stage6_item.get("Skewness") is not None
                    else None
                ),
                "transformed_difference_vs_stage6": (
                    abs(transformed_value - float(stage6_item["Log1pSkewness"]))
                    if stage6_item.get("Log1pSkewness") is not None
                    else None
                ),
            }
            entry["comparison_pass"] = bool(
                entry["raw_difference_vs_stage6"] is None
                or entry["raw_difference_vs_stage6"] <= SKEWNESS_COMPARISON_TOLERANCE
            ) and bool(
                entry["transformed_difference_vs_stage6"] is None
                or entry["transformed_difference_vs_stage6"] <= SKEWNESS_COMPARISON_TOLERANCE
            )
            if not entry["comparison_pass"]:
                warnings.append(
                    f"Material skewness difference for {raw_feature}: {entry}"
                )
            skewness_comparison[raw_feature] = entry

        assembler = VectorAssembler(
            inputCols=FEATURE_ORDER,
            outputCol="features_improved_raw",
        )
        raw_vector_df = assembler.transform(transformed)
        raw_vector_quality = vector_quality(raw_vector_df, ["features_improved_raw"])
        if not raw_vector_quality["pass"]:
            raise ValueError("features_improved_raw validation failed.")

        scaler = StandardScaler(
            inputCol="features_improved_raw",
            outputCol="features_improved",
            withMean=True,
            withStd=True,
        )
        scaler_fit_start = time.perf_counter()
        scaler_model = scaler.fit(raw_vector_df)
        scaler_fit_runtime = time.perf_counter() - scaler_fit_start
        scaled_df = scaler_model.transform(raw_vector_df).select(*OUTPUT_REQUIRED_COLUMNS)
        standardization = standardization_validation(scaled_df, "features_improved")
        scaled_vector_quality = vector_quality(
            scaled_df, ["features_improved_raw", "features_improved"]
        )
        if not standardization["pass"] or not scaled_vector_quality["pass"]:
            raise ValueError(
                f"Standardization validation failed: {standardization}; {scaled_vector_quality}"
            )

        print("START IMPROVED FEATURE WRITE")
        scaled_df.write.mode("overwrite").parquet(str(output_path))
        readback = spark.read.parquet(str(output_path)).select(*OUTPUT_REQUIRED_COLUMNS)
        readback_identity = row_identity_validation(readback)
        readback_scalar = scalar_quality(readback, TRANSFORMED_FEATURES)
        readback_vectors = vector_quality(
            readback, ["features_improved_raw", "features_improved"]
        )
        readback_standardization = standardization_validation(readback, "features_improved")
        readback_validation = {
            "schema_columns": readback.columns,
            "identity": readback_identity,
            "transformed_scalar_quality": readback_scalar,
            "vector_quality": readback_vectors,
            "standardization": readback_standardization,
            "pass": bool(
                readback_identity["pass"]
                and readback_scalar["pass"]
                and readback_vectors["pass"]
                and readback_standardization["pass"]
                and readback.columns == OUTPUT_REQUIRED_COLUMNS
            ),
        }
        if not readback_validation["pass"]:
            raise ValueError(f"Improved feature read-back validation failed: {readback_validation}")

        scaler_mean = [float(value) for value in scaler_model.mean.toArray().tolist()]
        scaler_std = [float(value) for value in scaler_model.std.toArray().tolist()]
        transformation_mapping = {
            "Recency": "identity -> Recency_improved",
            "Frequency": "log1p -> Frequency_log1p",
            "Monetary": "log1p -> Monetary_log1p",
            "AvgBasketSize": "log1p -> AvgBasketSize_log1p",
            "ProductVariety": "log1p -> ProductVariety_log1p",
        }
        semantic_notes = {
            "Frequency": "Number of distinct purchase dates per customer, not invoice/order count.",
            "AvgBasketSize": "TotalItems / Frequency; proxy for average item count per distinct purchase date.",
            "ProductVariety": "Number of distinct article_id values purchased by customer.",
            "Recency": "Days from last purchase date to analysis date; retained raw by Stage 6 authority.",
            "Monetary": "SUM(price).",
        }
        validation = {
            "input": input_validation,
            "transformation": transformation_check,
            "raw_vector": raw_vector_quality,
            "scaled_vector": scaled_vector_quality,
            "standardization": standardization,
            "readback": readback_validation,
            "pass": bool(
                input_validation["pass"]
                and transformation_check["pass"]
                and raw_vector_quality["pass"]
                and scaled_vector_quality["pass"]
                and standardization["pass"]
                and readback_validation["pass"]
            ),
        }
        if not validation["pass"]:
            raise ValueError("Stage 12 validation failed.")

        scaler_metadata = {
            "stage": 12,
            "authority": "IMPROVED_ERFM_FEATURE_SPACE",
            "row_count": EXPECTED_ROW_COUNT,
            "source": str(source_path),
            "raw_features": RAW_FEATURES,
            "transformed_features": TRANSFORMED_FEATURES,
            "feature_order": {
                str(index): feature for index, feature in enumerate(FEATURE_ORDER)
            },
            "transformation_mapping": transformation_mapping,
            "withMean": True,
            "withStd": True,
            "means": scaler_mean,
            "stds": scaler_std,
            "raw_vector_col": "features_improved_raw",
            "scaled_vector_col": "features_improved",
            "raw_vector_dimension": VECTOR_DIMENSION,
            "scaled_vector_dimension": VECTOR_DIMENSION,
            "semantic_notes": semantic_notes,
            "scaler_fit_runtime_sec": scaler_fit_runtime,
        }
        transformation_summary = {
            "stage": 12,
            "authority": "IMPROVED_ERFM_FEATURE_SPACE",
            "source": str(source_path),
            "row_count": EXPECTED_ROW_COUNT,
            "raw_features": RAW_FEATURES,
            "transformations": {
                "Recency": "identity",
                "Frequency": "log1p",
                "Monetary": "log1p",
                "AvgBasketSize": "log1p",
                "ProductVariety": "log1p",
            },
            "feature_order": FEATURE_ORDER,
            "semantic_notes": semantic_notes,
            "raw_vs_transformed_skewness": skewness_comparison,
            "scaler": {
                "inputCol": "features_improved_raw",
                "outputCol": "features_improved",
                "withMean": True,
                "withStd": True,
                "means": scaler_mean,
                "stds": scaler_std,
                "dimension": VECTOR_DIMENSION,
            },
            "validation": validation,
            "source_stage6_evidence": str(stage6_summary_path),
            "source_stage11_gate": str(stage11_summary_path),
            "scaler_fit_runtime_sec": scaler_fit_runtime,
            "ready_for_stage13": True,
        }
        write_json(scaler_metadata_path, scaler_metadata)
        write_json(summary_path, transformation_summary)
        write_json(
            skewness_path,
            {
                "stage": 12,
                "authority": "IMPROVED_ERFM_FEATURE_SPACE",
                "comparison_tolerance": SKEWNESS_COMPARISON_TOLERANCE,
                "raw_vs_transformed_skewness": skewness_comparison,
            },
        )
        report_lines = [
            "STAGE 12 — IMPROVED MODEL FEATURE PIPELINE",
            "=" * 115,
            "CURRENT_STAGE: Stage 12",
            "VERIFIED_AUTHORITIES",
            f"Stage 6 evidence: {stage6_summary_path}",
            f"Stage 11 gate: {stage11_summary_path}; main_model_branch_complete=True",
            f"IMPROVED_FEATURE_AUTHORITY: IMPROVED_ERFM_FEATURE_SPACE",
            f"INPUT_VALIDATION: {input_validation}",
            f"TRANSFORMATION_MAPPING: {transformation_mapping}",
            f"RAW_VS_TRANSFORMED_SKEWNESS: {skewness_comparison}",
            f"VECTORASSEMBLER_CONFIG: inputCols={FEATURE_ORDER}; outputCol=features_improved_raw",
            "STANDARDSCALER_CONFIG: inputCol=features_improved_raw; outputCol=features_improved; withMean=True; withStd=True",
            f"SCALER_MEAN: {scaler_mean}",
            f"SCALER_STD: {scaler_std}",
            f"OUTPUT_PATH: {output_path}",
            f"OUTPUT_ROW_COUNT: {EXPECTED_ROW_COUNT}",
            f"VECTOR_DIMENSION: {VECTOR_DIMENSION}",
            f"STANDARDIZATION_VALIDATION: {standardization}",
            f"READBACK_VALIDATION: {readback_validation}",
            "NO_MODEL_TRAINING: K-Means and GMM were not imported or fitted.",
            "ARTIFACTS_CREATED:",
            f"- {report_path}",
            f"- {scaler_metadata_path}",
            f"- {summary_path}",
            f"- {skewness_path}",
            "WARNINGS: Quantiles/skewness use the source Stage 6 methodology; AvgBasketSize is TotalItems/Frequency where Frequency is distinct purchase dates.",
            f"BLOCKERS: {blockers}",
            "READY_FOR_STAGE_13: True",
        ]
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        print(f"OUTPUT_ROW_COUNT={EXPECTED_ROW_COUNT}")
        print(f"VECTOR_DIMENSION={VECTOR_DIMENSION}")
        print("READY_FOR_STAGE_13=True")
        return 0
    except Exception as error:
        blockers.append(str(error))
        print(f"STAGE 12 FAILED CLOSED: {error}")
        write_failure(summary_path, report_path, blockers, source_path)
        return 3
    finally:
        if source_df is not None:
            source_df.unpersist()
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
