"""Stage 7 - Main RFM ML preprocessing pipeline.

The main model feature path is intentionally limited to:

    RFM -> VectorAssembler -> StandardScaler

No clustering model is trained in this stage.  The verified RFM Parquet
artifact is never overwritten.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    avg,
    col,
    count,
    countDistinct,
    element_at,
    isnan,
    max as spark_max,
    min as spark_min,
    size,
    stddev,
    sum as spark_sum,
    when,
)
from pyspark.storagelevel import StorageLevel


FEATURES = ["Recency", "Frequency", "Monetary"]
FEATURE_ORDER = {"0": "Recency", "1": "Frequency", "2": "Monetary"}
EXPECTED_ROW_COUNT = 1_362_281
VECTOR_DIMENSION = 3
STANDARD_TOLERANCE = 1e-6
POSITIVE_INFINITY = float("inf")
NEGATIVE_INFINITY = float("-inf")


def finite_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def json_safe_list(values: Any) -> List[Optional[float]]:
    return [finite_or_none(value) for value in values]


def invalid_numeric_expression(feature: str):
    value = col(feature).cast("double")
    return (
        value.isNull()
        | isnan(value)
        | (value == POSITIVE_INFINITY)
        | (value == NEGATIVE_INFINITY)
    )


def enable_windows_java_filesystem_fallback(spark: SparkSession) -> None:
    """Use Java File.canRead when Hadoop NativeIO is unavailable on Windows."""

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
    expressions = [
        count("*").alias("row_count"),
        countDistinct("customer_id").alias("distinct_customers"),
        spark_sum(when(col("customer_id").isNull(), 1).otherwise(0)).alias(
            "null_customer_id"
        ),
    ]
    for feature in FEATURES:
        invalid = invalid_numeric_expression(feature)
        expressions.extend(
            [
                spark_sum(when(invalid, 1).otherwise(0)).alias(
                    f"invalid_{feature}"
                ),
                spark_sum(
                    when((~invalid) & (col(feature) <= 0), 1).otherwise(0)
                ).alias(f"non_positive_{feature}"),
            ]
        )

    row = df.agg(*expressions).first().asDict()
    row["duplicate_customer_id"] = int(
        row["row_count"] - row["distinct_customers"]
    )
    row["invalid_total"] = int(
        sum(int(row[f"invalid_{feature}"] or 0) for feature in FEATURES)
    )
    row["non_positive_total"] = int(
        sum(int(row[f"non_positive_{feature}"] or 0) for feature in FEATURES)
    )
    row["pass"] = bool(
        row["row_count"] == EXPECTED_ROW_COUNT
        and row["distinct_customers"] == EXPECTED_ROW_COUNT
        and row["duplicate_customer_id"] == 0
        and row["null_customer_id"] == 0
        and row["invalid_total"] == 0
        and row["non_positive_total"] == 0
    )
    return row


def vector_quality(df: DataFrame, vector_col: str) -> Dict[str, Any]:
    values = vector_to_array(col(vector_col))
    expressions = [
        count("*").alias("row_count"),
        spark_sum(when(col(vector_col).isNull(), 1).otherwise(0)).alias(
            "null_vector"
        ),
        spark_sum(
            when(col(vector_col).isNotNull() & (size(values) != VECTOR_DIMENSION), 1)
            .otherwise(0)
        ).alias("wrong_dimension"),
    ]
    for index in range(1, VECTOR_DIMENSION + 1):
        value = element_at(values, index)
        expressions.extend(
            [
                spark_sum(when(isnan(value), 1).otherwise(0)).alias(
                    f"nan_{index - 1}"
                ),
                spark_sum(
                    when(
                        (value == POSITIVE_INFINITY) | (value == NEGATIVE_INFINITY),
                        1,
                    ).otherwise(0)
                ).alias(f"infinity_{index - 1}"),
            ]
        )

    row = df.agg(*expressions).first().asDict()
    row["nan_total"] = int(
        sum(int(row[f"nan_{index}"] or 0) for index in range(VECTOR_DIMENSION))
    )
    row["infinity_total"] = int(
        sum(
            int(row[f"infinity_{index}"] or 0)
            for index in range(VECTOR_DIMENSION)
        )
    )
    row["pass"] = bool(
        row["null_vector"] == 0
        and row["wrong_dimension"] == 0
        and row["nan_total"] == 0
        and row["infinity_total"] == 0
    )
    return row


def vector_statistics(df: DataFrame, vector_col: str) -> Dict[str, List[Optional[float]]]:
    values = vector_to_array(col(vector_col))
    means = df.agg(
        *[
            avg(element_at(values, index)).alias(f"mean_{index - 1}")
            for index in range(1, VECTOR_DIMENSION + 1)
        ]
    ).first()
    stds = df.agg(
        *[
            stddev(element_at(values, index)).alias(f"std_{index - 1}")
            for index in range(1, VECTOR_DIMENSION + 1)
        ]
    ).first()
    return {
        "mean": json_safe_list(
            [means[f"mean_{index}"] for index in range(VECTOR_DIMENSION)]
        ),
        "std": json_safe_list(
            [stds[f"std_{index}"] for index in range(VECTOR_DIMENSION)]
        ),
    }


def standardization_pass(stats: Dict[str, List[Optional[float]]]) -> bool:
    means_ok = all(
        value is not None and abs(value) <= STANDARD_TOLERANCE
        for value in stats["mean"]
    )
    stds_ok = all(
        value is not None and abs(value - 1.0) <= STANDARD_TOLERANCE
        for value in stats["std"]
    )
    return means_ok and stds_ok


def as_bool(value: Any) -> bool:
    return bool(value)


def write_report(
    report_path: Path,
    project_root: Path,
    input_path: Path,
    output_path: Path,
    rerun: bool,
    input_check: Dict[str, Any],
    scaler_mean: List[Optional[float]],
    scaler_std: List[Optional[float]],
    output_check: Dict[str, Any],
    raw_quality: Dict[str, Any],
    scaled_quality: Dict[str, Any],
    scaled_stats: Dict[str, List[Optional[float]]],
    readback_check: Dict[str, Any],
    ready: bool,
    blocker: str = "None",
) -> None:
    lines = [
        "STAGE 7 — MAIN RFM ML PREPROCESSING PIPELINE",
        "=" * 100,
        f"PROJECT_ROOT: {project_root}",
        f"INPUT_PATH: {input_path}",
        f"OUTPUT_PATH: {output_path}",
        f"RERUN_STAGE_7: {rerun}",
        "",
        "MAIN_MODEL_AUTHORITY",
        "RFM -> VectorAssembler -> StandardScaler -> K-Means (later stages)",
        "Stage 7 does not train a clustering model.",
        "",
        "INPUT_VALIDATION",
        f"INPUT_ROW_COUNT: {input_check.get('row_count')}",
        f"INPUT_DISTINCT_CUSTOMERS: {input_check.get('distinct_customers')}",
        f"DUPLICATE_CUSTOMER_ID: {input_check.get('duplicate_customer_id')}",
        f"NULL_CUSTOMER_ID: {input_check.get('null_customer_id')}",
        f"INVALID_NUMERIC_TOTAL: {input_check.get('invalid_total')}",
        f"NON_POSITIVE_TOTAL: {input_check.get('non_positive_total')}",
        f"INPUT_VALIDATION_PASS: {input_check.get('pass')}",
        "",
        f"FEATURE_ORDER: {FEATURES}",
        "VECTORASSEMBLER_CONFIG: inputCols=['Recency', 'Frequency', 'Monetary']; outputCol='features_raw'; handleInvalid='error'",
        "STANDARDSCALER_CONFIG: inputCol='features_raw'; outputCol='features'; withMean=True; withStd=True",
        f"SCALER_MEAN: {scaler_mean}",
        f"SCALER_STD: {scaler_std}",
        "",
        f"OUTPUT_ROW_COUNT: {output_check.get('row_count')}",
        f"OUTPUT_DISTINCT_CUSTOMERS: {output_check.get('distinct_customers')}",
        f"VECTOR_DIMENSION: {VECTOR_DIMENSION}",
        f"RAW_VECTOR_VALIDATION: {raw_quality}",
        f"SCALED_VECTOR_VALIDATION: {scaled_quality}",
        f"SCALED_FEATURE_MEAN: {scaled_stats.get('mean')}",
        f"SCALED_FEATURE_STD: {scaled_stats.get('std')}",
        f"STANDARDIZATION_TOLERANCE: {STANDARD_TOLERANCE}",
        f"STANDARDIZATION_PASS: {standardization_pass(scaled_stats)}",
        "",
        "READBACK_VALIDATION",
        f"{readback_check}",
        "",
        f"READY_FOR_STAGE_8: {ready}",
        f"BLOCKERS: {blocker}",
        "",
        "SEMANTIC LOCK",
        "Frequency = number of distinct purchase dates per customer; not invoice/order count.",
        "Monetary = SUM(price). Main model Stage 7 intentionally does not apply log1p.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "output" / "features" / "rfm"
    output_path = project_root / "output" / "ml" / "main_rfm_scaled"
    metadata_dir = project_root / "output" / "ml" / "main_rfm_scaler_metadata"
    metadata_path = metadata_dir / "scaler_metadata.json"
    summary_path = (
        project_root / "output" / "analysis" / "stage7" / "main_rfm_pipeline_summary.json"
    )
    report_path = project_root / "reports" / "stage7_main_rfm_ml_pipeline.txt"

    if not input_path.exists():
        print(f"ERROR: missing input path: {input_path}")
        return 2

    rerun = output_path.exists()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    spark_tmp = project_root.parent / "spark_tmp"
    spark_tmp.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .master("local[4]")
        .appName("Stage7-Main-RFM-ML-Pipeline")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.local.dir", str(spark_tmp))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    enable_windows_java_filesystem_fallback(spark)

    input_check: Dict[str, Any] = {}
    scaler_mean: List[Optional[float]] = []
    scaler_std: List[Optional[float]] = []
    output_check: Dict[str, Any] = {}
    raw_quality: Dict[str, Any] = {}
    scaled_quality: Dict[str, Any] = {}
    scaled_stats: Dict[str, List[Optional[float]]] = {"mean": [], "std": []}
    readback_check: Dict[str, Any] = {}
    ready = False
    blocker = "None"

    try:
        print(f"Loading RFM input: {input_path}")
        source = spark.read.parquet(str(input_path))
        required_columns = ["customer_id", *FEATURES]
        missing_columns = [c for c in required_columns if c not in source.columns]
        if missing_columns:
            blocker = f"Missing required columns: {missing_columns}"
            raise ValueError(blocker)

        input_df = source.select(
            col("customer_id"),
            *[col(feature).cast("double").alias(feature) for feature in FEATURES],
        ).persist(StorageLevel.MEMORY_AND_DISK)

        input_check = input_validation(input_df)
        print(f"Input validation: {input_check}")
        if not input_check["pass"]:
            blocker = "Input validation failed; fail-closed before pipeline fit."
            raise ValueError(blocker)

        assembler = VectorAssembler(
            inputCols=FEATURES,
            outputCol="features_raw",
            handleInvalid="error",
        )
        assembled = assembler.transform(input_df).persist(StorageLevel.MEMORY_AND_DISK)

        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withMean=True,
            withStd=True,
        )
        scaler_model = scaler.fit(assembled)
        scaler_mean = json_safe_list(scaler_model.mean.toArray().tolist())
        scaler_std = json_safe_list(scaler_model.std.toArray().tolist())

        transformed = scaler_model.transform(assembled).select(
            "customer_id",
            *FEATURES,
            "features_raw",
            "features",
        ).persist(StorageLevel.MEMORY_AND_DISK)

        raw_quality = vector_quality(transformed, "features_raw")
        scaled_quality = vector_quality(transformed, "features")
        scaled_stats = vector_statistics(transformed, "features")
        output_check = transformed.agg(
            count("*").alias("row_count"),
            countDistinct("customer_id").alias("distinct_customers"),
        ).first().asDict()

        if not (
            raw_quality["pass"]
            and scaled_quality["pass"]
            and output_check["row_count"] == input_check["row_count"]
            and output_check["distinct_customers"] == input_check["distinct_customers"]
            and standardization_pass(scaled_stats)
        ):
            blocker = "Pre-write validation failed; output was not written."
            raise ValueError(blocker)

        print(f"Writing Stage 7 output: {output_path}")
        transformed.write.mode("overwrite").parquet(str(output_path))

        readback = spark.read.parquet(str(output_path)).select(
            "customer_id",
            *FEATURES,
            "features_raw",
            "features",
        )
        readback_check = readback.agg(
            count("*").alias("row_count"),
            countDistinct("customer_id").alias("distinct_customers"),
        ).first().asDict()
        readback_raw_quality = vector_quality(readback, "features_raw")
        readback_scaled_quality = vector_quality(readback, "features")
        readback_pass = bool(
            readback_check["row_count"] == input_check["row_count"]
            and readback_check["distinct_customers"] == input_check["distinct_customers"]
            and readback_raw_quality["pass"]
            and readback_scaled_quality["pass"]
        )
        readback_check["features_raw_quality"] = readback_raw_quality
        readback_check["features_quality"] = readback_scaled_quality
        readback_check["pass"] = readback_pass
        print(f"Read-back validation: {readback_check}")

        if not readback_pass:
            blocker = "Read-back validation failed."
            raise ValueError(blocker)

        metadata = {
            "stage": 7,
            "model": "Main RFM K-Means",
            "input_features": FEATURES,
            "feature_order": FEATURE_ORDER,
            "vector_col": "features_raw",
            "scaled_vector_col": "features",
            "withMean": True,
            "withStd": True,
            "means": scaler_mean,
            "stds": scaler_std,
            "row_count": int(input_check["row_count"]),
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        summary = {
            "stage": 7,
            "name": "Main RFM ML preprocessing pipeline",
            "input": {
                "path": str(input_path),
                "row_count": int(input_check["row_count"]),
                "distinct_customers": int(input_check["distinct_customers"]),
            },
            "features": FEATURES,
            "vector_assembler": {
                "input_cols": FEATURES,
                "output_col": "features_raw",
                "handle_invalid": "error",
            },
            "standard_scaler": {
                "input_col": "features_raw",
                "output_col": "features",
                "withMean": True,
                "withStd": True,
                "mean": scaler_mean,
                "std": scaler_std,
            },
            "output": {
                "path": str(output_path),
                "row_count": int(output_check["row_count"]),
                "distinct_customers": int(output_check["distinct_customers"]),
                "vector_dimension": VECTOR_DIMENSION,
            },
            "validation": {
                "no_row_loss": output_check["row_count"] == input_check["row_count"],
                "no_duplicate_customer": output_check["row_count"]
                == output_check["distinct_customers"],
                "null_free": raw_quality["null_vector"] == 0
                and scaled_quality["null_vector"] == 0,
                "nan_free": raw_quality["nan_total"] == 0
                and scaled_quality["nan_total"] == 0,
                "infinity_free": raw_quality["infinity_total"] == 0
                and scaled_quality["infinity_total"] == 0,
                "standardization_pass": standardization_pass(scaled_stats),
                "readback_pass": readback_pass,
            },
            "scaled_feature_statistics": scaled_stats,
            "ready_for_stage8": True,
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        ready = True
        print("Stage 7 completed successfully.")
        return 0
    except Exception as error:
        if blocker == "None":
            blocker = str(error)
        print(f"STAGE 7 FAILED CLOSED: {error}")
        return 3
    finally:
        write_report(
            report_path,
            project_root,
            input_path,
            output_path,
            rerun,
            input_check,
            scaler_mean,
            scaler_std,
            output_check,
            raw_quality,
            scaled_quality,
            scaled_stats,
            readback_check,
            ready,
            blocker,
        )
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
