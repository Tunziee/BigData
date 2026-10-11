"""Stage 6 - RFM/eRFM distribution and outlier analysis.

This stage reads the verified RFM/eRFM Parquet artifacts and produces only
analysis artifacts.  It does not alter the source feature datasets and does
not apply capping, filtering, or transformations.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    avg,
    col,
    count,
    isnan,
    log1p,
    max as spark_max,
    min as spark_min,
    skewness,
    sum as spark_sum,
    stddev,
    when,
)
from pyspark.storagelevel import StorageLevel


FEATURES = {
    "rfm": ["Recency", "Frequency", "Monetary"],
    "erfm": [
        "Recency",
        "Frequency",
        "Monetary",
        "AvgBasketSize",
        "ProductVariety",
    ],
}

LOG_FEATURES = {
    "Frequency",
    "Monetary",
    "AvgBasketSize",
    "ProductVariety",
}

QUANTILE_PROBABILITIES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
QUANTILE_LABELS = ["P01", "P05", "P25", "P50", "P75", "P95", "P99"]
QUANTILE_RELATIVE_ERROR = 0.001


def finite_or_none(value: Any) -> Optional[float]:
    """Convert Spark numeric output to JSON-safe Python numbers."""

    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def integer_or_zero(value: Any) -> int:
    return int(value or 0)


def valid_numeric_expression(feature: str):
    """Return a numeric expression with NULL/NaN removed."""

    return when(
        col(feature).isNotNull() & ~isnan(col(feature)),
        col(feature),
    )


def count_condition(df: DataFrame, condition) -> int:
    return integer_or_zero(
        df.select(when(condition, 1).otherwise(0).alias("flag"))
        .agg({"flag": "sum"})
        .first()[0]
    )


def add_ratio(record: Dict[str, Any], count_key: str, total_count: int, ratio_key: str) -> None:
    record[ratio_key] = (
        (float(record[count_key]) / float(total_count) * 100.0)
        if total_count
        else None
    )


def feature_statistics(df: DataFrame, feature: str, dataset_count: int) -> Dict[str, Any]:
    """Compute distribution, quality, and outlier statistics for one feature."""

    valid = col(feature).isNotNull() & ~isnan(col(feature))
    clean = valid_numeric_expression(feature)

    aggregate_row = (
        df.agg(
            {feature: "count"}
        )
        .first()
    )
    valid_count = integer_or_zero(aggregate_row[0])

    quality_row = df.agg(
        spark_sum(when(col(feature).isNull(), 1).otherwise(0)).alias("null_count"),
        spark_sum(when(isnan(col(feature)), 1).otherwise(0)).alias("nan_count"),
        spark_sum(
            when(valid & (col(feature) <= 0), 1).otherwise(0)
        ).alias("non_positive_count"),
    ).first()

    clean_expr = clean.cast("double")
    summary_row = df.agg(
        count(clean_expr).alias("valid_count"),
        avg(clean_expr).alias("mean"),
        stddev(clean_expr).alias("stddev"),
        spark_min(clean_expr).alias("min"),
        spark_max(clean_expr).alias("max"),
        skewness(clean_expr).alias("skewness"),
    ).first()

    quantiles = df.select(feature).where(valid).approxQuantile(
        feature,
        QUANTILE_PROBABILITIES,
        QUANTILE_RELATIVE_ERROR,
    )
    quantile_values = {
        label: finite_or_none(value)
        for label, value in zip(QUANTILE_LABELS, quantiles)
    }

    q1 = quantile_values["P25"]
    q3 = quantile_values["P75"]
    iqr = (q3 - q1) if q1 is not None and q3 is not None else None
    iqr_lower = (q1 - 1.5 * iqr) if iqr is not None else None
    iqr_upper = (q3 + 1.5 * iqr) if iqr is not None else None

    iqr_outlier_count = (
        count_condition(
            df,
            valid & (
                (col(feature) < iqr_lower)
                | (col(feature) > iqr_upper)
            ),
        )
        if iqr_lower is not None and iqr_upper is not None
        else 0
    )

    p01 = quantile_values["P01"]
    p99 = quantile_values["P99"]
    p05 = quantile_values["P05"]
    p95 = quantile_values["P95"]

    p01_p99_low = (
        count_condition(df, valid & (col(feature) < p01))
        if p01 is not None
        else 0
    )
    p01_p99_high = (
        count_condition(df, valid & (col(feature) > p99))
        if p99 is not None
        else 0
    )
    p05_p95_low = (
        count_condition(df, valid & (col(feature) < p05))
        if p05 is not None
        else 0
    )
    p05_p95_high = (
        count_condition(df, valid & (col(feature) > p95))
        if p95 is not None
        else 0
    )

    record: Dict[str, Any] = {
        "Dataset": None,
        "Feature": feature,
        "Count": valid_count,
        "DatasetRowCount": dataset_count,
        "NullCount": integer_or_zero(quality_row["null_count"]),
        "NaNCount": integer_or_zero(quality_row["nan_count"]),
        "NonPositiveCount": integer_or_zero(quality_row["non_positive_count"]),
        "Mean": finite_or_none(summary_row["mean"]),
        "Std": finite_or_none(summary_row["stddev"]),
        "Min": finite_or_none(summary_row["min"]),
        "P01": quantile_values["P01"],
        "P05": quantile_values["P05"],
        "P25": quantile_values["P25"],
        "P50": quantile_values["P50"],
        "P75": quantile_values["P75"],
        "P95": quantile_values["P95"],
        "P99": quantile_values["P99"],
        "Max": finite_or_none(summary_row["max"]),
        "Skewness": finite_or_none(summary_row["skewness"]),
        "IQR": finite_or_none(iqr),
        "IQRLower": finite_or_none(iqr_lower),
        "IQRUpper": finite_or_none(iqr_upper),
        "IQROutlierCount": iqr_outlier_count,
        "P01P99LowCount": p01_p99_low,
        "P01P99HighCount": p01_p99_high,
        "P01P99OutlierCount": p01_p99_low + p01_p99_high,
        "P05P95LowCount": p05_p95_low,
        "P05P95HighCount": p05_p95_high,
        "P05P95OutlierCount": p05_p95_low + p05_p95_high,
    }

    add_ratio(record, "NullCount", dataset_count, "NullRatioPct")
    add_ratio(record, "NaNCount", dataset_count, "NaNRatioPct")
    add_ratio(record, "NonPositiveCount", dataset_count, "NonPositiveRatioPct")
    add_ratio(record, "IQROutlierCount", valid_count, "IQROutlierRatioPct")
    add_ratio(record, "P01P99OutlierCount", valid_count, "P01P99OutlierRatioPct")
    add_ratio(record, "P05P95OutlierCount", valid_count, "P05P95OutlierRatioPct")
    add_ratio(record, "P01P99LowCount", valid_count, "P01P99LowRatioPct")
    add_ratio(record, "P01P99HighCount", valid_count, "P01P99HighRatioPct")
    add_ratio(record, "P05P95LowCount", valid_count, "P05P95LowRatioPct")
    add_ratio(record, "P05P95HighCount", valid_count, "P05P95HighRatioPct")

    if feature in LOG_FEATURES:
        log_expr = when(valid & (col(feature) >= 0), log1p(col(feature))).cast("double")
        log_skew = df.agg(skewness(log_expr).alias("log_skewness")).first()["log_skewness"]
        record["Log1pSkewness"] = finite_or_none(log_skew)
        raw_skew = record["Skewness"]
        log_value = record["Log1pSkewness"]
        if raw_skew is not None and log_value is not None:
            record["SkewnessReductionPct"] = (
                (abs(raw_skew) - abs(log_value)) / abs(raw_skew) * 100.0
                if raw_skew != 0
                else 0.0
            )
        else:
            record["SkewnessReductionPct"] = None
    else:
        record["Log1pSkewness"] = None
        record["SkewnessReductionPct"] = None

    return record


def recommendation(record: Dict[str, Any]) -> str:
    """Return a data-driven recommendation without applying any change."""

    feature = record["Feature"]
    raw_skew = record["Skewness"]
    log_skew = record["Log1pSkewness"]
    reduction = record["SkewnessReductionPct"]
    iqr_ratio = record["IQROutlierRatioPct"] or 0.0
    p99_ratio = record["P01P99OutlierRatioPct"] or 0.0

    messages: List[str] = []
    if feature in LOG_FEATURES and raw_skew is not None and log_skew is not None:
        if reduction is not None and reduction >= 20.0:
            messages.append("evaluate log1p before StandardScaler/K-Means")
        else:
            messages.append("retain raw scale for now; log1p does not materially reduce skewness")
    elif feature == "Recency":
        messages.append("do not log by default; retain raw Recency unless Stage 7 justifies otherwise")

    if iqr_ratio >= 5.0 or p99_ratio >= 1.0:
        messages.append("evaluate percentile capping or robust treatment; do not remove rows automatically")
    else:
        messages.append("no broad capping decision indicated by the measured outlier ratios")

    return "; ".join(messages)


def write_text_report(
    report_path: Path,
    project_root: Path,
    input_paths: Dict[str, Path],
    all_records: Dict[str, List[Dict[str, Any]]],
    recommendations: Dict[str, List[Dict[str, str]]],
) -> None:
    lines: List[str] = []
    lines.extend(
        [
            "STAGE 6 — RFM/eRFM DISTRIBUTION & OUTLIER ANALYSIS",
            "=" * 100,
            f"Project root: {project_root}",
            "Analysis scope: verified RFM/eRFM Parquet only; no source overwrite and no automatic outlier removal.",
            f"Quantile method: approxQuantile(relativeError={QUANTILE_RELATIVE_ERROR})",
            "",
            "SEMANTIC NOTES",
            "- Frequency = number of distinct purchase dates per customer, not invoice/order count.",
            "- AvgBasketSize = average item count per distinct purchase date.",
            "- Monetary = SUM(price) on the cleaned transactions.",
            "- AnalysisDate and Recency follow Stage 5: max(t_dat) + 1 day and date difference to last purchase.",
            "",
            "INPUTS",
        ]
    )
    for name, path in input_paths.items():
        lines.append(f"{name}: {path}")

    for dataset_name, records in all_records.items():
        lines.extend(["", "=" * 100, f"DATASET: {dataset_name.upper()}", "=" * 100])
        lines.append(
            "Feature | Count | Mean | Std | Min | P01 | P05 | P25 | P50 | P75 | P95 | P99 | Max | Skewness | "
            "IQR outlier count | IQR outlier ratio % | P01-P99 outlier count | P01-P99 ratio % | "
            "P05-P95 outlier count | P05-P95 ratio %"
        )
        lines.append("-" * 260)
        for record in records:
            lines.append(
                " | ".join(
                    [
                        str(record["Feature"]),
                        str(record["Count"]),
                        str(record["Mean"]),
                        str(record["Std"]),
                        str(record["Min"]),
                        str(record["P01"]),
                        str(record["P05"]),
                        str(record["P25"]),
                        str(record["P50"]),
                        str(record["P75"]),
                        str(record["P95"]),
                        str(record["P99"]),
                        str(record["Max"]),
                        str(record["Skewness"]),
                        str(record["IQROutlierCount"]),
                        str(record["IQROutlierRatioPct"]),
                        str(record["P01P99OutlierCount"]),
                        str(record["P01P99OutlierRatioPct"]),
                        str(record["P05P95OutlierCount"]),
                        str(record["P05P95OutlierRatioPct"]),
                    ]
                )
            )

        lines.extend(["", "DATA QUALITY / NON-POSITIVE CHECKS"])
        for record in records:
            lines.append(
                f"{record['Feature']}: null={record['NullCount']} ({record['NullRatioPct']}%), "
                f"NaN={record['NaNCount']} ({record['NaNRatioPct']}%), "
                f"<=0={record['NonPositiveCount']} ({record['NonPositiveRatioPct']}%)"
            )

        lines.extend(["", "IQR BOUNDARIES AND EXTREME-VALUE IMPACT"])
        for record in records:
            lines.append(
                f"{record['Feature']}: IQR={record['IQR']}, lower={record['IQRLower']}, upper={record['IQRUpper']}; "
                f"P01-P99 low/high={record['P01P99LowCount']}/{record['P01P99HighCount']}; "
                f"P05-P95 low/high={record['P05P95LowCount']}/{record['P05P95HighCount']}"
            )

        lines.extend(["", "RAW VS LOG1P SKEWNESS"])
        for record in records:
            if record["Feature"] in LOG_FEATURES:
                lines.append(
                    f"{record['Feature']}: raw={record['Skewness']}, log1p={record['Log1pSkewness']}, "
                    f"absolute skewness reduction={record['SkewnessReductionPct']}%"
                )
            else:
                lines.append(f"{record['Feature']}: log1p not applied by design")

        lines.extend(["", "RECOMMENDED TRANSFORMATIONS / HANDLING FOR STAGE 7"])
        for recommendation_row in recommendations[dataset_name]:
            lines.append(
                f"{recommendation_row['Feature']}: {recommendation_row['Recommendation']}"
            )

    lines.extend(
        [
            "",
            "VALIDATION CHECKLIST",
            "- No full dataset was collected to Pandas.",
            "- NULL, NaN, non-positive values, skewness, IQR outliers, and percentile tails were measured.",
            "- IQR and percentile-capping effects are measured only; no rows were removed and no values were capped.",
            "- RFM/eRFM input namespaces were not overwritten.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def enable_windows_java_filesystem_fallback(spark: SparkSession) -> None:
    """Avoid Hadoop's optional NativeIO access check when winutils is absent.

    Hadoop 3.5 calls NativeIO$Windows.access0 while listing local Parquet
    directories on Windows.  The project machine has no Hadoop native
    binary, so that optional permission check raises UnsatisfiedLinkError.
    Spark itself can still use Java's normal File.canRead() implementation.
    This changes only Hadoop's in-process branch selection; it does not alter
    the input data or file permissions.
    """

    jvm = spark._jvm
    shell_class = jvm.java.lang.Class.forName("org.apache.hadoop.util.Shell")
    windows_field = shell_class.getDeclaredField("WINDOWS")
    windows_field.setAccessible(True)
    try:
        windows_field.setBoolean(None, False)
    except Exception:
        # Java 17+ may reject reflective writes to static final fields.  Unsafe
        # is used only as a last-resort local compatibility bridge for this
        # process; no installed jar or system setting is modified.
        unsafe_class = jvm.java.lang.Class.forName("sun.misc.Unsafe")
        unsafe_field = unsafe_class.getDeclaredField("theUnsafe")
        unsafe_field.setAccessible(True)
        unsafe = unsafe_field.get(None)
        base = unsafe.staticFieldBase(windows_field)
        offset = unsafe.staticFieldOffset(windows_field)
        unsafe.putBoolean(base, offset, False)
    print("Windows Hadoop NativeIO fallback: enabled")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / "output"
    features_root = output_root / "features"
    analysis_root = output_root / "analysis"
    report_path = project_root / "reports" / "rfm_distribution_analysis.txt"

    input_paths = {
        "rfm": features_root / "rfm",
        "erfm": features_root / "erfm",
    }

    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        print("ERROR: required input path(s) missing:")
        for path in missing:
            print(path)
        return 2

    analysis_root.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    for dataset_name in input_paths:
        (analysis_root / f"{dataset_name}_distribution").mkdir(
            parents=True,
            exist_ok=True,
        )

    spark_tmp = project_root.parent / "spark_tmp"
    spark_tmp.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .master("local[4]")
        .appName("Stage6-RFM-Distribution-Analysis")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.local.dir", str(spark_tmp))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    enable_windows_java_filesystem_fallback(spark)

    all_records: Dict[str, List[Dict[str, Any]]] = {}
    recommendations: Dict[str, List[Dict[str, str]]] = {}
    try:
        for dataset_name, path in input_paths.items():
            print(f"Loading {dataset_name}: {path}")
            df = spark.read.parquet(str(path))
            required_columns = FEATURES[dataset_name]
            missing_columns = [c for c in required_columns if c not in df.columns]
            if missing_columns:
                raise ValueError(
                    f"{dataset_name} missing required columns: {missing_columns}; "
                    f"available={df.columns}"
                )

            selected = df.select(*required_columns).persist(StorageLevel.MEMORY_AND_DISK)
            dataset_count = selected.count()
            if dataset_count == 0:
                raise ValueError(f"{dataset_name} input is empty")

            records: List[Dict[str, Any]] = []
            for feature in required_columns:
                record = feature_statistics(selected, feature, dataset_count)
                record["Dataset"] = dataset_name
                records.append(record)
                print(
                    f"{dataset_name}.{feature}: count={record['Count']}, "
                    f"skewness={record['Skewness']}, "
                    f"iqr_outlier_ratio={record['IQROutlierRatioPct']}%"
                )

            all_records[dataset_name] = records
            recommendations[dataset_name] = [
                {
                    "Feature": record["Feature"],
                    "Recommendation": recommendation(record),
                }
                for record in records
            ]

            summary_path = analysis_root / f"{dataset_name}_distribution" / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "dataset": dataset_name,
                        "input": str(path),
                        "row_count": dataset_count,
                        "quantile_relative_error": QUANTILE_RELATIVE_ERROR,
                        "features": records,
                        "recommendations": recommendations[dataset_name],
                    },
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            selected.unpersist()

        write_text_report(
            report_path,
            project_root,
            input_paths,
            all_records,
            recommendations,
        )
        print(f"Report written: {report_path}")
        print("Stage 6 completed successfully.")
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
