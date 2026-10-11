"""Stage 24: controlled CSV versus Parquet benchmark for transactions.

The benchmark compares the same logical transaction population.  It does not
cache or persist measured DataFrames; cache/page-cache effects are explicitly
treated as part of this controlled repeated-read experiment.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, count, countDistinct, lit, max as spark_max, min as spark_min, sum as spark_sum, to_date
from pyspark.sql.types import DateType, DoubleType, IntegerType, StringType, StructField, StructType


STAGE = 24
AUTHORITY = "CSV_PARQUET_IO_BENCHMARK_V1"
PRIMARY_RUNTIME_AUTHORITY = "PRIMARY_B0_M1_M2_RUNTIME_BENCHMARK_1M_V1"
EXPECTED_ROWS = 31_788_324
EXPECTED_MIN_DATE = "2018-09-20"
EXPECTED_MAX_DATE = "2020-09-22"
PRICE_SUM_RELATIVE_TOLERANCE = 1e-8
MEASURED_TRIALS = 3
WARMUP_TRIALS = 1
FORMAT_ORDER = ["CSV", "PARQUET"]
ROTATING_ORDER = [["CSV", "PARQUET"], ["PARQUET", "CSV"], ["CSV", "PARQUET"]]

TRANSACTION_SCHEMA = StructType([
    StructField("t_dat", DateType(), True),
    StructField("customer_id", StringType(), True),
    StructField("article_id", StringType(), True),
    StructField("price", DoubleType(), True),
    StructField("sales_channel_id", IntegerType(), True),
])
REQUIRED_COLUMNS = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        unsafe.putBoolean(unsafe.staticFieldBase(windows_field), unsafe.staticFieldOffset(windows_field), False)


def stats(values: List[float]) -> Dict[str, Any]:
    return {"min_sec": min(values), "max_sec": max(values), "mean_sec": statistics.mean(values), "median_sec": statistics.median(values), "population_std_sec": statistics.pstdev(values), "trial_times_sec": values, "all_finite": all(finite(value) and value >= 0 for value in values)}


def model_stage23_gate(project_root: Path) -> Tuple[List[str], Dict[str, Any]]:
    blockers: List[str] = []
    path = project_root / "output" / "benchmark" / "stage23" / "stage23_final_summary.json"
    if not path.exists():
        return ["Stage 23 final summary is missing."], {}
    summary = read_json(path)
    checks = {
        "stage": summary.get("stage") == 23,
        "authority": summary.get("authority") == PRIMARY_RUNTIME_AUTHORITY,
        "cohort_count": summary.get("cohort_count") == 1_000_000,
        "complete": summary.get("primary_runtime_benchmark_complete") is True,
        "ready": summary.get("ready_for_stage24") is True,
        "models_unmodified": summary.get("final_model_authorities_modified") is False,
    }
    for name, passed in checks.items():
        if not passed:
            blockers.append(f"Stage 23 gate mismatch: {name}.")
    return blockers, {"summary_path": str(path), "summary": summary, "checks": checks}


def discover_sources(project_root: Path) -> Tuple[List[str], Dict[str, Any]]:
    blockers: List[str] = []
    csv_path = project_root / "data" / "transactions_train.csv"
    parquet_path = project_root / "output" / "clean" / "transactions"
    stage2_script = project_root / "src" / "02_data_profiling.py"
    stage3_script = project_root / "src" / "03_clean_data.py"
    csv_files = [csv_path] if csv_path.is_file() else []
    parquet_files = sorted(path for path in parquet_path.rglob("*.parquet")) if parquet_path.is_dir() else []
    csv_header: List[str] = []
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            csv_header = next(csv.reader(handle))
    required_header = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
    script_evidence = {
        "stage2_script": str(stage2_script),
        "stage3_script": str(stage3_script),
        "stage2_exists": stage2_script.exists(),
        "stage3_exists": stage3_script.exists(),
        "stage2_contains_transactions_source": stage2_script.exists() and "transactions_train.csv" in stage2_script.read_text(encoding="utf-8"),
        "stage3_contains_transactions_source": stage3_script.exists() and "transactions_train.csv" in stage3_script.read_text(encoding="utf-8"),
        "stage3_contains_clean_transactions_output": stage3_script.exists() and "transactions_output" in stage3_script.read_text(encoding="utf-8") and "transactions_clean.write" in stage3_script.read_text(encoding="utf-8"),
    }
    checks = {
        "csv_exists": csv_path.is_file(),
        "parquet_directory_exists": parquet_path.is_dir(),
        "csv_data_file_count": len(csv_files) == 1,
        "parquet_data_file_count_positive": len(parquet_files) > 0,
        "csv_required_columns": all(column in csv_header for column in required_header),
        "stage2_stage3_evidence": all(script_evidence.values()),
    }
    for name, passed in checks.items():
        if not passed:
            blockers.append(f"Authoritative source discovery failed: {name}.")
    return blockers, {"csv_path": str(csv_path), "parquet_path": str(parquet_path), "csv_files": [str(path) for path in csv_files], "parquet_files": [str(path) for path in parquet_files], "csv_header": csv_header, "required_columns": required_header, "script_evidence": script_evidence, "checks": checks, "pass": not blockers}


def read_csv_authority(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.option("header", True).option("mode", "FAILFAST").schema(TRANSACTION_SCHEMA).csv(path).select(*REQUIRED_COLUMNS)


def read_parquet_authority(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.parquet(path).select(*REQUIRED_COLUMNS)


def semantic_stats(df: DataFrame) -> Dict[str, Any]:
    row = df.agg(
        count("*").alias("row_count"),
        countDistinct("customer_id").alias("distinct_customer_id"),
        spark_min("t_dat").alias("min_t_dat"),
        spark_max("t_dat").alias("max_t_dat"),
        spark_sum("price").alias("sum_price"),
        spark_min("price").alias("min_price"),
        spark_max("price").alias("max_price"),
        spark_sum(col("customer_id").isNull().cast("long")).alias("null_customer_id"),
        spark_sum(col("article_id").isNull().cast("long")).alias("null_article_id"),
        spark_sum(col("t_dat").isNull().cast("long")).alias("null_t_dat"),
        spark_sum(col("price").isNull().cast("long")).alias("null_price"),
    ).first().asDict()
    return {"row_count": int(row["row_count"]), "distinct_customer_id": int(row["distinct_customer_id"]), "min_t_dat": str(row["min_t_dat"]), "max_t_dat": str(row["max_t_dat"]), "sum_price": float(row["sum_price"]), "min_price": float(row["min_price"]), "max_price": float(row["max_price"]), "null_customer_id": int(row["null_customer_id"] or 0), "null_article_id": int(row["null_article_id"] or 0), "null_t_dat": int(row["null_t_dat"] or 0), "null_price": int(row["null_price"] or 0)}


def compare_semantics(csv_stats: Dict[str, Any], parquet_stats: Dict[str, Any]) -> Dict[str, Any]:
    sum_tolerance = PRICE_SUM_RELATIVE_TOLERANCE * max(1.0, abs(csv_stats["sum_price"]))
    exact_fields = ["row_count", "distinct_customer_id", "min_t_dat", "max_t_dat", "min_price", "max_price", "null_customer_id", "null_article_id", "null_t_dat", "null_price"]
    field_comparison = {field: {"csv": csv_stats[field], "parquet": parquet_stats[field], "equal": csv_stats[field] == parquet_stats[field]} for field in exact_fields}
    field_comparison["sum_price"] = {"csv": csv_stats["sum_price"], "parquet": parquet_stats["sum_price"], "absolute_difference": abs(csv_stats["sum_price"] - parquet_stats["sum_price"]), "tolerance": sum_tolerance, "equal_within_tolerance": abs(csv_stats["sum_price"] - parquet_stats["sum_price"]) <= sum_tolerance}
    pass_value = all(item["equal"] for key, item in field_comparison.items() if key != "sum_price") and field_comparison["sum_price"]["equal_within_tolerance"] and csv_stats["row_count"] == EXPECTED_ROWS and csv_stats["min_t_dat"] == EXPECTED_MIN_DATE and csv_stats["max_t_dat"] == EXPECTED_MAX_DATE and parquet_stats["row_count"] == EXPECTED_ROWS and parquet_stats["min_t_dat"] == EXPECTED_MIN_DATE and parquet_stats["max_t_dat"] == EXPECTED_MAX_DATE
    return {"expected_population": {"row_count": EXPECTED_ROWS, "min_t_dat": EXPECTED_MIN_DATE, "max_t_dat": EXPECTED_MAX_DATE}, "price_sum_tolerance_definition": f"absolute difference <= {PRICE_SUM_RELATIVE_TOLERANCE} * max(1, abs(CSV sum))", "csv": csv_stats, "parquet": parquet_stats, "field_comparison": field_comparison, "pass": pass_value}


def file_sizes(source: Dict[str, Any]) -> Dict[str, Any]:
    csv_paths = [Path(path) for path in source["csv_files"]]
    parquet_paths = [Path(path) for path in source["parquet_files"]]
    csv_bytes = sum(path.stat().st_size for path in csv_paths)
    parquet_bytes = sum(path.stat().st_size for path in parquet_paths)
    if csv_bytes <= 0 or parquet_bytes <= 0:
        raise ValueError("Physical source size could not be measured.")
    return {"csv_bytes": csv_bytes, "parquet_bytes": parquet_bytes, "parquet_to_csv_ratio": parquet_bytes / csv_bytes, "storage_reduction_pct": (csv_bytes - parquet_bytes) / csv_bytes * 100.0, "csv_files": [{"path": str(path), "bytes": path.stat().st_size} for path in csv_paths], "parquet_data_files": [{"path": str(path), "bytes": path.stat().st_size} for path in parquet_paths], "metadata_excluded": True, "size_measurement": "CSV sums .csv files; Parquet recursively sums .parquet data files only."}


def execute_q1(spark: SparkSession, fmt: str, source: Dict[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    df = read_csv_authority(spark, source["csv_path"]) if fmt == "CSV" else read_parquet_authority(spark, source["parquet_path"])
    result = int(df.count())
    elapsed = time.perf_counter() - started
    return {"format": fmt, "runtime_sec": elapsed, "row_count": result, "result": {"row_count": result}, "pass": finite(elapsed) and elapsed >= 0 and result == EXPECTED_ROWS}


def execute_q2(spark: SparkSession, fmt: str, source: Dict[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    df = read_csv_authority(spark, source["csv_path"]) if fmt == "CSV" else read_parquet_authority(spark, source["parquet_path"])
    result = df.select("customer_id", "t_dat", "price").where(col("t_dat") >= lit(date(2020, 1, 1))).agg(count("*").alias("filtered_row_count"), spark_sum("price").alias("filtered_sum_price")).first().asDict()
    elapsed = time.perf_counter() - started
    return {"format": fmt, "runtime_sec": elapsed, "result": {"filtered_row_count": int(result["filtered_row_count"]), "filtered_sum_price": float(result["filtered_sum_price"])}, "query": "select customer_id,t_dat,price where t_dat >= 2020-01-01; count and sum(price)", "pass": finite(elapsed) and elapsed >= 0}


def execute_q3(spark: SparkSession, fmt: str, source: Dict[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    df = read_csv_authority(spark, source["csv_path"]) if fmt == "CSV" else read_parquet_authority(spark, source["parquet_path"])
    grouped = df.select("customer_id", "t_dat", "price").groupBy("customer_id").agg(count("*").alias("transaction_row_count"), spark_sum("price").alias("customer_sum_price"), spark_max("t_dat").alias("customer_max_t_dat"))
    result = grouped.agg(count("*").alias("grouped_customer_count"), spark_sum("transaction_row_count").alias("sum_grouped_transaction_rows"), spark_sum("customer_sum_price").alias("sum_grouped_monetary"), spark_max("customer_max_t_dat").alias("max_customer_max_t_dat")).first().asDict()
    elapsed = time.perf_counter() - started
    return {"format": fmt, "runtime_sec": elapsed, "result": {"grouped_customer_count": int(result["grouped_customer_count"]), "sum_grouped_transaction_rows": int(result["sum_grouped_transaction_rows"]), "sum_grouped_monetary": float(result["sum_grouped_monetary"]), "max_customer_max_t_dat": str(result["max_customer_max_t_dat"])}, "query": "select customer_id,t_dat,price groupBy customer_id; count, sum(price), max(t_dat); scalar aggregate", "pass": finite(elapsed) and elapsed >= 0}


def capture_plan(spark: SparkSession, fmt: str, source: Dict[str, Any], query_id: str) -> str:
    df = read_csv_authority(spark, source["csv_path"]) if fmt == "CSV" else read_parquet_authority(spark, source["parquet_path"])
    if query_id == "Q2":
        query_df = df.select("customer_id", "t_dat", "price").where(col("t_dat") >= lit(date(2020, 1, 1))).agg(count("*").alias("filtered_row_count"), spark_sum("price").alias("filtered_sum_price"))
    else:
        grouped = df.select("customer_id", "t_dat", "price").groupBy("customer_id").agg(count("*").alias("transaction_row_count"), spark_sum("price").alias("customer_sum_price"), spark_max("t_dat").alias("customer_max_t_dat"))
        query_df = grouped.agg(count("*").alias("grouped_customer_count"), spark_sum("transaction_row_count").alias("sum_grouped_transaction_rows"), spark_sum("customer_sum_price").alias("sum_grouped_monetary"), spark_max("customer_max_t_dat").alias("max_customer_max_t_dat"))
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        query_df.explain(mode="formatted")
    return output.getvalue()


def plan_findings(plan: str) -> Dict[str, Any]:
    lines = [line.strip() for line in plan.splitlines() if line.strip()]
    evidence = [line for line in lines if any(token in line for token in ["FileScan", "PushedFilters", "ReadSchema", "PartitionFilters", "DataFilters", "Scan"])]
    return {"file_scan_evidence_lines": evidence, "plan_line_count": len(lines), "plan_captured_without_inference": True}


def equality_for_query(query_id: str, csv_result: Dict[str, Any], parquet_result: Dict[str, Any]) -> Dict[str, Any]:
    if query_id == "Q1":
        comparisons = {"row_count": {"csv": csv_result["row_count"], "parquet": parquet_result["row_count"], "equal": csv_result["row_count"] == parquet_result["row_count"]}}
    elif query_id == "Q2":
        tolerance = PRICE_SUM_RELATIVE_TOLERANCE * max(1.0, abs(csv_result["filtered_sum_price"]))
        comparisons = {"filtered_row_count": {"csv": csv_result["filtered_row_count"], "parquet": parquet_result["filtered_row_count"], "equal": csv_result["filtered_row_count"] == parquet_result["filtered_row_count"]}, "filtered_sum_price": {"csv": csv_result["filtered_sum_price"], "parquet": parquet_result["filtered_sum_price"], "absolute_difference": abs(csv_result["filtered_sum_price"] - parquet_result["filtered_sum_price"]), "tolerance": tolerance, "equal_within_tolerance": abs(csv_result["filtered_sum_price"] - parquet_result["filtered_sum_price"]) <= tolerance}}
    else:
        tolerance = PRICE_SUM_RELATIVE_TOLERANCE * max(1.0, abs(csv_result["sum_grouped_monetary"]))
        comparisons = {key: {"csv": csv_result[key], "parquet": parquet_result[key], "equal": csv_result[key] == parquet_result[key]} for key in ["grouped_customer_count", "sum_grouped_transaction_rows", "max_customer_max_t_dat"]}
        comparisons["sum_grouped_monetary"] = {"csv": csv_result["sum_grouped_monetary"], "parquet": parquet_result["sum_grouped_monetary"], "absolute_difference": abs(csv_result["sum_grouped_monetary"] - parquet_result["sum_grouped_monetary"]), "tolerance": tolerance, "equal_within_tolerance": abs(csv_result["sum_grouped_monetary"] - parquet_result["sum_grouped_monetary"]) <= tolerance}
    passed = all(item.get("equal", item.get("equal_within_tolerance", False)) for item in comparisons.values())
    return {"query_id": query_id, "comparisons": comparisons, "pass": passed}


def run_trials(spark: SparkSession, source: Dict[str, Any], query_id: str, executor: Callable[[SparkSession, str, Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
    warmup: List[Dict[str, Any]] = []
    for fmt in FORMAT_ORDER:
        result = executor(spark, fmt, source)
        warmup.append({"format": fmt, "runtime_sec": result["runtime_sec"], "result": result["result"], "pass": result["pass"]})
    measured: List[Dict[str, Any]] = []
    by_format = {fmt: [] for fmt in FORMAT_ORDER}
    for round_index, order in enumerate(ROTATING_ORDER, 1):
        for order_index, fmt in enumerate(order, 1):
            result = executor(spark, fmt, source)
            result.update({"round": round_index, "order_index": order_index, "trial": len(by_format[fmt]) + 1})
            measured.append(result)
            by_format[fmt].append(result)
    summary = {fmt: stats([float(record["runtime_sec"]) for record in by_format[fmt]]) for fmt in FORMAT_ORDER}
    if not all(record["pass"] and summary[record["format"]]["all_finite"] for record in measured):
        raise ValueError(f"{query_id} contained a failed or non-finite measured trial.")
    return {"query_id": query_id, "warmup_count": WARMUP_TRIALS, "measured_count": MEASURED_TRIALS, "warmup": warmup, "measured_order": ROTATING_ORDER, "trials": measured, "summary_by_format": summary}


def runtime_summary_entry(query_id: str, name: str, trial_package: Dict[str, Any], equality: Dict[str, Any]) -> Dict[str, Any]:
    csv_summary = trial_package["summary_by_format"]["CSV"]
    parquet_summary = trial_package["summary_by_format"]["PARQUET"]
    csv_median = csv_summary["median_sec"]
    parquet_median = parquet_summary["median_sec"]
    return {"query_id": query_id, "query_name": name, "CSV": csv_summary, "PARQUET": parquet_summary, "csv_over_parquet_ratio": csv_median / parquet_median, "parquet_runtime_reduction_pct": (csv_median - parquet_median) / csv_median * 100.0, "result_equality_pass": equality["pass"]}


def readback(paths: List[Path], project_root: Path) -> Dict[str, bool]:
    output: Dict[str, bool] = {}
    for path in paths:
        ok = path.exists() and path.stat().st_size > 0
        if ok and path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                ok = False
        if ok and path.suffix == ".csv":
            ok = len(path.read_text(encoding="utf-8").splitlines()) >= 4
        output[str(path.relative_to(project_root))] = ok
    return output


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "output" / "benchmark" / "stage24"
    reports_dir = project_root / "reports"
    report_path = reports_dir / "stage24_csv_vs_parquet_benchmark.txt"
    summary_path = output_dir / "stage24_final_summary.json"
    event_dir = output_dir / "spark-events"
    blockers: List[str] = []
    warnings: List[str] = ["This is a controlled repeated-read benchmark, not a cold-disk benchmark; Windows filesystem/page cache cannot be cleared reliably.", "Runtime differences include parsing, decoding, column pruning, predicate pushdown, planning, shuffle, and scheduling effects; they are not interpreted as pure disk I/O."]
    spark: SparkSession | None = None
    event_logging_enabled = False
    event_logging_issue: str | None = None
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        event_dir.mkdir(parents=True, exist_ok=True)
        stage23_blockers, stage23_context = model_stage23_gate(project_root)
        source_blockers, source = discover_sources(project_root)
        blockers.extend(stage23_blockers)
        blockers.extend(source_blockers)
        write_json(output_dir / "source_authority_validation.json", {"stage23_gate": stage23_context, "source_discovery": source, "pass": not blockers})
        if blockers:
            failed = {"stage": STAGE, "authority": AUTHORITY, "source_row_count": None, "semantic_equivalence_pass": False, "query_result_equality_pass": False, "cache_used_in_measured_trials": False, "primary_runtime_authority_modified": False, "models_modified": False, "benchmark_complete": False, "ready_for_stage25": False, "blockers": blockers}
            write_json(summary_path, failed)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("STAGE 24 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_25: False\n", encoding="utf-8")
            return 3

        event_uri = event_dir.resolve().as_uri()
        builder = SparkSession.builder.master("local[4]").appName("Stage24-Controlled-CSV-vs-Parquet-Benchmark").config("spark.sql.shuffle.partitions", "64").config("spark.sql.adaptive.enabled", "true").config("spark.local.dir", str(project_root.parent / "spark_tmp"))
        try:
            spark = builder.config("spark.eventLog.enabled", "true").config("spark.eventLog.dir", event_uri).getOrCreate()
            event_logging_enabled = True
        except Exception as event_error:
            event_logging_issue = f"Spark event logging unavailable on this Windows runtime: {type(event_error).__name__}: {event_error}"
            warnings.append(event_logging_issue)
            if spark is not None:
                try:
                    spark.stop()
                except Exception:
                    pass
                spark = None
            spark = builder.config("spark.eventLog.enabled", "false").getOrCreate()
            event_logging_enabled = False
        assert spark is not None
        spark.sparkContext.setLogLevel("ERROR")
        enable_windows_java_filesystem_fallback(spark)

        csv_df = read_csv_authority(spark, source["csv_path"])
        parquet_df = read_parquet_authority(spark, source["parquet_path"])
        csv_semantic = semantic_stats(csv_df)
        parquet_semantic = semantic_stats(parquet_df)
        semantic_equivalence = compare_semantics(csv_semantic, parquet_semantic)
        write_json(output_dir / "format_semantic_equivalence.json", semantic_equivalence)
        if not semantic_equivalence["pass"]:
            raise ValueError("CSV and Parquet semantic validation failed or transaction population is invalid.")

        storage = file_sizes(source)
        write_json(output_dir / "storage_size_comparison.json", storage)

        q1 = run_trials(spark, source, "Q1", execute_q1)
        q2 = run_trials(spark, source, "Q2", execute_q2)
        q3 = run_trials(spark, source, "Q3", execute_q3)
        write_json(output_dir / "full_scan_trials.json", q1)
        write_json(output_dir / "projection_filter_trials.json", q2)
        write_json(output_dir / "customer_aggregation_trials.json", q3)

        q1_results = {fmt: q1["trials"][next(index for index, item in enumerate(q1["trials"]) if item["format"] == fmt)]["result"] for fmt in FORMAT_ORDER}
        q2_results = {fmt: q2["trials"][next(index for index, item in enumerate(q2["trials"]) if item["format"] == fmt)]["result"] for fmt in FORMAT_ORDER}
        q3_results = {fmt: q3["trials"][next(index for index, item in enumerate(q3["trials"]) if item["format"] == fmt)]["result"] for fmt in FORMAT_ORDER}
        equalities = {"Q1": equality_for_query("Q1", q1_results["CSV"], q1_results["PARQUET"]), "Q2": equality_for_query("Q2", q2_results["CSV"], q2_results["PARQUET"]), "Q3": equality_for_query("Q3", q3_results["CSV"], q3_results["PARQUET"])}
        write_json(output_dir / "query_result_equality.json", {"queries": equalities, "price_sum_tolerance_definition": f"absolute difference <= {PRICE_SUM_RELATIVE_TOLERANCE} * max(1, abs(CSV sum))", "pass": all(item["pass"] for item in equalities.values())})
        if not all(item["pass"] for item in equalities.values()):
            raise ValueError("CSV and Parquet query result equality failed.")

        plans = {}
        for query_id in ["Q2", "Q3"]:
            for fmt in FORMAT_ORDER:
                plan_text = capture_plan(spark, fmt, source, query_id)
                plan_path = output_dir / f"physical_plan_{fmt.lower()}_{query_id.lower()}.txt"
                plan_path.write_text(plan_text, encoding="utf-8")
                plans[f"{fmt}_{query_id}"] = {"path": str(plan_path), **plan_findings(plan_text)}

        names = {"Q1": "FULL_SCAN", "Q2": "PROJECTION_FILTER", "Q3": "CUSTOMER_AGGREGATION"}
        runtime_entries = {query_id: runtime_summary_entry(query_id, names[query_id], package, equalities[query_id]) for query_id, package in [("Q1", q1), ("Q2", q2), ("Q3", q3)]}
        write_json(output_dir / "runtime_summary.json", runtime_entries)
        ratios = {query_id: {"csv_median_sec": entry["CSV"]["median_sec"], "parquet_median_sec": entry["PARQUET"]["median_sec"], "csv_over_parquet_ratio": entry["csv_over_parquet_ratio"], "parquet_runtime_reduction_pct": entry["parquet_runtime_reduction_pct"], "interpretation_rule": "ratio > 1 means Parquet is faster for this query"} for query_id, entry in runtime_entries.items()}
        write_json(output_dir / "runtime_ratios.json", ratios)
        write_csv(output_dir / "csv_vs_parquet_benchmark_summary.csv", ["QueryID", "QueryName", "CSVMedianSec", "ParquetMedianSec", "CSVMeanSec", "ParquetMeanSec", "CSVStdSec", "ParquetStdSec", "CSVOverParquetRatio", "ParquetRuntimeReductionPct", "ResultEqualityPass"], [{"QueryID": query_id, "QueryName": entry["query_name"], "CSVMedianSec": entry["CSV"]["median_sec"], "ParquetMedianSec": entry["PARQUET"]["median_sec"], "CSVMeanSec": entry["CSV"]["mean_sec"], "ParquetMeanSec": entry["PARQUET"]["mean_sec"], "CSVStdSec": entry["CSV"]["population_std_sec"], "ParquetStdSec": entry["PARQUET"]["population_std_sec"], "CSVOverParquetRatio": entry["csv_over_parquet_ratio"], "ParquetRuntimeReductionPct": entry["parquet_runtime_reduction_pct"], "ResultEqualityPass": entry["result_equality_pass"]} for query_id, entry in runtime_entries.items()])

        application_id = spark.sparkContext.applicationId
        ui_port = spark.conf.get("spark.ui.port", "4040")
        environment: Dict[str, Any] = {"stage": STAGE, "authority": AUTHORITY, "spark_version": spark.version, "python_version": platform.python_version(), "java_version": spark._jvm.java.lang.System.getProperty("java.version"), "os": platform.platform(), "cpu_logical_processors": os.cpu_count(), "system_memory_bytes": None, "application_id": application_id, "application_name": "Stage24-Controlled-CSV-vs-Parquet-Benchmark", "master": "local[4]", "shuffle_partitions": 64, "adaptive_enabled": True, "timing_api": "time.perf_counter", "warmup_trials_per_format_query": WARMUP_TRIALS, "measured_trials_per_format_query": MEASURED_TRIALS, "primary_runtime_statistic": "median", "cache_used_in_measured_trials": False, "persist_used_in_measured_trials": False, "order_rotation": ROTATING_ORDER, "benchmark_interpretation": "controlled repeated-read benchmark; not cold disk", "event_log_enabled": event_logging_enabled, "event_log_path": event_uri, "event_log_issue": event_logging_issue, "spark_ui_url_during_application": f"http://localhost:{ui_port}", "warnings": warnings}
        try:
            import psutil
            environment["system_memory_bytes"] = int(psutil.virtual_memory().total)
        except Exception as memory_error:
            environment["system_memory_unavailable_reason"] = f"{type(memory_error).__name__}: {memory_error}"
        write_json(output_dir / "benchmark_environment.json", environment)

        query_equality_pass = all(item["pass"] for item in equalities.values())
        primary_finding = "MIXED_RESULT"
        if storage["storage_reduction_pct"] > 0 and all(entry["csv_over_parquet_ratio"] > 1 for entry in runtime_entries.values()):
            primary_finding = "PARQUET_CLEAR_ADVANTAGE"
        elif storage["storage_reduction_pct"] > 0 and any(entry["csv_over_parquet_ratio"] <= 1 for entry in runtime_entries.values()):
            primary_finding = "PARQUET_STORAGE_ADVANTAGE_RUNTIME_MIXED"
        elif all(abs(entry["parquet_runtime_reduction_pct"]) < 5 for entry in runtime_entries.values()):
            primary_finding = "RUNTIME_SIMILAR"

        summary: Dict[str, Any] = {"stage": STAGE, "authority": AUTHORITY, "source_row_count": EXPECTED_ROWS, "source_paths": {"csv": source["csv_path"], "parquet": source["parquet_path"]}, "semantic_equivalence_pass": semantic_equivalence["pass"], "storage": storage, "queries": runtime_entries, "query_result_equality_pass": query_equality_pass, "physical_plan_evidence": plans, "primary_finding": primary_finding, "cache_used_in_measured_trials": False, "primary_runtime_authority_modified": False, "models_modified": False, "stage23_authority_preserved": True, "event_log_enabled": event_logging_enabled, "benchmark_complete": True, "ready_for_stage25": True, "warnings": warnings, "blockers": []}
        write_json(summary_path, summary)

        spark_manifest = {"application_id": application_id, "application_name": environment["application_name"], "spark_ui_url": environment["spark_ui_url_during_application"], "event_log_enabled": event_logging_enabled, "event_log_path": event_uri, "event_log_issue": event_logging_issue, "physical_plan_files": plans, "screenshots_fabricated": False, "evidence_scope": "captured Spark formatted physical plans and runtime environment only"}

        if spark is not None:
            spark.stop()
            spark = None

        artifact_paths = [output_dir / name for name in ["benchmark_environment.json", "source_authority_validation.json", "format_semantic_equivalence.json", "storage_size_comparison.json", "full_scan_trials.json", "projection_filter_trials.json", "customer_aggregation_trials.json", "runtime_summary.json", "runtime_ratios.json", "query_result_equality.json", "csv_vs_parquet_benchmark_summary.csv", "physical_plan_csv_q2.txt", "physical_plan_parquet_q2.txt", "physical_plan_csv_q3.txt", "physical_plan_parquet_q3.txt", "stage24_final_summary.json"]] + [report_path, reports_dir / "stage24_run.log"]
        report_lines = ["STAGE 24 — CONTROLLED CSV VS PARQUET STORAGE & QUERY BENCHMARK", "", f"AUTHORITY: {AUTHORITY}", f"SOURCE_ROWS: {EXPECTED_ROWS}", f"CSV_PATH: {source['csv_path']}", f"PARQUET_PATH: {source['parquet_path']}", f"SPARK_APPLICATION_ID: {application_id}", f"PRIMARY_FINDING: {primary_finding}", f"SEMANTIC_EQUIVALENCE_PASS: {semantic_equivalence['pass']}", f"QUERY_RESULT_EQUALITY_PASS: {query_equality_pass}", f"CACHE_USED_IN_MEASURED_TRIALS: False", "", "STORAGE:", json.dumps(storage, ensure_ascii=False, indent=2), "", "RUNTIME_SUMMARY:", json.dumps(runtime_entries, ensure_ascii=False, indent=2), "", "RUNTIME_RATIOS:", json.dumps(ratios, ensure_ascii=False, indent=2), "", "PHYSICAL_PLAN_FINDINGS:", json.dumps(plans, ensure_ascii=False, indent=2), "", "PRIMARY_RUNTIME_AUTHORITY_MODIFIED: False", "MODELS_MODIFIED: False", "", "BENCHMARK_COMPLETE: True", "READY_FOR_STAGE_25: True"]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        artifact_readback = readback(artifact_paths, project_root)
        summary["artifact_readback"] = artifact_readback
        summary["artifact_readback_pass"] = all(artifact_readback.values())
        summary["benchmark_complete"] = bool(summary["benchmark_complete"] and summary["artifact_readback_pass"])
        summary["ready_for_stage25"] = bool(summary["benchmark_complete"])
        if not summary["artifact_readback_pass"]:
            summary["blockers"] = ["One or more Stage 24 artifacts failed read-back."]
        write_json(summary_path, summary)
        print("STAGE23_GATE_PASS=True")
        print("SOURCE_DISCOVERY_PASS=True")
        print("SEMANTIC_EQUIVALENCE_PASS=True")
        print("QUERY_RESULT_EQUALITY_PASS=True")
        print(f"SPARK_APPLICATION_ID={application_id}")
        print(f"PRIMARY_FINDING={primary_finding}")
        print(f"BENCHMARK_COMPLETE={summary['benchmark_complete']}")
        print(f"READY_FOR_STAGE_25={summary['ready_for_stage25']}")
        return 0 if summary["ready_for_stage25"] else 3
    except Exception as error:
        blockers.append(f"{type(error).__name__}: {error}")
        failed = {"stage": STAGE, "authority": AUTHORITY, "source_row_count": EXPECTED_ROWS, "semantic_equivalence_pass": False, "query_result_equality_pass": False, "cache_used_in_measured_trials": False, "primary_runtime_authority_modified": False, "models_modified": False, "benchmark_complete": False, "ready_for_stage25": False, "warnings": warnings, "blockers": blockers}
        write_json(summary_path, failed)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("STAGE 24 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_25: False\n", encoding="utf-8")
        print(f"STAGE 24 FAILED CLOSED: {error}")
        return 3
    finally:
        if spark is not None:
            try:
                spark.stop()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
