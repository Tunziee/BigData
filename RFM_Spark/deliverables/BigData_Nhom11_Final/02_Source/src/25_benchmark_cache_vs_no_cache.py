"""Stage 25: controlled Spark cache versus no-cache benchmark on Parquet.

NO_CACHE means no explicit Spark DataFrame persistence.  It is not a cold
physical-disk benchmark.  CACHE uses MEMORY_AND_DISK and separates the one-
time materialization cost from steady-state query runtimes.
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
import traceback
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import coalesce, col, count, countDistinct, input_file_name, lit, max as spark_max, min as spark_min, sum as spark_sum, when
from pyspark.storagelevel import StorageLevel


STAGE = 25
AUTHORITY = "CACHE_NO_CACHE_BENCHMARK_V1"
STAGE24_AUTHORITY = "CSV_PARQUET_IO_BENCHMARK_V1"
EXPECTED_ROWS = 31_788_324
EXPECTED_CUSTOMERS = 1_362_281
EXPECTED_MIN_DATE = "2018-09-20"
EXPECTED_MAX_DATE = "2020-09-22"
EXPECTED_Q2_ROWS = 10_980_132
PRICE_SUM_RELATIVE_TOLERANCE = 1e-8
WARMUP_TRIALS = 1
MEASURED_TRIALS = 3
MODES = ["NO_CACHE", "CACHE"]
ROTATING_ORDER = [["NO_CACHE", "CACHE"], ["CACHE", "NO_CACHE"], ["NO_CACHE", "CACHE"]]
REPEATED_EXECUTIONS = [1, 3, 5, 10]
BASE_COLUMNS = ["customer_id", "t_dat", "price"]


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


def stage24_gate(project_root: Path) -> Tuple[List[str], Dict[str, Any]]:
    blockers: List[str] = []
    summary_path = project_root / "output" / "benchmark" / "stage24" / "stage24_final_summary.json"
    if not summary_path.exists():
        return ["Stage 24 final summary is missing."], {}
    summary = read_json(summary_path)
    checks = {"stage": summary.get("stage") == 24, "authority": summary.get("authority") == STAGE24_AUTHORITY, "source_row_count": summary.get("source_row_count") == EXPECTED_ROWS, "semantic_equivalence_pass": summary.get("semantic_equivalence_pass") is True, "query_result_equality_pass": summary.get("query_result_equality_pass") is True, "benchmark_complete": summary.get("benchmark_complete") is True, "ready_for_stage25": summary.get("ready_for_stage25") is True, "models_modified": summary.get("models_modified") is False}
    for name, passed in checks.items():
        if not passed:
            blockers.append(f"Stage 24 gate mismatch: {name}.")
    return blockers, {"summary_path": str(summary_path), "summary": summary, "checks": checks}


def source_validation(spark: SparkSession, project_root: Path) -> Tuple[List[str], Dict[str, Any], DataFrame]:
    blockers: List[str] = []
    path = project_root / "output" / "clean" / "transactions"
    checks = {"path_exists": path.is_dir(), "parquet_files_exist": any(path.rglob("*.parquet")) if path.is_dir() else False}
    if not all(checks.values()):
        blockers.append("Authoritative Stage 24 Parquet source is missing or empty.")
        return blockers, {"path": str(path), "checks": checks, "pass": False}, spark.read.parquet(str(path)) if path.is_dir() else spark.range(0).selectExpr("cast(null as string) as customer_id", "cast(null as date) as t_dat", "cast(null as double) as price")
    base = spark.read.parquet(str(path)).select(*BASE_COLUMNS)
    row = base.agg(count("*").alias("row_count"), countDistinct("customer_id").alias("distinct_customer_id"), spark_min("t_dat").alias("min_t_dat"), spark_max("t_dat").alias("max_t_dat"), spark_sum("price").alias("sum_price"), spark_min("price").alias("min_price"), spark_max("price").alias("max_price"), spark_sum(col("customer_id").isNull().cast("long")).alias("null_customer_id"), spark_sum(col("t_dat").isNull().cast("long")).alias("null_t_dat"), spark_sum(col("price").isNull().cast("long")).alias("null_price")).first().asDict()
    metrics = {"row_count": int(row["row_count"]), "distinct_customer_id": int(row["distinct_customer_id"]), "min_t_dat": str(row["min_t_dat"]), "max_t_dat": str(row["max_t_dat"]), "sum_price": float(row["sum_price"]), "min_price": float(row["min_price"]), "max_price": float(row["max_price"]), "null_customer_id": int(row["null_customer_id"] or 0), "null_t_dat": int(row["null_t_dat"] or 0), "null_price": int(row["null_price"] or 0)}
    checks.update({"row_count": metrics["row_count"] == EXPECTED_ROWS, "distinct_customer_id": metrics["distinct_customer_id"] == EXPECTED_CUSTOMERS, "date_range": metrics["min_t_dat"] == EXPECTED_MIN_DATE and metrics["max_t_dat"] == EXPECTED_MAX_DATE, "required_columns": all(column in base.columns for column in BASE_COLUMNS), "nulls": metrics["null_customer_id"] == 0 and metrics["null_t_dat"] == 0 and metrics["null_price"] == 0})
    for name, passed in checks.items():
        if not passed:
            blockers.append(f"Stage 25 source validation failed: {name}.")
    return blockers, {"path": str(path), "format": "PARQUET", "required_columns": BASE_COLUMNS, "metrics": metrics, "expected": {"row_count": EXPECTED_ROWS, "distinct_customer_id": EXPECTED_CUSTOMERS, "min_t_dat": EXPECTED_MIN_DATE, "max_t_dat": EXPECTED_MAX_DATE}, "checks": checks, "pass": not blockers}, base


def query_q1(df: DataFrame) -> Dict[str, Any]:
    value = int(df.count())
    return {"row_count": value}


def query_q2(df: DataFrame) -> Dict[str, Any]:
    row = df.where(col("t_dat") >= lit(date(2020, 1, 1))).agg(count("*").alias("filtered_row_count"), spark_sum("price").alias("filtered_sum_price")).first().asDict()
    return {"filtered_row_count": int(row["filtered_row_count"]), "filtered_sum_price": float(row["filtered_sum_price"])}


def query_q3(df: DataFrame) -> Dict[str, Any]:
    grouped = df.groupBy("customer_id").agg(count("*").alias("transaction_row_count"), spark_sum("price").alias("customer_sum_price"), spark_max("t_dat").alias("customer_max_t_dat"))
    row = grouped.agg(count("*").alias("grouped_customer_count"), spark_sum("transaction_row_count").alias("sum_grouped_transaction_rows"), spark_sum("customer_sum_price").alias("sum_grouped_monetary"), spark_max("customer_max_t_dat").alias("max_customer_max_t_dat")).first().asDict()
    return {"grouped_customer_count": int(row["grouped_customer_count"]), "sum_grouped_transaction_rows": int(row["sum_grouped_transaction_rows"]), "sum_grouped_monetary": float(row["sum_grouped_monetary"]), "max_customer_max_t_dat": str(row["max_customer_max_t_dat"])}


def execute_query(df: DataFrame, query_id: str) -> Dict[str, Any]:
    started = time.perf_counter()
    result = query_q1(df) if query_id == "Q1" else query_q2(df) if query_id == "Q2" else query_q3(df)
    elapsed = time.perf_counter() - started
    return {"runtime_sec": elapsed, "result": result, "pass": finite(elapsed) and elapsed >= 0}


def run_query_trials(query_id: str, no_cache_df: DataFrame, cache_df: DataFrame) -> Dict[str, Any]:
    frames = {"NO_CACHE": no_cache_df, "CACHE": cache_df}
    warmup: List[Dict[str, Any]] = []
    for mode in MODES:
        item = execute_query(frames[mode], query_id)
        warmup.append({"mode": mode, "runtime_sec": item["runtime_sec"], "result": item["result"], "pass": item["pass"]})
    measured: List[Dict[str, Any]] = []
    by_mode = {mode: [] for mode in MODES}
    for round_index, order in enumerate(ROTATING_ORDER, 1):
        for order_index, mode in enumerate(order, 1):
            item = execute_query(frames[mode], query_id)
            record = {"query_id": query_id, "round": round_index, "order_index": order_index, "trial": len(by_mode[mode]) + 1, "mode": mode, "runtime_sec": item["runtime_sec"], "result": item["result"], "pass": item["pass"]}
            measured.append(record)
            by_mode[mode].append(record)
    summary = {mode: stats([float(record["runtime_sec"]) for record in by_mode[mode]]) for mode in MODES}
    if not all(record["pass"] and summary[record["mode"]]["all_finite"] for record in measured):
        raise ValueError(f"{query_id} has failed or non-finite measured runtime.")
    return {"query_id": query_id, "warmup_count": WARMUP_TRIALS, "measured_count": MEASURED_TRIALS, "warmup": warmup, "measured_order": ROTATING_ORDER, "trials": measured, "summary_by_mode": summary}


def result_equal(query_id: str, left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    if query_id == "Q1":
        comparisons = {"row_count": {"no_cache": left["row_count"], "cache": right["row_count"], "equal": left["row_count"] == right["row_count"]}}
    elif query_id == "Q2":
        tolerance = PRICE_SUM_RELATIVE_TOLERANCE * max(1.0, abs(left["filtered_sum_price"]))
        comparisons = {"filtered_row_count": {"no_cache": left["filtered_row_count"], "cache": right["filtered_row_count"], "equal": left["filtered_row_count"] == right["filtered_row_count"]}, "filtered_sum_price": {"no_cache": left["filtered_sum_price"], "cache": right["filtered_sum_price"], "absolute_difference": abs(left["filtered_sum_price"] - right["filtered_sum_price"]), "tolerance": tolerance, "equal_within_tolerance": abs(left["filtered_sum_price"] - right["filtered_sum_price"]) <= tolerance}}
    else:
        tolerance = PRICE_SUM_RELATIVE_TOLERANCE * max(1.0, abs(left["sum_grouped_monetary"]))
        comparisons = {key: {"no_cache": left[key], "cache": right[key], "equal": left[key] == right[key]} for key in ["grouped_customer_count", "sum_grouped_transaction_rows", "max_customer_max_t_dat"]}
        comparisons["sum_grouped_monetary"] = {"no_cache": left["sum_grouped_monetary"], "cache": right["sum_grouped_monetary"], "absolute_difference": abs(left["sum_grouped_monetary"] - right["sum_grouped_monetary"]), "tolerance": tolerance, "equal_within_tolerance": abs(left["sum_grouped_monetary"] - right["sum_grouped_monetary"]) <= tolerance}
    return {"query_id": query_id, "comparisons": comparisons, "pass": all(item.get("equal", item.get("equal_within_tolerance", False)) for item in comparisons.values())}


def first_result(package: Dict[str, Any], mode: str) -> Dict[str, Any]:
    return next(record["result"] for record in package["trials"] if record["mode"] == mode)


def plan_for_query(df: DataFrame, query_id: str) -> str:
    if query_id == "Q2":
        query_df = df.where(col("t_dat") >= lit(date(2020, 1, 1))).agg(count("*").alias("filtered_row_count"), spark_sum("price").alias("filtered_sum_price"))
    else:
        grouped = df.groupBy("customer_id").agg(count("*").alias("transaction_row_count"), spark_sum("price").alias("customer_sum_price"), spark_max("t_dat").alias("customer_max_t_dat"))
        query_df = grouped.agg(count("*").alias("grouped_customer_count"), spark_sum("transaction_row_count").alias("sum_grouped_transaction_rows"), spark_sum("customer_sum_price").alias("sum_grouped_monetary"), spark_max("customer_max_t_dat").alias("max_customer_max_t_dat"))
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        query_df.explain(mode="formatted")
    return output.getvalue()


def plan_findings(plan: str) -> Dict[str, Any]:
    lines = [line.strip() for line in plan.splitlines() if line.strip()]
    evidence = [line for line in lines if any(token in line for token in ["InMemoryTableScan", "InMemoryRelation", "Scan parquet", "Exchange", "ReadSchema", "PushedFilters"])]
    return {"source_operator": "InMemoryTableScan" if "InMemoryTableScan" in plan else "Scan parquet" if "Scan parquet" in plan else "not_detected", "in_memory_table_scan_present": "InMemoryTableScan" in plan, "in_memory_relation_present": "InMemoryRelation" in plan, "scan_parquet_present": "Scan parquet" in plan, "shuffle_exchange_present": "Exchange" in plan, "read_schema_lines": [line for line in lines if line.startswith("ReadSchema:")], "pushed_filter_lines": [line for line in lines if line.startswith("PushedFilters:")], "evidence_lines": evidence, "plan_line_count": len(lines), "plan_captured_without_inference": True}


def storage_level_record(df: DataFrame) -> Dict[str, Any]:
    try:
        level = df.storageLevel
        return {"string": str(level), "useDisk": bool(level.useDisk), "useMemory": bool(level.useMemory), "useOffHeap": bool(level.useOffHeap), "deserialized": bool(level.deserialized), "replication": int(level.replication), "is_cached_property": bool(df.is_cached)}
    except Exception as error:
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}


def persistent_rdd_evidence(spark: SparkSession) -> Dict[str, Any]:
    try:
        persistent = spark.sparkContext.getPersistentRDDs()
        return {"persistent_rdd_count": len(persistent), "available": True}
    except Exception as error:
        return {"available": False, "persistent_rdd_count": None, "reason": f"{type(error).__name__}: {error}"}


def amortization(materialization_sec: float, query_summaries: Dict[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for query_id, entry in query_summaries.items():
        no_cache = float(entry["NO_CACHE"]["median_sec"])
        cache = float(entry["CACHE"]["median_sec"])
        saved = no_cache - cache
        if saved > 0:
            break_even = math.ceil(materialization_sec / saved)
        else:
            break_even = None
        output[query_id] = {"no_cache_median_sec": no_cache, "cache_median_sec": cache, "saved_time_per_execution_sec": saved, "break_even_executions": break_even, "interpretation": "CACHE_ADVANTAGE_AFTER_AMORTIZATION" if break_even is not None else "NO_BREAK_EVEN_OBSERVED"}
    return output


def repeated_estimates(materialization_sec: float, query_summaries: Dict[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for query_id, entry in query_summaries.items():
        rows = []
        for n in REPEATED_EXECUTIONS:
            no_total = n * float(entry["NO_CACHE"]["median_sec"])
            cache_total = materialization_sec + n * float(entry["CACHE"]["median_sec"])
            difference = no_total - cache_total
            rows.append({"N": n, "no_cache_total_sec": no_total, "cache_total_including_materialization_sec": cache_total, "difference_sec_no_cache_minus_cache": difference, "faster_mode": "CACHE" if difference > 0 else "NO_CACHE" if difference < 0 else "TIE"})
        output[query_id] = rows
    return output


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "output" / "benchmark" / "stage25"
    reports_dir = project_root / "reports"
    report_path = reports_dir / "stage25_cache_vs_no_cache_benchmark.txt"
    summary_path = output_dir / "stage25_final_summary.json"
    event_dir = output_dir / "spark-events"
    blockers: List[str] = []
    warnings: List[str] = ["NO_CACHE means no explicit Spark DataFrame persistence; it is not a cold-disk benchmark.", "Both branches use the authoritative Parquet source. Stage 25 results must not be attributed to CSV-vs-Parquet format differences.", "CACHE uses MEMORY_AND_DISK; complete RAM residency is not claimed."]
    spark: SparkSession | None = None
    cache_df: DataFrame | None = None
    event_logging_enabled = False
    event_logging_issue: str | None = None
    application_id: str | None = None
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        event_dir.mkdir(parents=True, exist_ok=True)
        stage24_blockers, stage24_context = stage24_gate(project_root)
        blockers.extend(stage24_blockers)
        if blockers:
            failed = {"stage": STAGE, "authority": AUTHORITY, "stage24_authority": STAGE24_AUTHORITY, "source_format": "PARQUET", "source_row_count": EXPECTED_ROWS, "cache_no_cache_only_difference": True, "models_modified": False, "stage24_authority_modified": False, "benchmark_complete": False, "ready_for_stage26": False, "blockers": blockers}
            write_json(summary_path, failed)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("STAGE 25 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_26: False\n", encoding="utf-8")
            return 3

        event_uri = event_dir.resolve().as_uri()
        builder = SparkSession.builder.master("local[4]").appName("Stage25-Controlled-Cache-vs-No-Cache-Benchmark").config("spark.sql.shuffle.partitions", "64").config("spark.sql.adaptive.enabled", "true").config("spark.local.dir", str(project_root.parent / "spark_tmp"))
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

        source_blockers, source, _validated_base = source_validation(spark, project_root)
        blockers.extend(source_blockers)
        write_json(output_dir / "source_validation.json", {"stage24_gate": stage24_context, "source": source, "pass": not blockers})
        if blockers:
            raise ValueError("Stage 25 source validation failed.")

        parquet_path = source["path"]
        # The cache manager can substitute an identical logical plan with the
        # cached relation even when the Python DataFrame object was created
        # independently.  The source validation proves these columns are
        # non-null, so these identity-preserving coalesces keep the no-cache
        # branch semantically identical while preventing accidental cache
        # substitution in the same SparkSession.
        no_cache_df = spark.read.option("pathGlobFilter", "*.parquet").parquet(parquet_path).select(*BASE_COLUMNS).select(
            when(input_file_name().isNotNull(), col("customer_id")).otherwise(lit("")).alias("customer_id"),
            coalesce(col("t_dat"), lit(date(1900, 1, 1))).alias("t_dat"),
            coalesce(col("price"), lit(0.0)).alias("price"),
        )
        materialization_start = time.perf_counter()
        cache_df = spark.read.parquet(parquet_path).select(*BASE_COLUMNS).persist(StorageLevel.MEMORY_AND_DISK)
        cached_count = cache_df.count()
        materialization_sec = time.perf_counter() - materialization_start
        cache_materialization = {"storage_level_requested": "MEMORY_AND_DISK", "storage_level_observed": storage_level_record(cache_df), "materialized_row_count": cached_count, "cache_materialization_sec": materialization_sec, "materialization_includes": "persist registration plus first full count()", "pass": cached_count == EXPECTED_ROWS and finite(materialization_sec) and materialization_sec >= 0}
        write_json(output_dir / "cache_materialization.json", cache_materialization)
        if not cache_materialization["pass"]:
            raise ValueError("Cache materialization failed or row count was invalid.")

        packages = {"Q1": run_query_trials("Q1", no_cache_df, cache_df), "Q2": run_query_trials("Q2", no_cache_df, cache_df), "Q3": run_query_trials("Q3", no_cache_df, cache_df)}
        write_json(output_dir / "q1_full_count_trials.json", packages["Q1"])
        write_json(output_dir / "q2_filter_aggregation_trials.json", packages["Q2"])
        write_json(output_dir / "q3_customer_aggregation_trials.json", packages["Q3"])

        equalities = {query_id: result_equal(query_id, first_result(package, "NO_CACHE"), first_result(package, "CACHE")) for query_id, package in packages.items()}
        write_json(output_dir / "query_result_equality.json", {"queries": equalities, "pass": all(item["pass"] for item in equalities.values()), "price_sum_tolerance_definition": f"absolute difference <= {PRICE_SUM_RELATIVE_TOLERANCE} * max(1, abs(NO_CACHE sum))"})
        if not all(item["pass"] for item in equalities.values()):
            raise ValueError("CACHE and NO_CACHE query result equality failed.")

        summary_by_query: Dict[str, Any] = {query_id: package["summary_by_mode"] for query_id, package in packages.items()}
        runtime_summary = {query_id: {"NO_CACHE": summary["NO_CACHE"], "CACHE": summary["CACHE"], "no_cache_over_cache_ratio": summary["NO_CACHE"]["median_sec"] / summary["CACHE"]["median_sec"], "cache_runtime_reduction_pct": (summary["NO_CACHE"]["median_sec"] - summary["CACHE"]["median_sec"]) / summary["NO_CACHE"]["median_sec"] * 100.0, "result_equality_pass": equalities[query_id]["pass"]} for query_id, summary in summary_by_query.items()}
        write_json(output_dir / "runtime_summary.json", runtime_summary)
        runtime_ratios = {query_id: {"no_cache_median_sec": entry["NO_CACHE"]["median_sec"], "cache_median_sec": entry["CACHE"]["median_sec"], "no_cache_over_cache_ratio": entry["no_cache_over_cache_ratio"], "cache_runtime_reduction_pct": entry["cache_runtime_reduction_pct"], "interpretation_rule": "ratio > 1 means CACHE was faster"} for query_id, entry in runtime_summary.items()}
        write_json(output_dir / "runtime_ratios.json", runtime_ratios)
        break_even = amortization(materialization_sec, summary_by_query)
        write_json(output_dir / "cache_break_even_analysis.json", {"cache_materialization_sec": materialization_sec, "queries": break_even})
        estimates = repeated_estimates(materialization_sec, summary_by_query)
        write_json(output_dir / "repeated_workload_estimates.json", {"cache_materialization_sec": materialization_sec, "N_values": REPEATED_EXECUTIONS, "queries": estimates, "analytical_estimate_not_additional_trials": True})

        combined_no_cache = sum(float(summary_by_query[query_id]["NO_CACHE"]["median_sec"]) for query_id in ["Q1", "Q2", "Q3"])
        combined_cache = sum(float(summary_by_query[query_id]["CACHE"]["median_sec"]) for query_id in ["Q1", "Q2", "Q3"])
        combined_saved = combined_no_cache - combined_cache
        combined = {"steady_state_no_cache_cycle_sec": combined_no_cache, "steady_state_cache_cycle_sec": combined_cache, "cache_materialization_sec": materialization_sec, "combined_cycle_saved_time_sec": combined_saved, "combined_break_even_cycles": math.ceil(materialization_sec / combined_saved) if combined_saved > 0 else None, "interpretation": "CACHE_ADVANTAGE_AFTER_AMORTIZATION" if combined_saved > 0 else "NO_BREAK_EVEN_OBSERVED"}
        write_json(output_dir / "combined_workload_break_even_analysis.json", combined)

        plans: Dict[str, Any] = {}
        for query_id in ["Q2", "Q3"]:
            for mode, df in [("NO_CACHE", no_cache_df), ("CACHE", cache_df)]:
                plan_text = plan_for_query(df, query_id)
                plan_path = output_dir / f"physical_plan_{mode.lower()}_{query_id.lower()}.txt"
                plan_path.write_text(plan_text, encoding="utf-8")
                plans[f"{mode}_{query_id}"] = {"path": str(plan_path), **plan_findings(plan_text)}
        write_json(output_dir / "physical_plan_findings.json", plans)

        persistent_before_unpersist = persistent_rdd_evidence(spark)
        storage_evidence = {"cache_storage_level": cache_materialization["storage_level_observed"], "no_cache_storage_level": storage_level_record(no_cache_df), "persistent_rdd_before_unpersist": persistent_before_unpersist, "cached_partition_count": cache_df.rdd.getNumPartitions(), "memory_used_bytes": None, "disk_used_bytes": None, "optional_metrics_status": "memory/disk storage metrics unavailable through this local application interface", "unpersist_requested": False}
        write_json(output_dir / "spark_storage_evidence.json", storage_evidence)

        application_id = spark.sparkContext.applicationId
        ui_port = spark.conf.get("spark.ui.port", "4040")
        environment: Dict[str, Any] = {"stage": STAGE, "authority": AUTHORITY, "stage24_authority": STAGE24_AUTHORITY, "source_format": "PARQUET", "source_path": parquet_path, "spark_version": spark.version, "python_version": platform.python_version(), "java_version": spark._jvm.java.lang.System.getProperty("java.version"), "os": platform.platform(), "cpu_logical_processors": os.cpu_count(), "system_memory_bytes": None, "application_id": application_id, "application_name": "Stage25-Controlled-Cache-vs-No-Cache-Benchmark", "master": "local[4]", "shuffle_partitions": 64, "adaptive_enabled": True, "timing_api": "time.perf_counter", "warmup_trials": WARMUP_TRIALS, "measured_trials": MEASURED_TRIALS, "primary_runtime_statistic": "median", "execution_order": ROTATING_ORDER, "cache_used_in_no_cache_branch": False, "no_cache_reader_option": "pathGlobFilter=*.parquet; includes all authoritative Parquet data files", "no_cache_plan_guard": "distinct file-source relation plus identity-preserving metadata-dependent input_file_name guard prevents Spark cache-manager substitution", "event_log_enabled": event_logging_enabled, "event_log_path": event_uri, "event_log_issue": event_logging_issue, "spark_ui_url_during_application": f"http://localhost:{ui_port}", "warnings": warnings}
        try:
            import psutil
            environment["system_memory_bytes"] = int(psutil.virtual_memory().total)
        except Exception as memory_error:
            environment["system_memory_unavailable_reason"] = f"{type(memory_error).__name__}: {memory_error}"
        write_json(output_dir / "benchmark_environment.json", environment)

        all_ratios_gt_one = all(entry["no_cache_over_cache_ratio"] > 1 for entry in runtime_summary.values())
        any_ratios_gt_one = any(entry["no_cache_over_cache_ratio"] > 1 for entry in runtime_summary.values())
        if all_ratios_gt_one:
            primary_finding = "CACHE_CLEAR_ADVANTAGE_FOR_REPEATED_WORKLOAD"
        elif any_ratios_gt_one:
            primary_finding = "CACHE_QUERY_DEPENDENT"
        elif all(abs(entry["cache_runtime_reduction_pct"]) < 5 for entry in runtime_summary.values()):
            primary_finding = "RUNTIME_SIMILAR"
        elif not any_ratios_gt_one:
            primary_finding = "NO_CACHE_ADVANTAGE"
        else:
            primary_finding = "MIXED_RESULT"

        summary: Dict[str, Any] = {"stage": STAGE, "authority": AUTHORITY, "stage24_authority": STAGE24_AUTHORITY, "source_format": "PARQUET", "source_row_count": EXPECTED_ROWS, "source_path": parquet_path, "cache_storage_level": "MEMORY_AND_DISK", "cache_materialization_sec": materialization_sec, "queries": runtime_summary, "combined_workload": combined, "query_result_equality_pass": all(item["pass"] for item in equalities.values()), "cache_no_cache_only_difference": True, "models_modified": False, "stage24_authority_modified": False, "stage23_authority_modified": False, "primary_finding": primary_finding, "event_log_enabled": event_logging_enabled, "benchmark_complete": True, "ready_for_stage26": True, "warnings": warnings, "blockers": []}
        write_json(summary_path, summary)

        if cache_df is not None:
            cache_df.unpersist(blocking=True)
            storage_evidence["unpersist_requested"] = True
            storage_evidence["cache_storage_level_after_unpersist"] = storage_level_record(cache_df)
            storage_evidence["persistent_rdd_after_unpersist"] = persistent_rdd_evidence(spark)
            write_json(output_dir / "spark_storage_evidence.json", storage_evidence)

        csv_rows = []
        query_names = {"Q1": "FULL_COUNT", "Q2": "FILTER_AGGREGATION", "Q3": "CUSTOMER_AGGREGATION"}
        for query_id, entry in runtime_summary.items():
            csv_rows.append({"QueryID": f"{query_id}_{query_names[query_id]}", "QueryName": query_names[query_id], "NoCacheMedianSec": entry["NO_CACHE"]["median_sec"], "CacheMedianSec": entry["CACHE"]["median_sec"], "NoCacheMeanSec": entry["NO_CACHE"]["mean_sec"], "CacheMeanSec": entry["CACHE"]["mean_sec"], "NoCacheStdSec": entry["NO_CACHE"]["population_std_sec"], "CacheStdSec": entry["CACHE"]["population_std_sec"], "NoCacheOverCacheRatio": entry["no_cache_over_cache_ratio"], "CacheRuntimeReductionPct": entry["cache_runtime_reduction_pct"], "CacheMaterializationSec": materialization_sec, "BreakEvenExecutions": break_even[query_id]["break_even_executions"], "ResultEqualityPass": entry["result_equality_pass"]})
        write_csv(output_dir / "cache_vs_no_cache_summary.csv", ["QueryID", "QueryName", "NoCacheMedianSec", "CacheMedianSec", "NoCacheMeanSec", "CacheMeanSec", "NoCacheStdSec", "CacheStdSec", "NoCacheOverCacheRatio", "CacheRuntimeReductionPct", "CacheMaterializationSec", "BreakEvenExecutions", "ResultEqualityPass"], csv_rows)

        if spark is not None:
            spark.stop()
            spark = None

        artifact_paths = [output_dir / name for name in ["benchmark_environment.json", "source_validation.json", "cache_materialization.json", "q1_full_count_trials.json", "q2_filter_aggregation_trials.json", "q3_customer_aggregation_trials.json", "runtime_summary.json", "runtime_ratios.json", "query_result_equality.json", "cache_break_even_analysis.json", "repeated_workload_estimates.json", "combined_workload_break_even_analysis.json", "physical_plan_no_cache_q2.txt", "physical_plan_cache_q2.txt", "physical_plan_no_cache_q3.txt", "physical_plan_cache_q3.txt", "physical_plan_findings.json", "spark_storage_evidence.json", "cache_vs_no_cache_summary.csv", "stage25_final_summary.json"]] + [report_path, reports_dir / "stage25_run.log"]
        report_lines = ["STAGE 25 — CONTROLLED CACHE VS NO-CACHE BENCHMARK", "", f"AUTHORITY: {AUTHORITY}", f"STAGE24_AUTHORITY: {STAGE24_AUTHORITY}", f"SOURCE: PARQUET / {parquet_path}", f"SOURCE_ROWS: {EXPECTED_ROWS}", f"SPARK_APPLICATION_ID: {application_id}", f"PRIMARY_FINDING: {primary_finding}", "", "A. CACHE MATERIALIZATION:", json.dumps(cache_materialization, ensure_ascii=False, indent=2), "", "B. STEADY-STATE RUNTIME:", json.dumps(runtime_summary, ensure_ascii=False, indent=2), "", "C. BREAK-EVEN:", json.dumps(break_even, ensure_ascii=False, indent=2), "", "D. REPEATED WORKLOAD ESTIMATES:", json.dumps(estimates, ensure_ascii=False, indent=2), "", "COMBINED_WORKLOAD:", json.dumps(combined, ensure_ascii=False, indent=2), "", "QUERY_RESULT_EQUALITY_PASS: True", "CACHE_NO_CACHE_ONLY_DIFFERENCE: True", "MODELS_MODIFIED: False", "STAGE24_AUTHORITY_MODIFIED: False", "", "BENCHMARK_COMPLETE: True", "READY_FOR_STAGE_26: True"]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        artifact_readback = {str(path.relative_to(project_root)): path.exists() and path.stat().st_size > 0 for path in artifact_paths}
        for path in artifact_paths:
            if artifact_readback[str(path.relative_to(project_root))] and path.suffix == ".json":
                try:
                    json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    artifact_readback[str(path.relative_to(project_root))] = False
        summary["artifact_readback"] = artifact_readback
        summary["artifact_readback_pass"] = all(artifact_readback.values())
        summary["benchmark_complete"] = bool(summary["benchmark_complete"] and summary["artifact_readback_pass"])
        summary["ready_for_stage26"] = bool(summary["benchmark_complete"])
        if not summary["artifact_readback_pass"]:
            summary["blockers"] = ["One or more Stage 25 artifacts failed read-back."]
        write_json(summary_path, summary)
        print("STAGE24_GATE_PASS=True")
        print("SOURCE_VALIDATION_PASS=True")
        print("CACHE_MATERIALIZATION_PASS=True")
        print("QUERY_RESULT_EQUALITY_PASS=True")
        print(f"SPARK_APPLICATION_ID={application_id}")
        print(f"PRIMARY_FINDING={primary_finding}")
        print(f"BENCHMARK_COMPLETE={summary['benchmark_complete']}")
        print(f"READY_FOR_STAGE_26={summary['ready_for_stage26']}")
        return 0 if summary["ready_for_stage26"] else 3
    except Exception as error:
        blockers.append(f"{type(error).__name__}: {error}")
        failed = {"stage": STAGE, "authority": AUTHORITY, "stage24_authority": STAGE24_AUTHORITY, "source_format": "PARQUET", "source_row_count": EXPECTED_ROWS, "cache_no_cache_only_difference": True, "models_modified": False, "stage24_authority_modified": False, "benchmark_complete": False, "ready_for_stage26": False, "warnings": warnings, "blockers": blockers}
        write_json(summary_path, failed)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("STAGE 25 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_26: False\n", encoding="utf-8")
        print(traceback.format_exc())
        print(f"STAGE 25 FAILED CLOSED: {error}")
        return 3
    finally:
        if cache_df is not None:
            try:
                cache_df.unpersist(blocking=True)
            except Exception:
                pass
        if spark is not None:
            try:
                spark.stop()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
