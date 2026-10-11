#!/usr/bin/env python3
"""Stage 26: collect live Spark execution evidence for Q2 and Q3.

This is an evidence collector, not a runtime benchmark.  It deliberately does
not persist/cache the workload DataFrames and does not touch model artifacts.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import platform
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    countDistinct,
    max as spark_max,
    min as spark_min,
    sum as spark_sum,
)


STAGE = 26
AUTHORITY = "SPARK_EXECUTION_EVIDENCE_V1"
STAGE25_AUTHORITY = "CACHE_NO_CACHE_BENCHMARK_V1"
EXPECTED_ROWS = 31_788_324
EXPECTED_CUSTOMERS = 1_362_281
EXPECTED_MIN_DATE = "2018-09-20"
EXPECTED_MAX_DATE = "2020-09-22"
EXPECTED_Q2_ROWS = 10_980_132
EXPECTED_Q2_SUM = 298167.136781439
EXPECTED_Q3_SUM = 884645.974050622
EXPECTED_SHUFFLE_PARTITIONS = 64
TOLERANCE_FACTOR = 1e-8

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE25_SUMMARY = PROJECT_ROOT / "output" / "benchmark" / "stage25" / "stage25_final_summary.json"
PARQUET_PATH = PROJECT_ROOT / "output" / "clean" / "transactions"
OUT = PROJECT_ROOT / "output" / "analysis" / "stage26"
REPORT = PROJECT_ROOT / "reports" / "stage26_spark_execution_evidence.txt"
RUN_LOG = PROJECT_ROOT / "reports" / "stage26_run.log"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def enable_windows_java_filesystem_fallback(spark: SparkSession) -> None:
    """Use the same verified Hadoop Windows workaround as Stage 25.

    SparkContext can start without winutils, but Hadoop's Windows native access
    check can fail when the Parquet directory is listed.  Stage 25 verified
    this narrow fallback; it changes no data or Spark workload semantics.
    """
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


def safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def jvalue(value: Any, default: Any = None) -> Any:
    """Read a Py4J object/property without assuming one Spark version."""
    if value is None:
        return default
    try:
        if callable(value):
            return value()
    except Exception:
        pass
    return value


def java_seq_to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        return list(value)
    except Exception:
        try:
            return [value[i] for i in range(value.__len__())]
        except Exception:
            return []


def capture_explain(df: Any) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        df.explain(mode="formatted")
    return buf.getvalue()


def executed_plan_text(df: Any) -> str:
    try:
        return str(df._jdf.queryExecution().executedPlan().toString())
    except Exception as exc:
        return f"UNAVAILABLE: {type(exc).__name__}: {exc}"


def rdd_lineage_text(df: Any) -> str:
    try:
        value = df.rdd.toDebugString()
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
    except Exception as exc:
        return f"UNAVAILABLE: {type(exc).__name__}: {exc}"


def query_execution_flags(df: Any, plan: str) -> dict[str, Any]:
    flags: dict[str, Any] = {
        "contains_scan_parquet": "Scan parquet" in plan or "FileScan parquet" in plan,
        "contains_filter": "Filter" in plan,
        "contains_project": "Project" in plan,
        "contains_hash_aggregate": "HashAggregate" in plan,
        "contains_exchange": "Exchange" in plan,
        "contains_hash_partitioning": "hashpartitioning" in plan.lower(),
        "contains_aqe_shuffle_read": "AQEShuffleRead" in plan,
        "contains_single_partition": "SinglePartition" in plan,
        "read_schema_lines": [line.strip() for line in plan.splitlines() if "ReadSchema" in line],
        "hash_partitioning_matches_64": bool(re.search(r"hashpartitioning\([^\n]*,\s*64\)", plan, re.I)),
    }
    try:
        executed = df._jdf.queryExecution().executedPlan()
        flags["is_final_plan"] = bool(executed.isFinalPlan())
    except Exception as exc:
        flags["is_final_plan"] = None
        flags["is_final_plan_error"] = f"{type(exc).__name__}: {exc}"
    return flags


def fetch_json(url: str) -> dict[str, Any] | list[Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=8) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_endpoint(base: str | None, suffix: str) -> dict[str, Any]:
    if not base:
        return {"available": False, "reason": "Spark UI URL unavailable", "url": None}
    url = base.rstrip("/") + suffix
    try:
        return {"available": True, "url": url, "response": fetch_json(url)}
    except Exception as exc:
        return {"available": False, "url": url, "reason": f"{type(exc).__name__}: {exc}"}


def status_job_info(status_tracker: Any, job_id: int) -> dict[str, Any]:
    try:
        info = status_tracker.getJobInfo(job_id)
        if info is None:
            return {"job_id": job_id, "available": False}
        return {
            "job_id": job_id,
            "available": True,
            "status": str(jvalue(getattr(info, "status", None), None)),
            "stage_ids": [safe_int(x) for x in java_seq_to_list(jvalue(getattr(info, "stageIds", None), []))],
        }
    except Exception as exc:
        return {"job_id": job_id, "available": False, "error": f"{type(exc).__name__}: {exc}"}


def stage_ids_for_group(status_tracker: Any, group_id: str) -> tuple[list[int], list[dict[str, Any]]]:
    try:
        job_ids = [safe_int(x) for x in java_seq_to_list(status_tracker.getJobIdsForGroup(group_id))]
        job_ids = [x for x in job_ids if x is not None]
    except Exception:
        job_ids = []
    infos = [status_job_info(status_tracker, job_id) for job_id in job_ids]
    stage_ids = sorted({stage_id for info in infos for stage_id in (info.get("stage_ids") or []) if stage_id is not None})
    return job_ids, infos, stage_ids


def rest_response(response: Any) -> Any:
    return response.get("response") if isinstance(response, dict) else None


def rest_job_lookup(jobs_raw: dict[str, Any], job_ids: list[int]) -> list[dict[str, Any]]:
    jobs = rest_response(jobs_raw) or []
    return [job for job in jobs if safe_int(job.get("jobId")) in set(job_ids)]


def stage_record_from_rest(stage: dict[str, Any], workload: str, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    record = {
        "workload": workload,
        "stage_id": safe_int(stage.get("stageId")),
        "attempt_id": safe_int(stage.get("attemptId", 0)),
        "name": stage.get("name"),
        "status": stage.get("status"),
        "num_tasks": safe_int(stage.get("numTasks")),
        "num_active_tasks": safe_int(stage.get("numActiveTasks")),
        "num_complete_tasks": safe_int(stage.get("numCompleteTasks")),
        "num_failed_tasks": safe_int(stage.get("numFailedTasks")),
        "executor_run_time": safe_int(stage.get("executorRunTime")),
        "executor_cpu_time": safe_int(stage.get("executorCpuTime")),
        "input_bytes": safe_int(stage.get("inputBytes")),
        "input_records": safe_int(stage.get("inputRecords")),
        "output_bytes": safe_int(stage.get("outputBytes")),
        "output_records": safe_int(stage.get("outputRecords")),
        "shuffle_read_bytes": safe_int(stage.get("shuffleReadBytes")),
        "shuffle_read_records": safe_int(stage.get("shuffleReadRecords")),
        "shuffle_write_bytes": safe_int(stage.get("shuffleWriteBytes")),
        "shuffle_write_records": safe_int(stage.get("shuffleWriteRecords")),
        "memory_bytes_spilled": safe_int(stage.get("memoryBytesSpilled")),
        "disk_bytes_spilled": safe_int(stage.get("diskBytesSpilled")),
        "jvm_gc_time": safe_int(stage.get("jvmGcTime")),
        "result_size": safe_int(stage.get("resultSize")),
        "peak_execution_memory": safe_int(stage.get("peakExecutionMemory")),
    }
    if detail:
        record["submission_time"] = detail.get("submissionTime")
        record["completion_time"] = detail.get("completionTime")
        record["task_summary"] = detail.get("taskSummary")
        record["task_metrics_available"] = bool(detail.get("taskSummary") or detail.get("tasks"))
        record["task_count_from_detail"] = len(detail.get("tasks", {}) or {}) if isinstance(detail.get("tasks"), dict) else None
    else:
        record["task_metrics_available"] = False
    return record


def first_present(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def sum_metric(records: list[dict[str, Any]], key: str) -> int | None:
    values = [safe_int(record.get(key)) for record in records]
    values = [value for value in values if value is not None]
    return sum(values) if values else None


def shuffle_summary(records: list[dict[str, Any]], job_ids: list[int], stage_ids: list[int]) -> dict[str, Any]:
    reads = [record for record in records if (safe_int(record.get("shuffle_read_bytes")) or 0) > 0 or (safe_int(record.get("shuffle_read_records")) or 0) > 0]
    writes = [record for record in records if (safe_int(record.get("shuffle_write_bytes")) or 0) > 0 or (safe_int(record.get("shuffle_write_records")) or 0) > 0]
    return {
        "job_ids": job_ids,
        "stage_ids": stage_ids,
        "shuffle_read_bytes": sum_metric(records, "shuffle_read_bytes"),
        "shuffle_write_bytes": sum_metric(records, "shuffle_write_bytes"),
        "shuffle_read_records": sum_metric(records, "shuffle_read_records"),
        "shuffle_write_records": sum_metric(records, "shuffle_write_records"),
        "stages_with_shuffle_read": [record.get("stage_id") for record in reads],
        "stages_with_shuffle_write": [record.get("stage_id") for record in writes],
    }


def stage_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "Workload", "JobID", "StageID", "AttemptID", "StageName", "Status", "NumTasks",
        "CompletedTasks", "FailedTasks", "ExecutorRunTimeMs", "ExecutorCpuTimeNs",
        "InputBytes", "InputRecords", "ShuffleReadBytes", "ShuffleReadRecords",
        "ShuffleWriteBytes", "ShuffleWriteRecords", "MemoryBytesSpilled", "DiskBytesSpilled", "JvmGcTimeMs",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "Workload": record.get("workload"),
                "JobID": record.get("job_id"),
                "StageID": record.get("stage_id"),
                "AttemptID": record.get("attempt_id"),
                "StageName": record.get("name"),
                "Status": record.get("status"),
                "NumTasks": record.get("num_tasks"),
                "CompletedTasks": record.get("num_complete_tasks"),
                "FailedTasks": record.get("num_failed_tasks"),
                "ExecutorRunTimeMs": record.get("executor_run_time"),
                "ExecutorCpuTimeNs": record.get("executor_cpu_time"),
                "InputBytes": record.get("input_bytes"),
                "InputRecords": record.get("input_records"),
                "ShuffleReadBytes": record.get("shuffle_read_bytes"),
                "ShuffleReadRecords": record.get("shuffle_read_records"),
                "ShuffleWriteBytes": record.get("shuffle_write_bytes"),
                "ShuffleWriteRecords": record.get("shuffle_write_records"),
                "MemoryBytesSpilled": record.get("memory_bytes_spilled"),
                "DiskBytesSpilled": record.get("disk_bytes_spilled"),
                "JvmGcTimeMs": record.get("jvm_gc_time"),
            })


def gate_stage25() -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    try:
        summary = json.loads(STAGE25_SUMMARY.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"Cannot read Stage 25 summary: {type(exc).__name__}: {exc}"], {}
    checks = {
        "stage": (summary.get("stage"), 25),
        "authority": (summary.get("authority"), STAGE25_AUTHORITY),
        "source_row_count": (summary.get("source_row_count"), EXPECTED_ROWS),
        "query_result_equality_pass": (summary.get("query_result_equality_pass"), True),
        "benchmark_complete": (summary.get("benchmark_complete"), True),
        "ready_for_stage26": (summary.get("ready_for_stage26"), True),
        "models_modified": (summary.get("models_modified"), False),
        "stage24_authority_modified": (summary.get("stage24_authority_modified"), False),
    }
    for key, (actual, expected) in checks.items():
        if actual != expected:
            blockers.append(f"Stage25 gate mismatch {key}: actual={actual!r}, expected={expected!r}")
    return blockers, summary


def validate_artifacts(paths: dict[str, Path]) -> dict[str, bool]:
    """Validate current-run artifacts by existence, size and content type."""
    result: dict[str, bool] = {}
    for label, path in paths.items():
        valid = path.is_file() and path.stat().st_size > 0
        if valid and path.suffix.lower() == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                valid = False
        result[label] = valid
    return result


def main() -> int:
    clean_namespace_at_start = not OUT.exists()
    blockers, stage25_summary = gate_stage25()
    warnings: list[str] = []
    spark: SparkSession | None = None
    formal_records: list[dict[str, Any]] = []
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        if not clean_namespace_at_start:
            blockers.append("Stage 26 output namespace was not clean at process start")
        if blockers:
            raise RuntimeError("FAIL CLOSED: " + "; ".join(blockers))
        if not PARQUET_PATH.exists():
            raise RuntimeError(f"Authoritative Parquet path does not exist: {PARQUET_PATH}")

        spark = (
            SparkSession.builder
            .appName("Stage26_Spark_Execution_Evidence")
            .master("local[4]")
            .config("spark.sql.shuffle.partitions", str(EXPECTED_SHUFFLE_PARTITIONS))
            .config("spark.sql.adaptive.enabled", "true")
            .getOrCreate()
        )
        sc = spark.sparkContext
        enable_windows_java_filesystem_fallback(spark)
        status_tracker = sc.statusTracker()
        application_id = sc.applicationId
        ui_url = getattr(sc, "uiWebUrl", None)
        ui_url = jvalue(ui_url, None)
        if ui_url and not str(ui_url).startswith("http"):
            ui_url = "http://" + str(ui_url)
        rest_base = str(ui_url).rstrip("/") + "/api/v1/applications/" + application_id if ui_url else None

        environment = {
            "stage": STAGE,
            "authority": AUTHORITY,
            "spark_version": spark.version,
            "python_version": platform.python_version(),
            "java_version": str(spark._jvm.java.lang.System.getProperty("java.version")),
            "os": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "ram_bytes": None,
            "master": sc.master,
            "spark_application_id": application_id,
            "spark_ui_url": ui_url,
            "spark_ui_available": bool(ui_url),
            "rest_base_url": rest_base,
            "spark_sql_shuffle_partitions": EXPECTED_SHUFFLE_PARTITIONS,
            "spark_sql_adaptive_enabled": True,
            "event_log_enabled": spark.conf.get("spark.eventLog.enabled", "false"),
            "historical_event_log_limitation": "Stage 25 proved Windows event logging unavailable because HADOOP_HOME/hadoop.home.dir and winutils.exe are unavailable; Stage 26 uses live UI REST API and StatusTracker.",
        }
        try:
            import psutil  # type: ignore
            environment["ram_bytes"] = psutil.virtual_memory().total
        except Exception:
            pass
        write_json(OUT / "environment.json", environment)

        # Validation-only actions. They are intentionally not assigned to Q2/Q3 groups.
        source = spark.read.parquet(str(PARQUET_PATH)).select("customer_id", "t_dat", "price")
        validation = source.agg(
            count("*").alias("rows"),
            countDistinct("customer_id").alias("distinct_customers"),
            spark_min("t_dat").alias("min_date"),
            spark_max("t_dat").alias("max_date"),
            spark_sum("price").alias("sum_price"),
        ).collect()[0].asDict()
        validation = {key: (value.isoformat() if isinstance(value, date) else value) for key, value in validation.items()}
        validation["null_customer_id"] = source.filter(col("customer_id").isNull()).limit(1).count()
        validation["null_t_dat"] = source.filter(col("t_dat").isNull()).limit(1).count()
        validation["null_price"] = source.filter(col("price").isNull()).limit(1).count()
        validation["expected_rows"] = EXPECTED_ROWS
        validation["expected_distinct_customers"] = EXPECTED_CUSTOMERS
        validation["pass"] = (
            validation.get("rows") == EXPECTED_ROWS
            and validation.get("distinct_customers") == EXPECTED_CUSTOMERS
            and validation.get("min_date") == EXPECTED_MIN_DATE
            and validation.get("max_date") == EXPECTED_MAX_DATE
            and validation.get("null_customer_id") == 0
            and validation.get("null_t_dat") == 0
            and validation.get("null_price") == 0
        )
        write_json(OUT / "source_validation.json", validation)
        if not validation["pass"]:
            raise RuntimeError(f"Source validation failed: {validation}")

        # Capture live UI raw responses before formal workloads and again after them.
        jobs_raw = fetch_endpoint(rest_base, "/jobs")
        stages_raw = fetch_endpoint(rest_base, "/stages")
        executors_raw = fetch_endpoint(rest_base, "/executors")
        environment_raw = fetch_endpoint(rest_base, "/environment")
        sql_raw = fetch_endpoint(rest_base, "/sql")

        # Q2: filter + scalar aggregation.
        q2 = source.filter(col("t_dat") >= "2020-01-01").agg(
            count("*").alias("filtered_rows"),
            spark_sum("price").alias("sum_price"),
        )
        sc.setJobGroup("STAGE26_Q2", "Stage26 Q2 Filter Aggregation", interruptOnCancel=False)
        q2_before = capture_explain(q2)
        q2_lineage = rdd_lineage_text(q2)
        q2_result = q2.collect()[0].asDict()
        q2_after = capture_explain(q2)
        q2_executed = executed_plan_text(q2)
        q2_job_ids, q2_job_infos, q2_stage_ids = stage_ids_for_group(status_tracker, "STAGE26_Q2")

        # Q3: customer-level wide aggregation followed by a scalar aggregation.
        q3_grouped = source.groupBy("customer_id").agg(
            count("*").alias("customer_transaction_count"),
            spark_sum("price").alias("customer_sum_price"),
            spark_max("t_dat").alias("customer_max_t_dat"),
        )
        q3 = q3_grouped.agg(
            count("*").alias("grouped_customer_count"),
            spark_sum("customer_transaction_count").alias("sum_grouped_transaction_rows"),
            spark_sum("customer_sum_price").alias("sum_grouped_monetary"),
            spark_max("customer_max_t_dat").alias("max_customer_max_t_dat"),
        )
        sc.setJobGroup("STAGE26_Q3", "Stage26 Q3 Customer Aggregation", interruptOnCancel=False)
        q3_before = capture_explain(q3)
        q3_lineage = rdd_lineage_text(q3)
        q3_result = q3.collect()[0].asDict()
        q3_after = capture_explain(q3)
        q3_executed = executed_plan_text(q3)
        q3_job_ids, q3_job_infos, q3_stage_ids = stage_ids_for_group(status_tracker, "STAGE26_Q3")

        for result in (q2_result, q3_result):
            for key, value in list(result.items()):
                if isinstance(value, date):
                    result[key] = value.isoformat()

        q2_result["result_validation_pass"] = (
            q2_result.get("filtered_rows") == EXPECTED_Q2_ROWS
            and abs(float(q2_result.get("sum_price") or 0) - EXPECTED_Q2_SUM) <= TOLERANCE_FACTOR * max(1.0, abs(EXPECTED_Q2_SUM))
        )
        q3_result["result_validation_pass"] = (
            q3_result.get("grouped_customer_count") == EXPECTED_CUSTOMERS
            and q3_result.get("sum_grouped_transaction_rows") == EXPECTED_ROWS
            and q3_result.get("max_customer_max_t_dat") == EXPECTED_MAX_DATE
            and abs(float(q3_result.get("sum_grouped_monetary") or 0) - EXPECTED_Q3_SUM) <= TOLERANCE_FACTOR * max(1.0, abs(EXPECTED_Q3_SUM))
        )
        write_json(OUT / "q2_result.json", q2_result)
        write_json(OUT / "q3_result.json", q3_result)
        write_text(OUT / "q2_physical_plan_before.txt", q2_before)
        write_text(OUT / "q2_physical_plan_after.txt", q2_after)
        write_text(OUT / "q3_physical_plan_before.txt", q3_before)
        write_text(OUT / "q3_physical_plan_after.txt", q3_after)
        write_text(OUT / "q2_executed_plan.txt", q2_executed)
        write_text(OUT / "q3_executed_plan.txt", q3_executed)
        write_text(OUT / "q2_rdd_lineage.txt", "RDD lineage textual evidence\n" + q2_lineage)
        write_text(OUT / "q3_rdd_lineage.txt", "RDD lineage textual evidence\n" + q3_lineage)

        # Refresh REST after Q2/Q3 so status and stage metrics are populated.
        jobs_raw = fetch_endpoint(rest_base, "/jobs")
        stages_raw = fetch_endpoint(rest_base, "/stages")
        executors_raw = fetch_endpoint(rest_base, "/executors")
        environment_raw = fetch_endpoint(rest_base, "/environment")
        sql_raw = fetch_endpoint(rest_base, "/sql")
        write_json(OUT / "spark_ui_jobs_raw.json", jobs_raw)
        write_json(OUT / "spark_ui_stages_raw.json", stages_raw)
        write_json(OUT / "spark_ui_executors_raw.json", executors_raw)
        write_json(OUT / "spark_ui_environment_raw.json", environment_raw)
        write_json(OUT / "spark_ui_sql_raw.json", sql_raw)

        q2_rest_jobs = rest_job_lookup(jobs_raw, q2_job_ids)
        q3_rest_jobs = rest_job_lookup(jobs_raw, q3_job_ids)
        stage_list = rest_response(stages_raw) or []
        relevant_stage_ids = set(q2_stage_ids + q3_stage_ids)
        stage_details: dict[str, Any] = {}
        for stage_id in sorted(relevant_stage_ids):
            candidates = [stage for stage in stage_list if safe_int(stage.get("stageId")) == stage_id]
            for candidate in candidates:
                attempt_id = safe_int(candidate.get("attemptId", 0)) or 0
                detail = fetch_endpoint(rest_base, f"/stages/{stage_id}/{attempt_id}")
                stage_details[f"{stage_id}/{attempt_id}"] = rest_response(detail) if detail.get("available") else detail
                workload = "Q2_FILTER_AGGREGATION" if stage_id in q2_stage_ids else "Q3_CUSTOMER_AGGREGATION"
                detail_response = rest_response(detail) if detail.get("available") else None
                record = stage_record_from_rest(candidate, workload, detail_response)
                record["job_ids"] = q2_job_ids if workload.startswith("Q2") else q3_job_ids
                formal_records.append(record)

        mapping = {
            "Q2_FILTER_AGGREGATION": {"job_group": "STAGE26_Q2", "job_ids": q2_job_ids, "status_tracker_job_info": q2_job_infos, "rest_jobs": q2_rest_jobs, "stage_ids": q2_stage_ids},
            "Q3_CUSTOMER_AGGREGATION": {"job_group": "STAGE26_Q3", "job_ids": q3_job_ids, "status_tracker_job_info": q3_job_infos, "rest_jobs": q3_rest_jobs, "stage_ids": q3_stage_ids},
        }
        write_json(OUT / "workload_job_stage_mapping.json", mapping)
        write_json(OUT / "stage_task_metrics.json", {"records": formal_records, "stage_details": stage_details})
        stage_csv(OUT / "stage_task_metrics.csv", formal_records)

        q2_records = [record for record in formal_records if record["workload"] == "Q2_FILTER_AGGREGATION"]
        q3_records = [record for record in formal_records if record["workload"] == "Q3_CUSTOMER_AGGREGATION"]
        q2_shuffle = shuffle_summary(q2_records, q2_job_ids, q2_stage_ids)
        q3_shuffle = shuffle_summary(q3_records, q3_job_ids, q3_stage_ids)
        write_json(OUT / "shuffle_metrics.json", {"Q2_FILTER_AGGREGATION": q2_shuffle, "Q3_CUSTOMER_AGGREGATION": q3_shuffle})

        q2_flags = query_execution_flags(q2, q2_after + "\n" + q2_executed)
        q3_flags = query_execution_flags(q3, q3_after + "\n" + q3_executed)
        q2_q3_comparison = {
            "Q2_FILTER_AGGREGATION": {
                "physical_structure": "Parquet scan -> filter -> aggregation; actual Exchange presence is recorded from plan flags.",
                "job_count": len(q2_job_ids), "stage_count": len(q2_stage_ids), "task_count": sum_metric(q2_records, "num_tasks"),
                "shuffle": q2_shuffle, "plan_flags": q2_flags,
            },
            "Q3_CUSTOMER_AGGREGATION": {
                "physical_structure": "Plan-derived customer aggregation structure; Exchange/hashpartitioning is recorded only when visible.",
                "job_count": len(q3_job_ids), "stage_count": len(q3_stage_ids), "task_count": sum_metric(q3_records, "num_tasks"),
                "shuffle": q3_shuffle, "plan_flags": q3_flags,
            },
            "comparison_type": "structural only; no performance comparison",
        }
        write_json(OUT / "q2_q3_execution_structure_comparison.json", q2_q3_comparison)

        q3_runtime_shuffle = bool((q3_shuffle.get("shuffle_read_bytes") or 0) > 0 or (q3_shuffle.get("shuffle_write_bytes") or 0) > 0 or q3_shuffle.get("stages_with_shuffle_read") or q3_shuffle.get("stages_with_shuffle_write"))
        q3_plan_shuffle = bool(q3_flags.get("contains_exchange") and q3_flags.get("contains_hash_partitioning"))
        q3_shuffle_confirmed = q3_plan_shuffle or q3_runtime_shuffle
        graph_summary = {
            "Q2_FILTER_AGGREGATION": ["Parquet scan", "Filter", "Aggregation", "Exchange/final aggregation only if visible in actual plan"],
            "Q3_CUSTOMER_AGGREGATION": ["Parquet scan", "Partial hash aggregate by customer", "Hashpartitioning shuffle if visible", "Final customer aggregate", "Scalar aggregate"],
            "evidence_basis": ["physical plan", "job-stage mapping", "stage/task metrics", "RDD lineage"],
            "dag_terminology_note": "Spark execution graph / DAG evidence is reconstructed from job-stage mapping, RDD lineage and physical-plan shuffle boundaries; no UI screenshot is claimed.",
        }
        if not q2_flags.get("contains_scan_parquet") or not q2_result.get("result_validation_pass"):
            blockers.append("Q2 physical plan or result validation incomplete")
        if not q3_result.get("result_validation_pass"):
            blockers.append("Q3 result validation failed")
        if not q2_job_ids or not q3_job_ids or not q2_stage_ids or not q3_stage_ids:
            blockers.append("Formal job IDs or stage IDs could not be mapped")
        if not q3_shuffle_confirmed:
            blockers.append("Q3 shuffle evidence missing from both physical plan and runtime metrics")
        if not ui_url and not (q2_job_ids and q3_job_ids):
            blockers.append("Spark UI unavailable with no usable StatusTracker fallback")

        write_json(OUT / "execution_graph_summary.json", graph_summary)
        rest_available = bool(jobs_raw.get("available") and stages_raw.get("available") and executors_raw.get("available") and environment_raw.get("available"))
        manifest_paths = [
            "environment.json", "source_validation.json", "q2_result.json", "q3_result.json",
            "q2_physical_plan_before.txt", "q2_physical_plan_after.txt", "q3_physical_plan_before.txt", "q3_physical_plan_after.txt",
            "q2_executed_plan.txt", "q3_executed_plan.txt", "q2_rdd_lineage.txt", "q3_rdd_lineage.txt",
            "spark_ui_jobs_raw.json", "spark_ui_stages_raw.json", "spark_ui_executors_raw.json", "spark_ui_environment_raw.json", "spark_ui_sql_raw.json",
            "workload_job_stage_mapping.json", "stage_task_metrics.json", "stage_task_metrics.csv", "shuffle_metrics.json",
            "q2_q3_execution_structure_comparison.json", "execution_graph_summary.json", "spark_ui_evidence_manifest.json", "stage26_final_summary.json",
        ]
        write_json(OUT / "spark_ui_evidence_manifest.json", {"rest_base_url": rest_base, "endpoints": {"jobs": jobs_raw.get("available"), "stages": stages_raw.get("available"), "executors": executors_raw.get("available"), "environment": environment_raw.get("available"), "sql": sql_raw.get("available")}, "raw_artifacts": manifest_paths[:17]})

        # The report and the outer Tee log must exist before the summary gate.
        report_lines = [
            "STAGE 26 SPARK EXECUTION EVIDENCE", f"AUTHORITY={AUTHORITY}", f"SPARK_APPLICATION_ID={application_id}", f"SPARK_UI_URL={ui_url}",
            f"REST_API_AVAILABLE={rest_available}", f"Q2_JOB_IDS={q2_job_ids}", f"Q2_STAGE_IDS={q2_stage_ids}", f"Q3_JOB_IDS={q3_job_ids}", f"Q3_STAGE_IDS={q3_stage_ids}",
            f"Q2_SHUFFLE={json.dumps(q2_shuffle, sort_keys=True)}", f"Q3_SHUFFLE={json.dumps(q3_shuffle, sort_keys=True)}", f"Q3_SHUFFLE_CONFIRMED={q3_shuffle_confirmed}",
            "NO_PERFORMANCE_COMPARISON=True", f"CURRENT_RUN_CLEAN_NAMESPACE={clean_namespace_at_start}",
        ]
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

        non_summary_paths = {
            f"output\\analysis\\stage26\\{name}": OUT / name for name in manifest_paths[:-1]
        }
        non_summary_paths.update({
            "reports\\stage26_spark_execution_evidence.txt": REPORT,
            "reports\\stage26_run.log": RUN_LOG,
        })
        artifact_readback = validate_artifacts(non_summary_paths)
        artifact_readback_pass = all(artifact_readback.values())
        if not artifact_readback_pass:
            blockers.append("Required artifacts failed read-back")

        summary = {
            "stage": STAGE, "authority": AUTHORITY, "stage25_authority": STAGE25_AUTHORITY,
            "source_row_count": EXPECTED_ROWS, "spark_application_id": application_id,
            "spark_ui_available": bool(ui_url), "spark_ui_url": ui_url, "rest_api_available": rest_available,
            "q2": {"result_validation_pass": q2_result["result_validation_pass"], "job_ids": q2_job_ids, "stage_ids": q2_stage_ids, "stage_count": len(q2_stage_ids), "task_count": sum_metric(q2_records, "num_tasks"), "shuffle_read_bytes": q2_shuffle["shuffle_read_bytes"], "shuffle_write_bytes": q2_shuffle["shuffle_write_bytes"]},
            "q3": {"result_validation_pass": q3_result["result_validation_pass"], "job_ids": q3_job_ids, "stage_ids": q3_stage_ids, "stage_count": len(q3_stage_ids), "task_count": sum_metric(q3_records, "num_tasks"), "shuffle_read_bytes": q3_shuffle["shuffle_read_bytes"], "shuffle_write_bytes": q3_shuffle["shuffle_write_bytes"]},
            "q3_shuffle_confirmed": q3_shuffle_confirmed, "physical_plan_evidence_complete": bool(q2_flags and q3_flags),
            "job_stage_mapping_complete": bool(q2_job_ids and q3_job_ids and q2_stage_ids and q3_stage_ids),
            "models_modified": False, "stage24_modified": False, "stage25_modified": False,
            "execution_evidence_complete": False,
            "ready_for_stage27": False,
            "event_log_limitation": environment["historical_event_log_limitation"],
            "warnings": warnings, "blockers": blockers, "artifact_readback": artifact_readback,
            "artifact_readback_pass": artifact_readback_pass,
            "final_summary_self_readback_pass": False,
            "current_run_clean_namespace": clean_namespace_at_start,
        }
        write_json(OUT / "stage26_final_summary.json", summary)

        # Self-readback is deliberately separate from the pre-summary artifact gate.
        try:
            parsed_summary = json.loads((OUT / "stage26_final_summary.json").read_text(encoding="utf-8"))
            final_summary_self_readback_pass = bool(
                (OUT / "stage26_final_summary.json").stat().st_size > 0
                and parsed_summary.get("stage") == STAGE
                and parsed_summary.get("authority") == AUTHORITY
                and parsed_summary.get("artifact_readback_pass") is True
            )
        except Exception:
            final_summary_self_readback_pass = False
        if not final_summary_self_readback_pass:
            blockers.append("Final summary self-readback failed")
        summary["final_summary_self_readback_pass"] = final_summary_self_readback_pass
        summary["execution_evidence_complete"] = not blockers and artifact_readback_pass and final_summary_self_readback_pass
        summary["ready_for_stage27"] = not blockers and artifact_readback_pass and final_summary_self_readback_pass
        summary["blockers"] = blockers
        write_json(OUT / "stage26_final_summary.json", summary)
        # Final reopen verifies the summary after its final write, not a stale prior file.
        try:
            final_summary = json.loads((OUT / "stage26_final_summary.json").read_text(encoding="utf-8"))
            final_summary_self_readback_pass = bool(
                (OUT / "stage26_final_summary.json").stat().st_size > 0
                and final_summary.get("final_summary_self_readback_pass") is True
                and final_summary.get("artifact_readback_pass") is True
            )
        except Exception:
            final_summary_self_readback_pass = False
        summary["final_summary_self_readback_pass"] = final_summary_self_readback_pass
        summary["execution_evidence_complete"] = not blockers and artifact_readback_pass and final_summary_self_readback_pass
        summary["ready_for_stage27"] = not blockers and artifact_readback_pass and final_summary_self_readback_pass
        summary["blockers"] = blockers
        write_json(OUT / "stage26_final_summary.json", summary)
        if not final_summary_self_readback_pass:
            blockers.append("Final summary final reopen failed")
            summary["blockers"] = blockers
            summary["execution_evidence_complete"] = False
            summary["ready_for_stage27"] = False
            write_json(OUT / "stage26_final_summary.json", summary)
        print(f"SPARK_APPLICATION_ID={application_id}")
        print(f"SPARK_UI_URL={ui_url}")
        print(f"REST_API_AVAILABLE={rest_available}")
        print(f"Q2_JOB_IDS={q2_job_ids}")
        print(f"Q3_JOB_IDS={q3_job_ids}")
        print(f"Q3_SHUFFLE_CONFIRMED={q3_shuffle_confirmed}")
        print(f"ARTIFACT_READBACK_PASS={artifact_readback_pass}")
        print(f"FINAL_SUMMARY_SELF_READBACK_PASS={final_summary_self_readback_pass}")
        print(f"EXECUTION_EVIDENCE_COMPLETE={summary['execution_evidence_complete']}")
        print(f"READY_FOR_STAGE_27={summary['ready_for_stage27']}")
        return 0 if summary["ready_for_stage27"] else 3
    except Exception as exc:
        if not blockers:
            blockers.append(f"{type(exc).__name__}: {exc}")
        failed = {
            "stage": STAGE, "authority": AUTHORITY, "stage25_authority": STAGE25_AUTHORITY,
            "source_row_count": EXPECTED_ROWS, "models_modified": False, "stage24_modified": False, "stage25_modified": False,
            "execution_evidence_complete": False, "ready_for_stage27": False, "warnings": warnings, "blockers": blockers,
            "artifact_readback_pass": False, "final_summary_self_readback_pass": False,
            "current_run_clean_namespace": clean_namespace_at_start,
        }
        write_json(OUT / "stage26_final_summary.json", failed)
        write_text(REPORT, "STAGE 26 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_27=False\n")
        print(traceback.format_exc())
        print("EXECUTION_EVIDENCE_COMPLETE=False")
        print("READY_FOR_STAGE_27=False")
        return 3
    finally:
        if spark is not None:
            try:
                spark.stop()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
