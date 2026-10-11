#!/usr/bin/env python3
"""Stage 28 V2: pre-registered FP-Growth support-ladder experiment."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import os
import platform
import sys
import time
from datetime import date, datetime
from functools import reduce
from pathlib import Path
from typing import Any

from pyspark import StorageLevel
from pyspark.ml.fpm import FPGrowth, FPGrowthModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    array_contains,
    array_distinct,
    array_intersect,
    array_join,
    avg,
    col,
    collect_set,
    concat_ws,
    count,
    explode,
    isnan,
    lit,
    max as spark_max,
    min as spark_min,
    percentile_approx,
    sha2,
    size,
    sort_array,
    sum as spark_sum,
    to_date,
    when,
)


STAGE = 28
VERSION = 2
AUTHORITY = "FP_GROWTH_ASSOCIATION_RULES_V2"
STAGE27_AUTHORITY = "SPARK_SQL_ANALYTICAL_LAYER_V1"
V1_AUTHORITY = "FP_GROWTH_ASSOCIATION_RULES_V1"
EXPECTED_ROWS = 31_788_324
EXPECTED_CUSTOMERS = 1_362_281
EXPECTED_BASKETS = 9_080_179
EXPECTED_MIN_DATE = "2018-09-20"
EXPECTED_MAX_DATE = "2020-09-22"
SUPPORT_LADDER = [0.0005, 0.0002, 0.0001, 0.00005]
MIN_CONFIDENCE = 0.20
NUM_PARTITIONS = 64
SAFETY_LIMIT = 2_000_000
RULE_TOLERANCE = 1e-10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE27_SUMMARY = PROJECT_ROOT / "output" / "sql" / "stage27" / "stage27_final_summary.json"
V1_DIR = PROJECT_ROOT / "output" / "fpgrowth" / "stage28"
TRANSACTIONS_PATH = PROJECT_ROOT / "output" / "clean" / "transactions"
ARTICLES_PATH = PROJECT_ROOT / "output" / "clean" / "articles"
RFM_PATH = PROJECT_ROOT / "output" / "features" / "customer_features"
OUT = PROJECT_ROOT / "output" / "fpgrowth" / "stage28_v2"
MODEL_DIR = OUT / "model"
REPORT = PROJECT_ROOT / "reports" / "stage28_v2_fp_growth_association_rules.txt"
RUN_LOG = PROJECT_ROOT / "reports" / "stage28_v2_run.log"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def rows_as_dicts(rows: list[Any]) -> list[dict[str, Any]]:
    return [{key: json_value(value) for key, value in row.asDict().items()} for row in rows]


def capture_explain(df: DataFrame) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        df.explain(mode="formatted")
    return buffer.getvalue()


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


def gate_stage27() -> list[str]:
    required = {
        "stage": 27,
        "authority": STAGE27_AUTHORITY,
        "transaction_row_count": EXPECTED_ROWS,
        "distinct_transaction_customers": EXPECTED_CUSTOMERS,
        "spark_sql_analytical_layer_complete": True,
        "artifact_readback_pass": True,
        "final_summary_self_readback_pass": True,
        "ready_for_stage28": True,
        "blockers": [],
        "models_modified": False,
    }
    try:
        summary = read_json(STAGE27_SUMMARY)
    except Exception as exc:
        return [f"Cannot read Stage 27 summary: {type(exc).__name__}: {exc}"]
    return [f"Stage27 gate mismatch {key}: actual={summary.get(key)!r}, expected={expected!r}" for key, expected in required.items() if summary.get(key) != expected]


def gate_v1() -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    paths = {
        "summary": V1_DIR / "stage28_final_summary.json",
        "basket_profile": V1_DIR / "basket_profile.json",
        "basket_rfm": V1_DIR / "basket_rfm_frequency_validation.json",
        "singleton": V1_DIR / "singleton_support_preflight.json",
        "config": V1_DIR / "fp_growth_config.json",
        "itemsets": V1_DIR / "frequent_itemset_summary.json",
    }
    try:
        for name, path in paths.items():
            if not path.is_file():
                blockers.append(f"Missing V1 artifact: {name}")
        if blockers:
            return blockers, {}
        summary = read_json(paths["summary"])
        required_summary = {
            "stage": 28,
            "authority": V1_AUTHORITY,
            "fp_growth_analysis_complete": False,
            "ready_for_stage29": False,
        }
        blockers.extend(f"V1 gate mismatch {key}: actual={summary.get(key)!r}, expected={expected!r}" for key, expected in required_summary.items() if summary.get(key) != expected)
        if not any("FP-Growth produced no association rules" in str(item) for item in summary.get("blockers", [])):
            blockers.append("V1 blocker evidence missing: FP-Growth produced no association rules")
        profile = read_json(paths["basket_profile"])
        rfm = read_json(paths["basket_rfm"])
        singleton = read_json(paths["singleton"])
        config = read_json(paths["config"])
        itemsets = read_json(paths["itemsets"])
        if profile.get("basket_count") != EXPECTED_BASKETS or profile.get("distinct_customers") != EXPECTED_CUSTOMERS or not profile.get("items_are_unique_within_basket"):
            blockers.append("V1 basket profile semantics failed")
        if not rfm.get("pass") or rfm.get("basket_count") != EXPECTED_BASKETS or rfm.get("frequency_mismatch_count") != 0:
            blockers.append("V1 basket/RFM Frequency semantics failed")
        if not singleton.get("pass") or config.get("minSupport") != 0.001 or config.get("minConfidence") != MIN_CONFIDENCE:
            blockers.append("V1 preflight/config evidence failed")
        if not itemsets.get("pass") or itemsets.get("maximum_itemset_size") != 1:
            blockers.append("V1 negative frequent-itemset evidence failed")
        return blockers, {"summary": summary, "basket_profile": profile, "basket_rfm": rfm, "singleton": singleton, "config": config, "itemsets": itemsets}
    except Exception as exc:
        blockers.append(f"V1 gate read failure: {type(exc).__name__}: {exc}")
        return blockers, {}


def validate_artifacts(paths: dict[str, Path], spark: SparkSession | None) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for label, path in paths.items():
        if path.suffix.lower() == ".parquet":
            valid = path.is_file() or (path.is_dir() and any(path.iterdir()))
            if valid and spark is not None:
                try:
                    valid = spark.read.parquet(str(path)).limit(1).count() >= 0
                except Exception:
                    valid = False
        else:
            valid = path.is_file() and path.stat().st_size > 0
        if valid and path.suffix.lower() == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                valid = False
        if valid and path.suffix.lower() == ".csv":
            try:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    valid = bool(handle.readline().strip()) and any(line.strip() for line in handle)
            except Exception:
                valid = False
        result[label] = valid
    return result


def write_run_log(metadata: dict[str, Any]) -> None:
    """Overwrite the V2 provenance log with metadata from this run only."""
    lines = [
        f"stage = {metadata.get('stage', STAGE)}",
        f"version = {metadata.get('version', VERSION)}",
        f"authority = {metadata.get('authority', AUTHORITY)}",
        f"spark_application_id = {metadata.get('spark_application_id')}",
        f"selected_minSupport = {metadata.get('selected_minSupport')}",
        f"frequent_itemset_count = {metadata.get('frequent_itemset_count')}",
        f"association_rule_count = {metadata.get('association_rule_count')}",
        f"manual_rule_validation_pass = {metadata.get('manual_rule_validation_pass')}",
        f"model_reload_validation_pass = {metadata.get('model_reload_validation_pass')}",
        f"artifact_readback_pass = {metadata.get('artifact_readback_pass')}",
        f"final_summary_self_readback_pass = {metadata.get('final_summary_self_readback_pass')}",
        f"ready_for_stage29 = {metadata.get('ready_for_stage29')}",
    ]
    write_text(RUN_LOG, "\n".join(lines) + "\n")


def validate_run_log(metadata: dict[str, Any]) -> bool:
    if not RUN_LOG.is_file() or RUN_LOG.stat().st_size <= 0:
        return False
    text = RUN_LOG.read_text(encoding="utf-8")
    required = {
        "stage": f"stage = {metadata.get('stage')}",
        "version": f"version = {metadata.get('version')}",
        "authority": f"authority = {metadata.get('authority')}",
        "spark_application_id": f"spark_application_id = {metadata.get('spark_application_id')}",
        "selected_minSupport": f"selected_minSupport = {metadata.get('selected_minSupport')}",
        "frequent_itemset_count": f"frequent_itemset_count = {metadata.get('frequent_itemset_count')}",
        "association_rule_count": f"association_rule_count = {metadata.get('association_rule_count')}",
        "manual_rule_validation_pass": f"manual_rule_validation_pass = {metadata.get('manual_rule_validation_pass')}",
        "model_reload_validation_pass": f"model_reload_validation_pass = {metadata.get('model_reload_validation_pass')}",
        "artifact_readback_pass": f"artifact_readback_pass = {metadata.get('artifact_readback_pass')}",
        "final_summary_self_readback_pass": f"final_summary_self_readback_pass = {metadata.get('final_summary_self_readback_pass')}",
        "ready_for_stage29": f"ready_for_stage29 = {metadata.get('ready_for_stage29')}",
    }
    return all(value in text for value in required.values())


def capped_count(df: DataFrame, limit: int = SAFETY_LIMIT) -> tuple[int, bool]:
    observed = df.limit(limit + 1).count()
    return (limit + 1, True) if observed > limit else (observed, False)


def condition_for_items(items: list[str]) -> Any:
    if not items:
        return lit(True)
    return reduce(lambda left, item: left & array_contains(col("items"), lit(str(item))), items[1:], array_contains(col("items"), lit(str(items[0]))))


def rule_csv_row(rule: dict[str, Any], ranking: str) -> dict[str, Any]:
    return {
        "ranking": ranking,
        "antecedent": "|".join(str(item) for item in rule["antecedent"]),
        "consequent": "|".join(str(item) for item in rule["consequent"]),
        "antecedent_size": rule.get("antecedent_size"),
        "consequent_size": rule.get("consequent_size"),
        "support": rule.get("support"),
        "confidence": rule.get("confidence"),
        "lift": rule.get("lift"),
    }


def main() -> int:
    clean_namespace_at_start = not OUT.exists() and not REPORT.exists() and not RUN_LOG.exists()
    blockers: list[str] = []
    warnings: list[str] = []
    spark: SparkSession | None = None
    application_id: str | None = None
    try:
        blockers.extend(gate_stage27())
        v1_blockers, v1 = gate_v1()
        blockers.extend(v1_blockers)
        if not clean_namespace_at_start:
            blockers.append("Stage 28 V2 namespace was not clean at process start")
        if blockers:
            OUT.mkdir(parents=True, exist_ok=True)
            failed = {"stage": STAGE, "version": VERSION, "authority": AUTHORITY, "stage27_authority": STAGE27_AUTHORITY, "v1_authority": V1_AUTHORITY, "current_run_clean_namespace": clean_namespace_at_start, "fp_growth_analysis_complete": False, "ready_for_stage29": False, "artifact_readback_pass": False, "final_summary_self_readback_pass": False, "models_modified": False, "stage27_modified": False, "v1_modified": False, "warnings": warnings, "blockers": blockers}
            write_json(OUT / "stage28_v2_final_summary.json", failed)
            write_text(REPORT, "STAGE 28 V2 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_29=False\n")
            write_run_log(failed)
            print("STAGE_28_V2_FAILED_CLOSED=True")
            print("READY_FOR_STAGE_29=False")
            return 3

        OUT.mkdir(parents=True, exist_ok=True)
        spark = (
            SparkSession.builder.appName("Stage28_V2_FPGrowth_Threshold_Sensitivity")
            .master("local[4]")
            .config("spark.sql.shuffle.partitions", str(NUM_PARTITIONS))
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.parquet.enableVectorizedReader", "false")
            .config("spark.sql.parquet.columnarReaderBatchSize", "512")
            .getOrCreate()
        )
        enable_windows_java_filesystem_fallback(spark)
        spark.sparkContext.setLogLevel("WARN")
        application_id = spark.sparkContext.applicationId
        ui_url = getattr(spark.sparkContext, "uiWebUrl", None)
        environment = {
            "stage": STAGE,
            "version": VERSION,
            "authority": AUTHORITY,
            "spark_version": spark.version,
            "python_version": platform.python_version(),
            "java_version": spark.sparkContext._jvm.java.lang.System.getProperty("java.version"),
            "os": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "ram_bytes": None,
            "master": spark.sparkContext.master,
            "spark_application_id": application_id,
            "spark_ui_url": ui_url,
            "spark_sql_shuffle_partitions": NUM_PARTITIONS,
            "spark_sql_adaptive_enabled": True,
            "fp_growth_num_partitions": NUM_PARTITIONS,
            "event_log_enabled": spark.conf.get("spark.eventLog.enabled", "false"),
        }
        try:
            import psutil  # type: ignore
            environment["ram_bytes"] = psutil.virtual_memory().total
        except Exception:
            pass
        write_json(OUT / "environment.json", environment)

        transactions = spark.read.parquet(str(TRANSACTIONS_PATH)).select(col("customer_id").cast("string").alias("customer_id"), to_date(col("t_dat")).alias("t_dat"), col("article_id").cast("string").alias("article_id"))
        articles_raw = spark.read.parquet(str(ARTICLES_PATH))
        required_article_columns = {"article_id", "prod_name", "product_group_name"}
        if not required_article_columns.issubset(set(articles_raw.columns)):
            raise RuntimeError(f"Article authority missing required columns: {sorted(required_article_columns - set(articles_raw.columns))}")
        articles = articles_raw.select(col("article_id").cast("string").alias("article_id"), col("prod_name").cast("string").alias("prod_name"), col("product_group_name").cast("string").alias("product_group_name")).dropDuplicates(["article_id"])
        source_rows = transactions.count()
        distinct_customers = transactions.select("customer_id").distinct().count()
        nulls = transactions.agg(spark_sum(when(col("customer_id").isNull(), 1).otherwise(0)).alias("customer"), spark_sum(when(col("t_dat").isNull(), 1).otherwise(0)).alias("date"), spark_sum(when(col("article_id").isNull(), 1).otherwise(0)).alias("article")).collect()[0].asDict()
        source_min = transactions.agg(spark_min("t_dat")).collect()[0][0]
        source_max = transactions.agg(spark_max("t_dat")).collect()[0][0]
        missing_articles = transactions.select("article_id").distinct().join(articles.select("article_id"), "article_id", "left_anti").count()
        source_validation = {
            "row_count": source_rows,
            "distinct_customers": distinct_customers,
            "null_customer_id": int(nulls["customer"] or 0),
            "null_t_dat": int(nulls["date"] or 0),
            "null_article_id": int(nulls["article"] or 0),
            "missing_article_references": missing_articles,
            "min_date": str(source_min),
            "max_date": str(source_max),
            "pass": source_rows == EXPECTED_ROWS and distinct_customers == EXPECTED_CUSTOMERS and str(source_min) == EXPECTED_MIN_DATE and str(source_max) == EXPECTED_MAX_DATE and all(int(nulls[key] or 0) == 0 for key in ["customer", "date", "article"]) and missing_articles == 0,
        }
        write_json(OUT / "source_validation.json", source_validation)
        if not source_validation["pass"]:
            raise RuntimeError("Stage 28 V2 source validation failed")

        baskets = transactions.groupBy("customer_id", "t_dat").agg(sort_array(collect_set("article_id")).alias("items")).withColumn("basket_proxy_id", sha2(concat_ws("||", col("customer_id"), col("t_dat").cast("string")), 256)).select("basket_proxy_id", "customer_id", "t_dat", "items")
        write_text(OUT / "basket_construction_physical_plan.txt", capture_explain(baskets))
        basket_count = baskets.count()
        distinct_basket_customers = baskets.select("customer_id").distinct().count()
        profile = baskets.select(size("items").alias("unique_item_count"))
        profile_agg = profile.agg(spark_min("unique_item_count").alias("min"), spark_max("unique_item_count").alias("max"), avg("unique_item_count").alias("mean"), percentile_approx("unique_item_count", 0.5, 10000).alias("median"), percentile_approx("unique_item_count", 0.75, 10000).alias("p75"), percentile_approx("unique_item_count", 0.9, 10000).alias("p90"), percentile_approx("unique_item_count", 0.95, 10000).alias("p95"), percentile_approx("unique_item_count", 0.99, 10000).alias("p99")).collect()[0].asDict()
        single_count = baskets.filter(size("items") == 1).count()
        multi_count = baskets.filter(size("items") >= 2).count()
        distinct_customer_date_article_rows = transactions.dropDuplicates(["customer_id", "t_dat", "article_id"]).count()
        basket_profile = {
            "basket_count": basket_count,
            "distinct_customers": distinct_basket_customers,
            "min_basket_unique_items": profile_agg["min"],
            "max_basket_unique_items": profile_agg["max"],
            "mean_basket_unique_items": profile_agg["mean"],
            "median": profile_agg["median"],
            "p75": profile_agg["p75"],
            "p90": profile_agg["p90"],
            "p95": profile_agg["p95"],
            "p99": profile_agg["p99"],
            "single_item_basket_count": single_count,
            "multi_item_basket_count": multi_count,
            "multi_item_basket_ratio": multi_count / basket_count if basket_count else 0.0,
            "raw_transaction_item_rows": source_rows,
            "distinct_customer_date_article_rows": distinct_customer_date_article_rows,
            "duplicate_same_article_same_day_rows_collapsed": source_rows - distinct_customer_date_article_rows,
            "basket_proxy_definition": "customer_id + t_dat",
            "items_are_unique_within_basket": True,
        }
        write_json(OUT / "basket_profile.json", basket_profile)
        if basket_count != EXPECTED_BASKETS or distinct_basket_customers != EXPECTED_CUSTOMERS:
            raise RuntimeError("V2 basket universe differs from V1")

        rfm = spark.read.parquet(str(RFM_PATH)).select(col("customer_id").cast("string").alias("customer_id"), col("Frequency").cast("long").alias("Frequency")).dropDuplicates(["customer_id"])
        basket_per_customer = baskets.groupBy("customer_id").agg(count("*").alias("basket_count"))
        compare = basket_per_customer.alias("b").join(rfm.alias("r"), "customer_id", "full")
        compare_stats = compare.agg(count(when(col("b.customer_id").isNull(), 1)).alias("missing_basket"), count(when(col("r.customer_id").isNull(), 1)).alias("missing_rfm"), count(when(col("b.basket_count") != col("r.Frequency"), 1)).alias("mismatch")).collect()[0].asDict()
        frequency_sum = rfm.agg(spark_sum("Frequency")).collect()[0][0]
        basket_rfm = {
            "basket_count": basket_count,
            "sum_stage5_frequency": frequency_sum,
            "missing_basket_customer_count": int(compare_stats["missing_basket"] or 0),
            "missing_rfm_customer_count": int(compare_stats["missing_rfm"] or 0),
            "frequency_mismatch_count": int(compare_stats["mismatch"] or 0),
            "pass": basket_count == int(frequency_sum or 0) and int(compare_stats["missing_basket"] or 0) == 0 and int(compare_stats["missing_rfm"] or 0) == 0 and int(compare_stats["mismatch"] or 0) == 0,
        }
        write_json(OUT / "basket_rfm_frequency_validation.json", basket_rfm)
        if not basket_rfm["pass"]:
            raise RuntimeError("V2 basket/RFM Frequency validation failed")

        baskets = baskets.persist(StorageLevel.MEMORY_AND_DISK)
        materialization_start = time.perf_counter()
        baskets.count()
        materialization_seconds = time.perf_counter() - materialization_start
        storage_level = str(baskets.storageLevel)
        singleton_support = baskets.select(explode("items").alias("article_id")).groupBy("article_id").agg(count("*").alias("basket_frequency")).withColumn("support", col("basket_frequency") / lit(float(basket_count)))
        write_text(OUT / "singleton_support_physical_plan.txt", capture_explain(singleton_support))
        write_json(OUT / "threshold_protocol.json", {
            "selection_protocol": "HIGHEST_PREDEFINED_SUPPORT_MEETING_RULE_GATE",
            "support_ladder": SUPPORT_LADDER,
            "minConfidence": MIN_CONFIDENCE,
            "numPartitions": NUM_PARTITIONS,
            "safety_limit_records": SAFETY_LIMIT,
            "basket_count": basket_count,
            "minimum_support_basket_counts": {str(value): int(math.ceil(basket_count * value)) for value in SUPPORT_LADDER},
            "basket_storage_level": storage_level,
            "materialization_count": 1,
            "materialization_seconds_context_only": materialization_seconds,
            "selection_rule": "First threshold in descending ladder with itemset_size >= 2, at least 10 rules, valid metrics, and completed fit.",
        })

        trials: list[dict[str, Any]] = []
        selected_model: Any = None
        selected_freq: DataFrame | None = None
        selected_rules: DataFrame | None = None
        selected_support: float | None = None
        safety_blocked = False
        for trial_order, support_value in enumerate(SUPPORT_LADDER, start=1):
            min_count = int(math.ceil(basket_count * support_value))
            record: dict[str, Any] = {"TrialOrder": trial_order, "MinSupport": support_value, "MinimumSupportBasketCount": min_count, "FrequentItemsetCount": 0, "SingletonItemsetCount": 0, "PairOrLargerItemsetCount": 0, "MaximumItemsetSize": 0, "AssociationRuleCount": 0, "SelectionGatePass": False, "Selected": False, "FailureReason": ""}
            try:
                estimator = FPGrowth(itemsCol="items", minSupport=support_value, minConfidence=MIN_CONFIDENCE, numPartitions=NUM_PARTITIONS)
                trial_model = estimator.fit(baskets)
                record["fit_status"] = "completed"
                trial_freq = trial_model.freqItemsets.withColumn("itemset_size", size("items")).withColumn("support", col("freq") / lit(float(basket_count)))
                freq_count, freq_limited = capped_count(trial_freq)
                record["FrequentItemsetCount"] = freq_count
                if freq_limited:
                    record["FailureReason"] = "RESOURCE_SAFETY_LIMIT_EXCEEDED"
                    trials.append(record)
                    safety_blocked = True
                    break
                record["SingletonItemsetCount"] = trial_freq.filter(col("itemset_size") == 1).count()
                record["PairOrLargerItemsetCount"] = trial_freq.filter(col("itemset_size") >= 2).count()
                record["MaximumItemsetSize"] = int(trial_freq.agg(spark_max("itemset_size")).collect()[0][0]) if freq_count else 0
                trial_rules = trial_model.associationRules.withColumn("antecedent_size", size("antecedent")).withColumn("consequent_size", size("consequent"))
                rule_count, rule_limited = capped_count(trial_rules)
                record["AssociationRuleCount"] = rule_count
                if rule_limited:
                    record["FailureReason"] = "RESOURCE_SAFETY_LIMIT_EXCEEDED"
                    trials.append(record)
                    safety_blocked = True
                    break
                invalid_rules = trial_rules.filter((size("antecedent") <= 0) | (size("consequent") <= 0) | (size(array_intersect("antecedent", "consequent")) != 0) | col("confidence").isNull() | isnan(col("confidence")) | (col("confidence") < 0) | (col("confidence") > 1) | col("support").isNull() | isnan(col("support")) | (col("support") < 0) | (col("support") > 1) | col("lift").isNull() | isnan(col("lift")) | (col("lift") <= 0)).count()
                metrics_valid = invalid_rules == 0
                record["FailureReason"] = "" if metrics_valid and record["PairOrLargerItemsetCount"] > 0 and rule_count >= 10 else ("NO_PAIR_OR_LARGER_ITEMSETS" if record["PairOrLargerItemsetCount"] == 0 else ("ASSOCIATION_RULE_COUNT_BELOW_10" if rule_count < 10 else "INVALID_RULE_METRICS"))
                record["SelectionGatePass"] = record["PairOrLargerItemsetCount"] > 0 and rule_count >= 10 and metrics_valid
                if record["SelectionGatePass"]:
                    record["Selected"] = True
                    selected_model = trial_model
                    selected_freq = trial_freq
                    selected_rules = trial_rules
                    selected_support = support_value
                    trials.append(record)
                    break
                trials.append(record)
            except Exception as exc:
                record["fit_status"] = "failed"
                record["FailureReason"] = f"{type(exc).__name__}: {exc}"
                trials.append(record)
                raise

        write_json(OUT / "threshold_sensitivity.json", {"trials": trials, "safety_limit": SAFETY_LIMIT, "selection_stopped_on_first_pass": selected_support is not None, "resource_safety_blocked": safety_blocked})
        write_csv(OUT / "threshold_sensitivity.csv", trials, ["TrialOrder", "MinSupport", "MinimumSupportBasketCount", "FrequentItemsetCount", "SingletonItemsetCount", "PairOrLargerItemsetCount", "MaximumItemsetSize", "AssociationRuleCount", "SelectionGatePass", "Selected", "FailureReason"])

        if selected_model is None or selected_freq is None or selected_rules is None or selected_support is None:
            blockers.append("NO_CONFIG_SELECTED: no predefined support threshold yielded at least 10 valid association rules")
            if safety_blocked:
                blockers.append("RESOURCE_SAFETY_LIMIT_EXCEEDED")
            raise RuntimeError("NO_CONFIG_SELECTED")

        selected_freq.write.mode("overwrite").parquet(str(OUT / "selected_frequent_itemsets.parquet"))
        size_distribution = rows_as_dicts(selected_freq.groupBy("itemset_size").count().orderBy("itemset_size").collect())
        selected_freq_count = selected_freq.count()
        selected_max_size = int(selected_freq.agg(spark_max("itemset_size")).collect()[0][0])
        selected_freq_summary = {"frequent_itemset_count": selected_freq_count, "itemsets_size1": int(selected_freq.filter(col("itemset_size") == 1).count()), "itemsets_size2": int(selected_freq.filter(col("itemset_size") == 2).count()), "itemsets_size3_plus": int(selected_freq.filter(col("itemset_size") >= 3).count()), "maximum_itemset_size": selected_max_size, "count_by_itemset_size": size_distribution, "pass": selected_freq_count > 0 and int(selected_freq.filter(col("itemset_size") >= 2).count()) > 0}
        write_json(OUT / "selected_frequent_itemset_summary.json", selected_freq_summary)

        selected_rules = selected_rules.withColumn("antecedent_str", array_join(sort_array("antecedent"), "|")).withColumn("consequent_str", array_join(sort_array("consequent"), "|"))
        selected_rule_count = selected_rules.count()
        selected_rules.write.mode("overwrite").parquet(str(OUT / "selected_association_rules.parquet"))
        selected_rule_summary = {"association_rule_count": selected_rule_count, "invalid_rule_metric_count": selected_rules.filter((size("antecedent") <= 0) | (size("consequent") <= 0) | (size(array_intersect("antecedent", "consequent")) != 0) | col("confidence").isNull() | isnan(col("confidence")) | (col("confidence") < 0) | (col("confidence") > 1) | col("support").isNull() | isnan(col("support")) | (col("support") < 0) | (col("support") > 1) | col("lift").isNull() | isnan(col("lift")) | (col("lift") <= 0)).count(), "pass": selected_rule_count >= 10}
        write_json(OUT / "selected_association_rule_summary.json", selected_rule_summary)
        if not selected_rule_summary["pass"] or selected_freq_summary["pass"] is not True:
            raise RuntimeError("Selected model output validation failed")

        lift_rows = rows_as_dicts(selected_rules.orderBy(col("lift").desc(), col("support").desc(), col("confidence").desc(), col("antecedent_str").asc(), col("consequent_str").asc()).limit(20).collect())
        confidence_rows = rows_as_dicts(selected_rules.orderBy(col("confidence").desc(), col("support").desc(), col("lift").desc(), col("antecedent_str").asc(), col("consequent_str").asc()).limit(20).collect())
        support_rows = rows_as_dicts(selected_rules.orderBy(col("support").desc(), col("confidence").desc(), col("lift").desc(), col("antecedent_str").asc(), col("consequent_str").asc()).limit(20).collect())
        csv_fields = ["ranking", "antecedent", "consequent", "antecedent_size", "consequent_size", "support", "confidence", "lift"]
        write_csv(OUT / "top20_rules_by_lift.csv", [rule_csv_row(row, "lift") for row in lift_rows], csv_fields)
        write_csv(OUT / "top20_rules_by_confidence.csv", [rule_csv_row(row, "confidence") for row in confidence_rows], csv_fields)
        write_csv(OUT / "top20_rules_by_support.csv", [rule_csv_row(row, "support") for row in support_rows], csv_fields)

        article_lookup = {str(row["article_id"]): {"article_id": str(row["article_id"]), "prod_name": row["prod_name"], "product_group_name": row["product_group_name"]} for row in articles.collect()}
        top_itemsets = rows_as_dicts(selected_freq.orderBy(col("freq").desc(), col("support").desc(), array_join(sort_array("items"), "|").asc()).limit(20).collect())
        top_itemset_rows = [{"items": "|".join(str(item) for item in row["items"]), "freq": row["freq"], "support": row["support"], "itemset_size": row["itemset_size"], "products": json.dumps([article_lookup[str(item)] for item in row["items"]], ensure_ascii=False)} for row in top_itemsets]
        write_csv(OUT / "top_frequent_itemsets.csv", top_itemset_rows, ["items", "freq", "support", "itemset_size", "products"])
        presentation = selected_rules.filter((col("antecedent_size") <= 3) & (col("consequent_size") == 1)).orderBy(col("lift").desc(), col("support").desc(), col("confidence").desc(), col("antecedent_str").asc(), col("consequent_str").asc()).limit(60)
        presentation_rows = rows_as_dicts(presentation.collect())
        if not presentation_rows:
            raise RuntimeError("No presentation-friendly selected rules")
        enriched = []
        for row in presentation_rows:
            antecedent = [str(item) for item in row["antecedent"]]
            consequent = [str(item) for item in row["consequent"]]
            if any(item not in article_lookup for item in antecedent + consequent):
                raise RuntimeError("Article enrichment lost authority IDs")
            enriched.append({"antecedent": antecedent, "consequent": consequent, "support": row["support"], "confidence": row["confidence"], "lift": row["lift"], "antecedent_size": row["antecedent_size"], "consequent_size": row["consequent_size"], "antecedent_products": [article_lookup[item] for item in antecedent], "consequent_products": [article_lookup[item] for item in consequent]})
        write_json(OUT / "enriched_top_rules.json", enriched)
        write_csv(OUT / "enriched_top_rules.csv", [{"antecedent": "|".join(row["antecedent"]), "consequent": "|".join(row["consequent"]), "support": row["support"], "confidence": row["confidence"], "lift": row["lift"], "antecedent_products": json.dumps(row["antecedent_products"], ensure_ascii=False), "consequent_products": json.dumps(row["consequent_products"], ensure_ascii=False)} for row in enriched], ["antecedent", "consequent", "support", "confidence", "lift", "antecedent_products", "consequent_products"])

        top10 = lift_rows[:10]
        manual_exprs: list[Any] = []
        for index, row in enumerate(top10):
            a = condition_for_items([str(item) for item in row["antecedent"]])
            b = condition_for_items([str(item) for item in row["consequent"]])
            ab = a & b
            manual_exprs.extend([spark_sum(when(a, 1).otherwise(0)).alias(f"a_{index}"), spark_sum(when(b, 1).otherwise(0)).alias(f"b_{index}"), spark_sum(when(ab, 1).otherwise(0)).alias(f"ab_{index}")])
        manual_workload = baskets.agg(*manual_exprs)
        write_text(OUT / "rule_validation_physical_plan.txt", capture_explain(manual_workload))
        counts = manual_workload.collect()[0].asDict()
        manual_rows = []
        for index, row in enumerate(top10):
            count_a, count_b, count_ab = int(counts[f"a_{index}"] or 0), int(counts[f"b_{index}"] or 0), int(counts[f"ab_{index}"] or 0)
            support_manual = count_ab / basket_count
            confidence_manual = count_ab / count_a if count_a else 0.0
            lift_manual = confidence_manual / (count_b / basket_count) if count_b else 0.0
            ds = abs(support_manual - float(row["support"]))
            dc = abs(confidence_manual - float(row["confidence"]))
            dl = abs(lift_manual - float(row["lift"]))
            manual_rows.append({"antecedent": row["antecedent"], "consequent": row["consequent"], "count_a": count_a, "count_b": count_b, "count_ab": count_ab, "support_spark": row["support"], "confidence_spark": row["confidence"], "lift_spark": row["lift"], "support_manual": support_manual, "confidence_manual": confidence_manual, "lift_manual": lift_manual, "support_difference": ds, "confidence_difference": dc, "lift_difference": dl, "pass": ds <= RULE_TOLERANCE and dc <= RULE_TOLERANCE and dl <= RULE_TOLERANCE})
        manual_validation = {"selected_rule_count": len(manual_rows), "required_rule_count": 10, "tolerance": RULE_TOLERANCE, "rules": manual_rows, "overall_pass": len(manual_rows) == 10 and all(row["pass"] for row in manual_rows)}
        write_json(OUT / "rule_manual_validation.json", manual_validation)
        if not manual_validation["overall_pass"]:
            raise RuntimeError("Manual top-10 rule validation failed")

        selected_model.write().overwrite().save(str(MODEL_DIR))
        reloaded = FPGrowthModel.load(str(MODEL_DIR))
        reloaded_freq = reloaded.freqItemsets.count()
        reloaded_rules = reloaded.associationRules.count()
        reload_validation = {"selected_minSupport_before_reload": selected_support, "selected_minSupport_after_reload": float(reloaded.getOrDefault(reloaded.minSupport)), "minConfidence_after_reload": float(reloaded.getOrDefault(reloaded.minConfidence)), "numPartitions_after_reload": int(reloaded.getOrDefault(reloaded.numPartitions)), "frequent_itemset_count_before_reload": selected_freq_count, "frequent_itemset_count_after_reload": reloaded_freq, "association_rule_count_before_reload": selected_rule_count, "association_rule_count_after_reload": reloaded_rules, "model_reload_validation_pass": abs(float(reloaded.getOrDefault(reloaded.minSupport)) - selected_support) <= 1e-12 and abs(float(reloaded.getOrDefault(reloaded.minConfidence)) - MIN_CONFIDENCE) <= 1e-12 and int(reloaded.getOrDefault(reloaded.numPartitions)) == NUM_PARTITIONS and reloaded_freq == selected_freq_count and reloaded_rules == selected_rule_count}
        write_json(OUT / "model_reload_validation.json", reload_validation)
        if not reload_validation["model_reload_validation_pass"]:
            raise RuntimeError("Selected model reload validation failed")

        physical_texts = {name: (OUT / name).read_text(encoding="utf-8") for name in ["basket_construction_physical_plan.txt", "singleton_support_physical_plan.txt", "rule_validation_physical_plan.txt"]}
        physical_findings = {name.replace("_physical_plan.txt", ""): {"contains_scan_parquet": "Scan parquet" in text, "contains_hash_aggregate": "HashAggregate" in text or "ObjectHashAggregate" in text, "contains_exchange": "Exchange" in text, "contains_sort": "Sort" in text, "non_empty": bool(text.strip())} for name, text in physical_texts.items()}
        physical_complete = all(item["non_empty"] and item["contains_scan_parquet"] for item in physical_findings.values())
        if not physical_complete:
            raise RuntimeError("Selected physical-plan evidence incomplete")

        findings = {"v1_negative_result": "At minSupport 0.001, no article pair occurred in at least 0.1% of the complete customer-day basket universe, so no association rule was produced.", "v2_selection": "Stage 28 V2 evaluated the pre-registered descending support ladder and selected the highest threshold satisfying the predefined rule-generation gate.", "highest_lift_rule": enriched[0] if enriched else {}, "highest_confidence_rule": rule_csv_row(confidence_rows[0], "confidence") if confidence_rows else {}, "highest_support_rule": rule_csv_row(support_rows[0], "support") if support_rows else {}, "frequent_itemsets": top_itemset_rows[:5], "safe_interpretation": "Rules describe product co-occurrence within customer-day basket proxies and do not establish causality or classification accuracy."}
        write_json(OUT / "fp_growth_findings.json", findings)

        higher_results = [trial for trial in trials if selected_support is not None and trial["MinSupport"] > selected_support]
        selected_config = {"selection_protocol": "HIGHEST_PREDEFINED_SUPPORT_MEETING_RULE_GATE", "support_ladder": SUPPORT_LADDER, "selected_minSupport": selected_support, "minConfidence": MIN_CONFIDENCE, "numPartitions": NUM_PARTITIONS, "basket_count": basket_count, "minimum_support_basket_count": int(math.ceil(basket_count * selected_support)), "selection_reason": "First threshold in the pre-registered descending ladder satisfying pair-or-larger itemsets, at least 10 valid rules, and completed fit.", "higher_threshold_results": higher_results}
        write_json(OUT / "selected_fp_growth_config.json", selected_config)

        table_lines = ["Support Threshold Sensitivity", "| minSupport | min count | itemsets | pairs+ | rules | selected |", "|---:|---:|---:|---:|---:|:---:|"]
        table_lines.extend(f"| {row['MinSupport']} | {row['MinimumSupportBasketCount']} | {row['FrequentItemsetCount']} | {row['PairOrLargerItemsetCount']} | {row['AssociationRuleCount']} | {row['Selected']} |" for row in trials)
        report = ["STAGE 28 V2 FP-GROWTH THRESHOLD SENSITIVITY", "AUTHORITY=FP_GROWTH_ASSOCIATION_RULES_V2", "V1 remains FP_GROWTH_ASSOCIATION_RULES_V1 and is preserved as negative evidence.", "BASKET_LIMITATION=Giỏ hàng được xấp xỉ bằng tập sản phẩm phát sinh trên cùng một ngày mua của cùng một khách hàng do dữ liệu không cung cấp mã hóa đơn/order_id.", *table_lines, json.dumps(source_validation, ensure_ascii=False, default=str), json.dumps(basket_profile, ensure_ascii=False, default=str), json.dumps(basket_rfm, ensure_ascii=False, default=str), json.dumps(selected_config, ensure_ascii=False, default=str), json.dumps(selected_freq_summary, ensure_ascii=False, default=str), json.dumps(selected_rule_summary, ensure_ascii=False, default=str), json.dumps(manual_validation, ensure_ascii=False, default=str), json.dumps(reload_validation, ensure_ascii=False, default=str), json.dumps(physical_findings, ensure_ascii=False, default=str), json.dumps(findings, ensure_ascii=False, default=str), "V1_NOT_MODIFIED=True", "STAGE27_NOT_MODIFIED=True", "NO_STAGE29_EXECUTION=True"]
        write_text(REPORT, "\n".join(report) + "\n")
        write_run_log({
            "stage": STAGE,
            "version": VERSION,
            "authority": AUTHORITY,
            "spark_application_id": application_id,
            "selected_minSupport": selected_support,
            "frequent_itemset_count": selected_freq_count,
            "association_rule_count": selected_rule_count,
            "manual_rule_validation_pass": manual_validation["overall_pass"],
            "model_reload_validation_pass": reload_validation["model_reload_validation_pass"],
            "artifact_readback_pass": False,
            "final_summary_self_readback_pass": False,
            "ready_for_stage29": False,
        })

        required_files = {
            "environment.json": OUT / "environment.json", "source_validation.json": OUT / "source_validation.json", "basket_profile.json": OUT / "basket_profile.json", "basket_rfm_frequency_validation.json": OUT / "basket_rfm_frequency_validation.json", "threshold_protocol.json": OUT / "threshold_protocol.json", "threshold_sensitivity.json": OUT / "threshold_sensitivity.json", "threshold_sensitivity.csv": OUT / "threshold_sensitivity.csv", "selected_fp_growth_config.json": OUT / "selected_fp_growth_config.json", "selected_frequent_itemsets.parquet": OUT / "selected_frequent_itemsets.parquet", "selected_frequent_itemset_summary.json": OUT / "selected_frequent_itemset_summary.json", "selected_association_rules.parquet": OUT / "selected_association_rules.parquet", "selected_association_rule_summary.json": OUT / "selected_association_rule_summary.json", "top20_rules_by_lift.csv": OUT / "top20_rules_by_lift.csv", "top20_rules_by_confidence.csv": OUT / "top20_rules_by_confidence.csv", "top20_rules_by_support.csv": OUT / "top20_rules_by_support.csv", "top_frequent_itemsets.csv": OUT / "top_frequent_itemsets.csv", "enriched_top_rules.json": OUT / "enriched_top_rules.json", "enriched_top_rules.csv": OUT / "enriched_top_rules.csv", "rule_manual_validation.json": OUT / "rule_manual_validation.json", "model_reload_validation.json": OUT / "model_reload_validation.json", "basket_construction_physical_plan.txt": OUT / "basket_construction_physical_plan.txt", "singleton_support_physical_plan.txt": OUT / "singleton_support_physical_plan.txt", "rule_validation_physical_plan.txt": OUT / "rule_validation_physical_plan.txt", "fp_growth_findings.json": OUT / "fp_growth_findings.json", "report": REPORT, "run_log": RUN_LOG,
        }
        readback = validate_artifacts(required_files, spark)
        readback_pass = all(readback.values()) and MODEL_DIR.is_dir() and any(MODEL_DIR.iterdir())
        if not readback_pass:
            blockers.append("Stage 28 V2 artifact read-back failed")

        summary_path = OUT / "stage28_v2_final_summary.json"
        summary = {"stage": STAGE, "version": VERSION, "authority": AUTHORITY, "stage27_authority": STAGE27_AUTHORITY, "v1_authority": V1_AUTHORITY, "v1_result": "ZERO_ASSOCIATION_RULES_AT_SUPPORT_0.001", "basket_proxy_definition": "customer_id + t_dat", "basket_count": basket_count, "basket_rfm_frequency_equivalence": basket_rfm["pass"], "threshold_selection_protocol": "HIGHEST_PREDEFINED_SUPPORT_MEETING_RULE_GATE", "support_ladder": SUPPORT_LADDER, "selected_minSupport": selected_support, "minConfidence": MIN_CONFIDENCE, "numPartitions": NUM_PARTITIONS, "frequent_itemset_count": selected_freq_count, "pair_or_larger_itemset_count": selected_freq_summary["itemsets_size2"] + selected_freq_summary["itemsets_size3_plus"], "association_rule_count": selected_rule_count, "manual_rule_validation_pass": manual_validation["overall_pass"], "model_reload_validation_pass": reload_validation["model_reload_validation_pass"], "physical_plan_evidence_complete": physical_complete, "models_modified": False, "stage27_modified": False, "v1_modified": False, "current_run_clean_namespace": clean_namespace_at_start, "artifact_readback": readback, "artifact_readback_pass": readback_pass, "final_summary_self_readback_pass": False, "fp_growth_analysis_complete": False, "ready_for_stage29": False, "warnings": warnings, "blockers": blockers, "spark_application_id": application_id, "source_paths": {"transactions": str(TRANSACTIONS_PATH), "articles": str(ARTICLES_PATH), "rfm_frequency": str(RFM_PATH), "model": str(MODEL_DIR)}}
        write_json(summary_path, summary)
        try:
            initial = read_json(summary_path)
            self_readback = initial.get("stage") == STAGE and initial.get("version") == VERSION and initial.get("authority") == AUTHORITY and initial.get("artifact_readback_pass") is True
        except Exception:
            self_readback = False
        if not self_readback:
            blockers.append("Final Stage 28 V2 summary self-readback failed")
        summary["final_summary_self_readback_pass"] = self_readback
        summary["artifact_readback"]["stage28_v2_final_summary.json"] = True
        summary["fp_growth_analysis_complete"] = not blockers and readback_pass and self_readback
        summary["ready_for_stage29"] = summary["fp_growth_analysis_complete"]
        summary["blockers"] = blockers
        write_json(summary_path, summary)
        try:
            final = read_json(summary_path)
            final_self = final.get("stage") == STAGE and final.get("version") == VERSION and final.get("authority") == AUTHORITY and final.get("artifact_readback_pass") is True and final.get("final_summary_self_readback_pass") is True
        except Exception:
            final_self = False
        if not final_self:
            blockers.append("Final Stage 28 V2 summary final reopen failed")
            summary["final_summary_self_readback_pass"] = False
            summary["fp_growth_analysis_complete"] = False
            summary["ready_for_stage29"] = False
            summary["blockers"] = blockers
            write_json(summary_path, summary)

        # Final provenance pass: overwrite the log after the final summary is
        # finalized, then verify that the current log contains this run's
        # summary fields.  This prevents a stale log from satisfying readback.
        write_run_log(summary)
        run_log_provenance_pass = validate_run_log(summary)
        summary["run_log_provenance_pass"] = run_log_provenance_pass
        if not run_log_provenance_pass:
            blockers.append("Current Stage 28 V2 run-log provenance validation failed")
            summary["artifact_readback_pass"] = False
            summary["final_summary_self_readback_pass"] = False
            summary["fp_growth_analysis_complete"] = False
            summary["ready_for_stage29"] = False
            summary["blockers"] = blockers
            write_json(summary_path, summary)
            write_run_log(summary)
        write_json(summary_path, summary)
        try:
            final_provenance_summary = read_json(summary_path)
            final_provenance_self_readback = final_provenance_summary.get("stage") == STAGE and final_provenance_summary.get("version") == VERSION and final_provenance_summary.get("authority") == AUTHORITY and final_provenance_summary.get("current_run_clean_namespace") is True and final_provenance_summary.get("artifact_readback_pass") is True and final_provenance_summary.get("final_summary_self_readback_pass") is True and final_provenance_summary.get("blockers") == []
        except Exception:
            final_provenance_self_readback = False
        if not final_provenance_self_readback:
            blockers.append("Final Stage 28 V2 provenance summary reopen failed")
            summary["artifact_readback_pass"] = False
            summary["final_summary_self_readback_pass"] = False
            summary["fp_growth_analysis_complete"] = False
            summary["ready_for_stage29"] = False
            summary["blockers"] = blockers
            write_json(summary_path, summary)
            write_run_log(summary)

        baskets.unpersist(blocking=False)
        print(f"SPARK_APPLICATION_ID={application_id}")
        print(f"SELECTED_MINSUPPORT={selected_support}")
        print(f"FREQUENT_ITEMSET_COUNT={selected_freq_count}")
        print(f"ASSOCIATION_RULE_COUNT={selected_rule_count}")
        print(f"MANUAL_RULE_VALIDATION_PASS={manual_validation['overall_pass']}")
        print(f"MODEL_RELOAD_VALIDATION_PASS={reload_validation['model_reload_validation_pass']}")
        print(f"ARTIFACT_READBACK_PASS={summary['artifact_readback_pass']}")
        print(f"FINAL_SUMMARY_SELF_READBACK_PASS={summary['final_summary_self_readback_pass']}")
        print(f"FP_GROWTH_ANALYSIS_COMPLETE={summary['fp_growth_analysis_complete']}")
        print(f"READY_FOR_STAGE_29={summary['ready_for_stage29']}")
        return 0 if summary["ready_for_stage29"] else 3
    except Exception as exc:
        blockers.append(f"{type(exc).__name__}: {exc}")
        OUT.mkdir(parents=True, exist_ok=True)
        failed = {"stage": STAGE, "version": VERSION, "authority": AUTHORITY, "stage27_authority": STAGE27_AUTHORITY, "v1_authority": V1_AUTHORITY, "v1_result": "ZERO_ASSOCIATION_RULES_AT_SUPPORT_0.001", "basket_proxy_definition": "customer_id + t_dat", "models_modified": False, "stage27_modified": False, "v1_modified": False, "current_run_clean_namespace": clean_namespace_at_start, "artifact_readback_pass": False, "final_summary_self_readback_pass": False, "fp_growth_analysis_complete": False, "ready_for_stage29": False, "warnings": warnings, "blockers": blockers}
        write_json(OUT / "stage28_v2_final_summary.json", failed)
        write_text(REPORT, "STAGE 28 V2 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_29=False\n")
        write_run_log(failed)
        print(f"STAGE_28_V2_FAILED={type(exc).__name__}: {exc}")
        print("FP_GROWTH_ANALYSIS_COMPLETE=False")
        print("READY_FOR_STAGE_29=False")
        return 3
    finally:
        if spark is not None:
            try:
                spark.stop()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
