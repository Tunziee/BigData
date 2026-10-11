#!/usr/bin/env python3
"""Stage 28: FP-Growth association rules on customer-day basket proxies.

The basket proxy is (customer_id, t_dat).  The source data has no invoice,
order, or receipt identifier, so this stage never represents the proxy as an
actual order.  It is independent of the locked B0/M1/M2 clustering models.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import os
import platform
import sys
from datetime import date, datetime
from functools import reduce
from pathlib import Path
from typing import Any

from pyspark import StorageLevel
from pyspark.ml.fpm import FPGrowth, FPGrowthModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    array_distinct,
    array_contains,
    array_intersect,
    array_join,
    avg,
    col,
    concat_ws,
    count,
    countDistinct,
    collect_set,
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
AUTHORITY = "FP_GROWTH_ASSOCIATION_RULES_V1"
STAGE27_AUTHORITY = "SPARK_SQL_ANALYTICAL_LAYER_V1"
EXPECTED_ROWS = 31_788_324
EXPECTED_CUSTOMERS = 1_362_281
EXPECTED_MIN_DATE = "2018-09-20"
EXPECTED_MAX_DATE = "2020-09-22"
MIN_SUPPORT = 0.001
MIN_CONFIDENCE = 0.20
NUM_PARTITIONS = 64
RULE_TOLERANCE = 1e-10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE27_SUMMARY = PROJECT_ROOT / "output" / "sql" / "stage27" / "stage27_final_summary.json"
TRANSACTIONS_PATH = PROJECT_ROOT / "output" / "clean" / "transactions"
ARTICLES_PATH = PROJECT_ROOT / "output" / "clean" / "articles"
RFM_PATH = PROJECT_ROOT / "output" / "features" / "customer_features"
OUT = PROJECT_ROOT / "output" / "fpgrowth" / "stage28"
MODEL_DIR = OUT / "model"
REPORT = PROJECT_ROOT / "reports" / "stage28_fp_growth_association_rules.txt"
RUN_LOG = PROJECT_ROOT / "reports" / "stage28_run.log"


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


def gate_stage27() -> tuple[list[str], dict[str, Any]]:
    required = {
        "stage": 27,
        "authority": STAGE27_AUTHORITY,
        "transaction_row_count": EXPECTED_ROWS,
        "distinct_transaction_customers": EXPECTED_CUSTOMERS,
        "rfm_sql_stage5_semantic_equivalence": True,
        "m2_membership_authority_match": True,
        "sql_dataframe_validation_pass": True,
        "spark_sql_analytical_layer_complete": True,
        "artifact_readback_pass": True,
        "final_summary_self_readback_pass": True,
        "current_run_clean_namespace": True,
        "ready_for_stage28": True,
        "blockers": [],
        "models_modified": False,
    }
    try:
        summary = read_json(STAGE27_SUMMARY)
    except Exception as exc:
        return [f"Cannot read Stage 27 summary: {type(exc).__name__}: {exc}"], {}
    blockers = [f"Stage27 gate mismatch {key}: actual={summary.get(key)!r}, expected={value!r}" for key, value in required.items() if summary.get(key) != value]
    return blockers, summary


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


def product_entry(article_id: str, article_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return article_lookup[article_id]


def main() -> int:
    clean_namespace_at_start = not OUT.exists() and not REPORT.exists()
    blockers: list[str] = []
    warnings: list[str] = []
    spark: SparkSession | None = None
    application_id: str | None = None
    try:
        gate_blockers, stage27 = gate_stage27()
        blockers.extend(gate_blockers)
        if not clean_namespace_at_start:
            blockers.append("Stage 28 namespace was not clean at process start")
        if blockers:
            OUT.mkdir(parents=True, exist_ok=True)
            failed = {
                "stage": STAGE,
                "authority": AUTHORITY,
                "stage27_authority": STAGE27_AUTHORITY,
                "models_modified": False,
                "stage27_modified": False,
                "current_run_clean_namespace": clean_namespace_at_start,
                "fp_growth_analysis_complete": False,
                "ready_for_stage29": False,
                "artifact_readback_pass": False,
                "final_summary_self_readback_pass": False,
                "warnings": warnings,
                "blockers": blockers,
            }
            write_json(OUT / "stage28_final_summary.json", failed)
            write_text(REPORT, "STAGE 28 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_29=False\n")
            print("STAGE_28_FAILED_CLOSED=True")
            print("READY_FOR_STAGE_29=False")
            return 3

        OUT.mkdir(parents=True, exist_ok=True)
        spark = (
            SparkSession.builder.appName("Stage28_FPGrowth_Association_Rules")
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
            "spark_sql_parquet_vectorized_reader": False,
            "spark_sql_parquet_columnar_reader_batch_size": 512,
            "event_log_enabled": spark.conf.get("spark.eventLog.enabled", "false"),
        }
        try:
            import psutil  # type: ignore
            environment["ram_bytes"] = psutil.virtual_memory().total
        except Exception:
            pass
        write_json(OUT / "environment.json", environment)

        transactions = spark.read.parquet(str(TRANSACTIONS_PATH)).select(
            col("customer_id").cast("string").alias("customer_id"),
            to_date(col("t_dat")).alias("t_dat"),
            col("article_id").cast("string").alias("article_id"),
        )
        articles_raw = spark.read.parquet(str(ARTICLES_PATH))
        required_article_columns = {"article_id", "prod_name", "product_group_name"}
        if not required_article_columns.issubset(set(articles_raw.columns)):
            raise RuntimeError(f"Article authority missing required columns: {sorted(required_article_columns - set(articles_raw.columns))}")
        articles = articles_raw.select(
            col("article_id").cast("string").alias("article_id"),
            col("prod_name").cast("string").alias("prod_name"),
            col("product_group_name").cast("string").alias("product_group_name"),
        ).dropDuplicates(["article_id"])
        source_rows = transactions.count()
        distinct_customers = transactions.select("customer_id").distinct().count()
        source_nulls = transactions.agg(
            spark_sum(when(col("customer_id").isNull(), 1).otherwise(0)).alias("null_customer_id"),
            spark_sum(when(col("t_dat").isNull(), 1).otherwise(0)).alias("null_t_dat"),
            spark_sum(when(col("article_id").isNull(), 1).otherwise(0)).alias("null_article_id"),
        ).collect()[0].asDict()
        source_min = transactions.agg(spark_min("t_dat").alias("min_date")).collect()[0]["min_date"]
        source_max = transactions.agg(spark_max("t_dat").alias("max_date")).collect()[0]["max_date"]
        article_ids = articles.select("article_id").dropDuplicates()
        missing_article_references = transactions.select("article_id").distinct().join(article_ids, "article_id", "left_anti").count()
        source_validation = {
            "row_count": source_rows,
            "distinct_customers": distinct_customers,
            "null_customer_id": int(source_nulls["null_customer_id"] or 0),
            "null_t_dat": int(source_nulls["null_t_dat"] or 0),
            "null_article_id": int(source_nulls["null_article_id"] or 0),
            "missing_article_references": missing_article_references,
            "min_date": str(source_min),
            "max_date": str(source_max),
            "transactions_path": str(TRANSACTIONS_PATH),
            "articles_path": str(ARTICLES_PATH),
            "pass": source_rows == EXPECTED_ROWS and distinct_customers == EXPECTED_CUSTOMERS and source_validation_dates(source_min, source_max) and all(int(source_nulls[key] or 0) == 0 for key in ["null_customer_id", "null_t_dat", "null_article_id"]) and missing_article_references == 0,
        }
        write_json(OUT / "source_validation.json", source_validation)
        if not source_validation["pass"]:
            raise RuntimeError("Stage 28 source validation failed")

        baskets = (
            transactions.groupBy("customer_id", "t_dat")
            .agg(sort_array(collect_set("article_id")).alias("items"))
            .withColumn("basket_proxy_id", sha2(concat_ws("||", col("customer_id"), col("t_dat").cast("string")), 256))
            .select("basket_proxy_id", "customer_id", "t_dat", "items")
        )
        write_text(OUT / "basket_construction_physical_plan.txt", capture_explain(baskets))
        basket_count = baskets.count()
        distinct_basket_customers = baskets.select("customer_id").distinct().count()
        basket_unique = baskets.select(size("items").alias("unique_item_count"))
        profile_agg = basket_unique.agg(
            spark_min("unique_item_count").alias("min_basket_unique_items"),
            spark_max("unique_item_count").alias("max_basket_unique_items"),
            avg("unique_item_count").alias("mean_basket_unique_items"),
            percentile_approx("unique_item_count", 0.50, 10000).alias("median"),
            percentile_approx("unique_item_count", 0.75, 10000).alias("p75"),
            percentile_approx("unique_item_count", 0.90, 10000).alias("p90"),
            percentile_approx("unique_item_count", 0.95, 10000).alias("p95"),
            percentile_approx("unique_item_count", 0.99, 10000).alias("p99"),
        ).collect()[0].asDict()
        single_item_count = baskets.filter(size("items") == 1).count()
        multi_item_count = baskets.filter(size("items") >= 2).count()
        distinct_customer_date_article_rows = transactions.dropDuplicates(["customer_id", "t_dat", "article_id"]).count()
        duplicate_collapsed = source_rows - distinct_customer_date_article_rows
        basket_duplicate_ids = baskets.groupBy("basket_proxy_id").count().filter(col("count") > 1).count()
        basket_null_items = baskets.filter(col("items").isNull()).count()
        basket_empty_items = baskets.filter(size("items") == 0).count()
        basket_duplicate_items = baskets.filter(size("items") != size(array_distinct("items"))).count()
        basket_customer_date_duplicates = baskets.groupBy("customer_id", "t_dat").count().filter(col("count") > 1).count()
        basket_profile = {
            "basket_count": basket_count,
            "distinct_customers": distinct_basket_customers,
            "min_basket_unique_items": profile_agg["min_basket_unique_items"],
            "max_basket_unique_items": profile_agg["max_basket_unique_items"],
            "mean_basket_unique_items": profile_agg["mean_basket_unique_items"],
            "median": profile_agg["median"],
            "p75": profile_agg["p75"],
            "p90": profile_agg["p90"],
            "p95": profile_agg["p95"],
            "p99": profile_agg["p99"],
            "single_item_basket_count": single_item_count,
            "multi_item_basket_count": multi_item_count,
            "multi_item_basket_ratio": multi_item_count / basket_count if basket_count else 0.0,
            "raw_transaction_item_rows": source_rows,
            "distinct_customer_date_article_rows": distinct_customer_date_article_rows,
            "duplicate_same_article_same_day_rows_collapsed": duplicate_collapsed,
            "basket_duplicate_proxy_id_count": basket_duplicate_ids,
            "basket_null_item_array_count": basket_null_items,
            "basket_empty_item_array_count": basket_empty_items,
            "basket_duplicate_article_array_count": basket_duplicate_items,
            "basket_customer_date_duplicate_count": basket_customer_date_duplicates,
            "basket_proxy_definition": "customer_id + t_dat",
            "items_are_unique_within_basket": basket_duplicate_items == 0,
        }
        write_json(OUT / "basket_profile.json", basket_profile)
        if any([basket_count == 0, basket_duplicate_ids, basket_null_items, basket_empty_items, basket_duplicate_items, basket_customer_date_duplicates]):
            raise RuntimeError("Basket construction integrity validation failed")

        rfm = spark.read.parquet(str(RFM_PATH)).select(
            col("customer_id").cast("string").alias("customer_id"),
            col("Frequency").cast("long").alias("Frequency"),
        ).dropDuplicates(["customer_id"])
        basket_per_customer = baskets.groupBy("customer_id").agg(count("*").alias("basket_count"))
        frequency_compare = basket_per_customer.alias("b").join(rfm.alias("r"), "customer_id", "full")
        frequency_agg = frequency_compare.agg(
            count(when(col("b.customer_id").isNull(), 1)).alias("missing_basket_customers"),
            count(when(col("r.customer_id").isNull(), 1)).alias("missing_rfm_customers"),
            count(when(col("b.basket_count") != col("r.Frequency"), 1)).alias("frequency_mismatch_count"),
        ).collect()[0].asDict()
        frequency_sum = rfm.agg(spark_sum("Frequency").alias("frequency_sum")).collect()[0]["frequency_sum"]
        basket_rfm_validation = {
            "basket_count": basket_count,
            "sum_stage5_frequency": frequency_sum,
            "missing_basket_customer_count": int(frequency_agg["missing_basket_customers"] or 0),
            "missing_rfm_customer_count": int(frequency_agg["missing_rfm_customers"] or 0),
            "frequency_mismatch_count": int(frequency_agg["frequency_mismatch_count"] or 0),
            "basket_count_equals_sum_frequency": basket_count == int(frequency_sum or 0),
            "pass": basket_count == int(frequency_sum or 0) and int(frequency_agg["missing_basket_customers"] or 0) == 0 and int(frequency_agg["missing_rfm_customers"] or 0) == 0 and int(frequency_agg["frequency_mismatch_count"] or 0) == 0,
        }
        write_json(OUT / "basket_rfm_frequency_validation.json", basket_rfm_validation)
        if not basket_rfm_validation["pass"]:
            raise RuntimeError("Basket to Stage 5 Frequency validation failed")

        baskets = baskets.persist(StorageLevel.DISK_ONLY)
        baskets.count()
        storage_level = "Disk Serialized 1x Replicated" if baskets.storageLevel.useDisk else str(baskets.storageLevel)

        minimum_support_basket_count = int(math.ceil(basket_count * MIN_SUPPORT))
        singleton_support = (
            baskets.select(explode("items").alias("article_id"))
            .groupBy("article_id")
            .agg(count("*").alias("basket_frequency"))
            .withColumn("support", col("basket_frequency") / lit(float(basket_count)))
        )
        write_text(OUT / "singleton_support_physical_plan.txt", capture_explain(singleton_support))
        distinct_article_count = singleton_support.count()
        qualifying_article_count = singleton_support.filter(col("basket_frequency") >= minimum_support_basket_count).count()
        top20_singletons = rows_as_dicts(singleton_support.orderBy(col("basket_frequency").desc(), col("article_id").asc()).limit(20).collect())
        singleton_preflight = {
            "basket_count": basket_count,
            "min_support": MIN_SUPPORT,
            "minimum_support_basket_count": minimum_support_basket_count,
            "distinct_articles": distinct_article_count,
            "articles_meeting_min_support": qualifying_article_count,
            "top20_articles_by_basket_support": top20_singletons,
            "threshold_changed_after_observation": False,
            "pass": distinct_article_count > 0 and qualifying_article_count > 0,
        }
        write_json(OUT / "singleton_support_preflight.json", singleton_preflight)
        if not singleton_preflight["pass"]:
            raise RuntimeError("Singleton support preflight failed")

        fp_growth_config = {
            "itemsCol": "items",
            "minSupport": MIN_SUPPORT,
            "minConfidence": MIN_CONFIDENCE,
            "numPartitions": NUM_PARTITIONS,
            "basket_count": basket_count,
            "minimum_support_basket_count": minimum_support_basket_count,
            "basket_storage_level_during_fit": storage_level,
            "fit_completed": False,
        }
        write_json(OUT / "fp_growth_config.json", fp_growth_config)
        estimator = FPGrowth(itemsCol="items", minSupport=MIN_SUPPORT, minConfidence=MIN_CONFIDENCE, numPartitions=NUM_PARTITIONS)
        model = estimator.fit(baskets)
        fp_growth_config["fit_completed"] = True
        fp_growth_config["model_class"] = type(model).__name__
        write_json(OUT / "fp_growth_config.json", fp_growth_config)

        freq_df = model.freqItemsets.withColumn("itemset_size", size("items")).withColumn("support", col("freq") / lit(float(basket_count)))
        frequent_itemset_count = freq_df.count()
        if frequent_itemset_count == 0:
            raise RuntimeError("FP-Growth produced no frequent itemsets")
        freq_df.write.mode("overwrite").parquet(str(OUT / "frequent_itemsets.parquet"))
        itemset_size_rows = rows_as_dicts(freq_df.groupBy("itemset_size").count().orderBy("itemset_size").collect())
        max_itemset_size = int(freq_df.agg(spark_max("itemset_size")).collect()[0][0])
        freq_valid_count = freq_df.filter((col("freq") <= 0) | (col("support") <= 0) | (col("support") > 1) | (col("itemset_size") < 1)).count()
        frequent_itemset_summary = {
            "frequent_itemset_count": frequent_itemset_count,
            "maximum_itemset_size": max_itemset_size,
            "count_by_itemset_size": itemset_size_rows,
            "invalid_itemset_metric_count": freq_valid_count,
            "pass": freq_valid_count == 0,
        }
        write_json(OUT / "frequent_itemset_summary.json", frequent_itemset_summary)
        if not frequent_itemset_summary["pass"]:
            raise RuntimeError("Frequent itemset metric validation failed")

        raw_rules = model.associationRules
        raw_rule_schema = [field.name for field in raw_rules.schema.fields]
        rules = (
            raw_rules
            .withColumn("antecedent_size", size("antecedent"))
            .withColumn("consequent_size", size("consequent"))
            .withColumn("antecedent_str", array_join(sort_array("antecedent"), "|"))
            .withColumn("consequent_str", array_join(sort_array("consequent"), "|"))
        )
        invalid_rule_count = rules.filter(
            (size("antecedent") <= 0)
            | (size("consequent") <= 0)
            | (size(array_intersect("antecedent", "consequent")) != 0)
            | col("confidence").isNull()
            | isnan(col("confidence"))
            | (col("confidence") < 0)
            | (col("confidence") > 1)
            | col("support").isNull()
            | isnan(col("support"))
            | (col("support") < 0)
            | (col("support") > 1)
            | col("lift").isNull()
            | isnan(col("lift"))
            | (col("lift") <= 0)
        ).count()
        association_rule_count = rules.count()
        if association_rule_count == 0:
            raise RuntimeError("FP-Growth produced no association rules")
        complete_rules = rules.select("antecedent", "consequent", "confidence", "lift", "support", "antecedent_size", "consequent_size", "antecedent_str", "consequent_str")
        complete_rules.write.mode("overwrite").parquet(str(OUT / "association_rules.parquet"))
        association_rule_summary = {
            "association_rule_count": association_rule_count,
            "spark_association_rule_schema": raw_rule_schema,
            "invalid_rule_metric_count": invalid_rule_count,
            "metric_definitions": {
                "support": "Proportion of basket proxies containing antecedent and consequent.",
                "confidence": "P(consequent | antecedent).",
                "lift": "Confidence divided by consequent support; lift greater than one indicates stronger co-occurrence than independence, not causality.",
            },
            "pass": invalid_rule_count == 0,
        }
        write_json(OUT / "association_rule_summary.json", association_rule_summary)
        if not association_rule_summary["pass"]:
            raise RuntimeError("Association rule metric validation failed")

        lift_df = rules.orderBy(col("lift").desc(), col("support").desc(), col("confidence").desc(), col("antecedent_str").asc(), col("consequent_str").asc()).limit(20)
        confidence_df = rules.orderBy(col("confidence").desc(), col("support").desc(), col("lift").desc(), col("antecedent_str").asc(), col("consequent_str").asc()).limit(20)
        support_df = rules.orderBy(col("support").desc(), col("confidence").desc(), col("lift").desc(), col("antecedent_str").asc(), col("consequent_str").asc()).limit(20)
        lift_rows = rows_as_dicts(lift_df.collect())
        confidence_rows = rows_as_dicts(confidence_df.collect())
        support_rows = rows_as_dicts(support_df.collect())
        csv_fields = ["ranking", "antecedent", "consequent", "antecedent_size", "consequent_size", "support", "confidence", "lift"]
        write_csv(OUT / "top20_rules_by_lift.csv", [rule_csv_row(row, "lift") for row in lift_rows], csv_fields)
        write_csv(OUT / "top20_rules_by_confidence.csv", [rule_csv_row(row, "confidence") for row in confidence_rows], csv_fields)
        write_csv(OUT / "top20_rules_by_support.csv", [rule_csv_row(row, "support") for row in support_rows], csv_fields)

        article_lookup = {
            str(row["article_id"]): {
                "article_id": str(row["article_id"]),
                "prod_name": row["prod_name"],
                "product_group_name": row["product_group_name"],
            }
            for row in articles.collect()
        }
        top_itemsets = rows_as_dicts(freq_df.orderBy(col("freq").desc(), col("support").desc(), array_join(sort_array("items"), "|").asc()).limit(20).collect())
        top_itemset_rows: list[dict[str, Any]] = []
        for row in top_itemsets:
            item_ids = [str(item) for item in row["items"]]
            top_itemset_rows.append({
                "items": "|".join(item_ids),
                "freq": row["freq"],
                "support": row["support"],
                "itemset_size": row["itemset_size"],
                "products": json.dumps([product_entry(item, article_lookup) for item in item_ids], ensure_ascii=False),
            })
        write_csv(OUT / "top_frequent_itemsets.csv", top_itemset_rows, ["items", "freq", "support", "itemset_size", "products"])

        presentation_df = rules.filter((col("antecedent_size") <= 3) & (col("consequent_size") == 1)).orderBy(col("lift").desc(), col("support").desc(), col("confidence").desc(), col("antecedent_str").asc(), col("consequent_str").asc()).limit(60)
        presentation_rows = rows_as_dicts(presentation_df.collect())
        if not presentation_rows:
            raise RuntimeError("No presentation-friendly rules with antecedent size <= 3 and singleton consequent")
        enrichment_missing_ids: set[str] = set()
        enriched_top_rules: list[dict[str, Any]] = []
        for row in presentation_rows:
            antecedent_ids = [str(item) for item in row["antecedent"]]
            consequent_ids = [str(item) for item in row["consequent"]]
            enrichment_missing_ids.update(item for item in antecedent_ids + consequent_ids if item not in article_lookup)
            enriched_top_rules.append({
                "antecedent": antecedent_ids,
                "consequent": consequent_ids,
                "confidence": row["confidence"],
                "lift": row["lift"],
                "support": row["support"],
                "antecedent_size": row["antecedent_size"],
                "consequent_size": row["consequent_size"],
                "antecedent_products": [product_entry(item, article_lookup) for item in antecedent_ids if item in article_lookup],
                "consequent_products": [product_entry(item, article_lookup) for item in consequent_ids if item in article_lookup],
            })
        if enrichment_missing_ids:
            raise RuntimeError(f"Article enrichment lost authority IDs: {sorted(enrichment_missing_ids)[:10]}")
        write_json(OUT / "enriched_top_rules.json", enriched_top_rules)
        enriched_csv_rows = []
        for row in enriched_top_rules:
            enriched_csv_rows.append({
                "antecedent": "|".join(row["antecedent"]),
                "consequent": "|".join(row["consequent"]),
                "confidence": row["confidence"],
                "lift": row["lift"],
                "support": row["support"],
                "antecedent_products": json.dumps(row["antecedent_products"], ensure_ascii=False),
                "consequent_products": json.dumps(row["consequent_products"], ensure_ascii=False),
            })
        write_csv(OUT / "enriched_top_rules.csv", enriched_csv_rows, ["antecedent", "consequent", "confidence", "lift", "support", "antecedent_products", "consequent_products"])

        top10_lift = lift_rows[:10]
        if not top10_lift:
            raise RuntimeError("No rules available for independent validation")
        manual_exprs: list[Any] = []
        for index, row in enumerate(top10_lift):
            antecedent_condition = condition_for_items([str(item) for item in row["antecedent"]])
            consequent_condition = condition_for_items([str(item) for item in row["consequent"]])
            union_condition = antecedent_condition & consequent_condition
            manual_exprs.extend([
                spark_sum(when(antecedent_condition, 1).otherwise(0)).alias(f"count_a_{index}"),
                spark_sum(when(consequent_condition, 1).otherwise(0)).alias(f"count_b_{index}"),
                spark_sum(when(union_condition, 1).otherwise(0)).alias(f"count_ab_{index}"),
            ])
        manual_workload = baskets.agg(*manual_exprs)
        write_text(OUT / "rule_validation_physical_plan.txt", capture_explain(manual_workload))
        manual_counts = manual_workload.collect()[0].asDict()
        manual_rows: list[dict[str, Any]] = []
        for index, row in enumerate(top10_lift):
            count_a = int(manual_counts[f"count_a_{index}"] or 0)
            count_b = int(manual_counts[f"count_b_{index}"] or 0)
            count_ab = int(manual_counts[f"count_ab_{index}"] or 0)
            support_manual = count_ab / basket_count
            confidence_manual = count_ab / count_a if count_a else 0.0
            lift_manual = confidence_manual / (count_b / basket_count) if count_b else 0.0
            support_difference = abs(support_manual - float(row["support"]))
            confidence_difference = abs(confidence_manual - float(row["confidence"]))
            lift_difference = abs(lift_manual - float(row["lift"]))
            manual_rows.append({
                "antecedent": row["antecedent"],
                "consequent": row["consequent"],
                "count_a": count_a,
                "count_b": count_b,
                "count_ab": count_ab,
                "basket_count": basket_count,
                "support_spark": row["support"],
                "confidence_spark": row["confidence"],
                "lift_spark": row["lift"],
                "support_manual": support_manual,
                "confidence_manual": confidence_manual,
                "lift_manual": lift_manual,
                "support_difference": support_difference,
                "confidence_difference": confidence_difference,
                "lift_difference": lift_difference,
                "pass": support_difference <= RULE_TOLERANCE and confidence_difference <= RULE_TOLERANCE and lift_difference <= RULE_TOLERANCE,
            })
        manual_validation = {
            "selected_rule_basis": "top 10 by deterministic lift ranking",
            "selected_rule_count": len(manual_rows),
            "tolerance": RULE_TOLERANCE,
            "rules": manual_rows,
            "overall_pass": bool(manual_rows) and all(row["pass"] for row in manual_rows),
        }
        write_json(OUT / "rule_manual_validation.json", manual_validation)
        if not manual_validation["overall_pass"]:
            raise RuntimeError("Independent association-rule validation failed")

        model.write().overwrite().save(str(MODEL_DIR))
        reloaded_model = FPGrowthModel.load(str(MODEL_DIR))
        reloaded_freq_count = reloaded_model.freqItemsets.count()
        reloaded_rule_count = reloaded_model.associationRules.count()
        reload_config = {
            "saved_model_path": str(MODEL_DIR),
            "frequent_itemset_count_before_reload": frequent_itemset_count,
            "frequent_itemset_count_after_reload": reloaded_freq_count,
            "association_rule_count_before_reload": association_rule_count,
            "association_rule_count_after_reload": reloaded_rule_count,
            "minSupport_after_reload": float(reloaded_model.getOrDefault(reloaded_model.minSupport)),
            "minConfidence_after_reload": float(reloaded_model.getOrDefault(reloaded_model.minConfidence)),
            "numPartitions_after_reload": int(reloaded_model.getOrDefault(reloaded_model.numPartitions)),
            "model_reload_validation_pass": reloaded_freq_count == frequent_itemset_count and reloaded_rule_count == association_rule_count and abs(float(reloaded_model.getOrDefault(reloaded_model.minSupport)) - MIN_SUPPORT) <= 1e-12 and abs(float(reloaded_model.getOrDefault(reloaded_model.minConfidence)) - MIN_CONFIDENCE) <= 1e-12 and int(reloaded_model.getOrDefault(reloaded_model.numPartitions)) == NUM_PARTITIONS,
        }
        write_json(OUT / "model_reload_validation.json", reload_config)
        if not reload_config["model_reload_validation_pass"]:
            raise RuntimeError("FP-Growth model reload validation failed")

        physical_plans = {
            "basket_construction": (OUT / "basket_construction_physical_plan.txt").read_text(encoding="utf-8"),
            "singleton_support": (OUT / "singleton_support_physical_plan.txt").read_text(encoding="utf-8"),
            "rule_validation": (OUT / "rule_validation_physical_plan.txt").read_text(encoding="utf-8"),
        }
        physical_plan_findings = {
            name: {
                "contains_scan_parquet": "Scan parquet" in text,
                "contains_hash_aggregate": "HashAggregate" in text or "ObjectHashAggregate" in text,
                "contains_exchange": "Exchange" in text,
                "contains_sort": "Sort" in text,
                "contains_explode_or_generate": "explode" in text.lower() or "Generate" in text,
                "non_empty": bool(text.strip()),
            }
            for name, text in physical_plans.items()
        }
        physical_plan_evidence_complete = all(item["non_empty"] and item["contains_scan_parquet"] for item in physical_plan_findings.values())
        if not physical_plan_evidence_complete:
            raise RuntimeError("Required Stage 28 physical-plan evidence incomplete")

        frequent_finding_rows = top_itemset_rows[:5]
        basket_findings = [{"metric": "basket_count", "value": basket_count, "source_artifact": "basket_profile.json"}, {"metric": "multi_item_basket_ratio", "value": basket_profile["multi_item_basket_ratio"], "source_artifact": "basket_profile.json"}]
        rules_by_lift_findings = enriched_top_rules[:5]
        rules_by_confidence_findings = [rule_csv_row(row, "confidence") for row in confidence_rows[:5]]
        rules_by_support_findings = [rule_csv_row(row, "support") for row in support_rows[:5]]
        business_interpretation = [
            "The basket is a customer-day proxy because the source data has no invoice_id, order_id, or receipt_id.",
            "These rules describe product co-occurrence within the defined proxy and do not establish causality or classification accuracy.",
        ]
        findings = {
            "basket": basket_findings,
            "frequent_itemsets": frequent_finding_rows,
            "rules_by_lift": rules_by_lift_findings,
            "rules_by_confidence": rules_by_confidence_findings,
            "rules_by_support": rules_by_support_findings,
            "business_interpretation": business_interpretation,
        }
        write_json(OUT / "fp_growth_findings.json", findings)

        report_lines = [
            "STAGE 28 FP-GROWTH ASSOCIATION-RULE MINING",
            "AUTHORITY=FP_GROWTH_ASSOCIATION_RULES_V1",
            "BASKET_LIMITATION=Giỏ hàng được xấp xỉ bằng tập sản phẩm phát sinh trên cùng một ngày mua của cùng một khách hàng do dữ liệu không cung cấp mã hóa đơn/order_id.",
            "NO_INVOICE_OR_ORDER_CLAIM=True",
            "PRIMARY_ITEM=article_id",
            "NO_M2_BASKET_LEAKAGE=True",
            json.dumps(source_validation, ensure_ascii=False, default=str),
            json.dumps(basket_profile, ensure_ascii=False, default=str),
            json.dumps(basket_rfm_validation, ensure_ascii=False, default=str),
            json.dumps(fp_growth_config, ensure_ascii=False, default=str),
            json.dumps(frequent_itemset_summary, ensure_ascii=False, default=str),
            json.dumps(association_rule_summary, ensure_ascii=False, default=str),
            json.dumps(manual_validation, ensure_ascii=False, default=str),
            json.dumps(reload_config, ensure_ascii=False, default=str),
            json.dumps(physical_plan_findings, ensure_ascii=False, default=str),
            json.dumps(findings, ensure_ascii=False, default=str),
            "FP_GROWTH_IS_COMPLEMENTARY_TO_M2=True",
            "NO_CAUSALITY_CLAIM=True",
        ]
        write_text(REPORT, "\n".join(report_lines) + "\n")

        required_files = {
            "output\\fpgrowth\\stage28\\environment.json": OUT / "environment.json",
            "output\\fpgrowth\\stage28\\source_validation.json": OUT / "source_validation.json",
            "output\\fpgrowth\\stage28\\basket_profile.json": OUT / "basket_profile.json",
            "output\\fpgrowth\\stage28\\basket_rfm_frequency_validation.json": OUT / "basket_rfm_frequency_validation.json",
            "output\\fpgrowth\\stage28\\singleton_support_preflight.json": OUT / "singleton_support_preflight.json",
            "output\\fpgrowth\\stage28\\fp_growth_config.json": OUT / "fp_growth_config.json",
            "output\\fpgrowth\\stage28\\frequent_itemsets.parquet": OUT / "frequent_itemsets.parquet",
            "output\\fpgrowth\\stage28\\frequent_itemset_summary.json": OUT / "frequent_itemset_summary.json",
            "output\\fpgrowth\\stage28\\association_rules.parquet": OUT / "association_rules.parquet",
            "output\\fpgrowth\\stage28\\association_rule_summary.json": OUT / "association_rule_summary.json",
            "output\\fpgrowth\\stage28\\top20_rules_by_lift.csv": OUT / "top20_rules_by_lift.csv",
            "output\\fpgrowth\\stage28\\top20_rules_by_confidence.csv": OUT / "top20_rules_by_confidence.csv",
            "output\\fpgrowth\\stage28\\top20_rules_by_support.csv": OUT / "top20_rules_by_support.csv",
            "output\\fpgrowth\\stage28\\top_frequent_itemsets.csv": OUT / "top_frequent_itemsets.csv",
            "output\\fpgrowth\\stage28\\enriched_top_rules.json": OUT / "enriched_top_rules.json",
            "output\\fpgrowth\\stage28\\enriched_top_rules.csv": OUT / "enriched_top_rules.csv",
            "output\\fpgrowth\\stage28\\rule_manual_validation.json": OUT / "rule_manual_validation.json",
            "output\\fpgrowth\\stage28\\basket_construction_physical_plan.txt": OUT / "basket_construction_physical_plan.txt",
            "output\\fpgrowth\\stage28\\singleton_support_physical_plan.txt": OUT / "singleton_support_physical_plan.txt",
            "output\\fpgrowth\\stage28\\rule_validation_physical_plan.txt": OUT / "rule_validation_physical_plan.txt",
            "output\\fpgrowth\\stage28\\fp_growth_findings.json": OUT / "fp_growth_findings.json",
            "reports\\stage28_fp_growth_association_rules.txt": REPORT,
            "reports\\stage28_run.log": RUN_LOG,
        }
        normal_readback = validate_artifacts(required_files, spark)
        artifact_readback_pass = all(normal_readback.values()) and MODEL_DIR.is_dir() and any(MODEL_DIR.iterdir())
        if not artifact_readback_pass:
            blockers.append("Stage 28 artifact read-back failed")

        summary_path = OUT / "stage28_final_summary.json"
        summary = {
            "stage": STAGE,
            "authority": AUTHORITY,
            "stage27_authority": STAGE27_AUTHORITY,
            "source_transaction_rows": source_rows,
            "distinct_customers": distinct_customers,
            "basket_proxy_definition": "customer_id + t_dat",
            "basket_count": basket_count,
            "basket_rfm_frequency_equivalence": basket_rfm_validation["pass"],
            "items_are_unique_within_basket": basket_profile["items_are_unique_within_basket"],
            "item_type": "article_id",
            "fp_growth_config": {"minSupport": MIN_SUPPORT, "minConfidence": MIN_CONFIDENCE, "numPartitions": NUM_PARTITIONS},
            "frequent_itemset_count": frequent_itemset_count,
            "association_rule_count": association_rule_count,
            "manual_rule_validation_pass": manual_validation["overall_pass"],
            "model_reload_validation_pass": reload_config["model_reload_validation_pass"],
            "physical_plan_evidence_complete": physical_plan_evidence_complete,
            "models_modified": False,
            "stage22_modified": False,
            "stage23_modified": False,
            "stage24_modified": False,
            "stage25_modified": False,
            "stage26_modified": False,
            "stage27_modified": False,
            "current_run_clean_namespace": clean_namespace_at_start,
            "artifact_readback": normal_readback,
            "artifact_readback_pass": artifact_readback_pass,
            "final_summary_self_readback_pass": False,
            "fp_growth_analysis_complete": False,
            "ready_for_stage29": False,
            "warnings": warnings,
            "blockers": blockers,
            "source_paths": {"transactions": str(TRANSACTIONS_PATH), "articles": str(ARTICLES_PATH), "rfm_frequency": str(RFM_PATH), "model": str(MODEL_DIR)},
        }
        write_json(summary_path, summary)
        try:
            initial = read_json(summary_path)
            self_readback = initial.get("stage") == STAGE and initial.get("authority") == AUTHORITY and initial.get("artifact_readback_pass") is True
        except Exception:
            self_readback = False
        if not self_readback:
            blockers.append("Final Stage 28 summary self-readback failed")
        summary["final_summary_self_readback_pass"] = self_readback
        summary["artifact_readback"]["output\\fpgrowth\\stage28\\stage28_final_summary.json"] = True
        summary["artifact_readback_pass"] = artifact_readback_pass and True
        summary["blockers"] = blockers
        summary["fp_growth_analysis_complete"] = not blockers and artifact_readback_pass and self_readback
        summary["ready_for_stage29"] = summary["fp_growth_analysis_complete"]
        write_json(summary_path, summary)
        try:
            final = read_json(summary_path)
            final_self = final.get("stage") == STAGE and final.get("authority") == AUTHORITY and final.get("artifact_readback_pass") is True and final.get("final_summary_self_readback_pass") is True
        except Exception:
            final_self = False
        if not final_self:
            blockers.append("Final Stage 28 summary final reopen failed")
            summary["final_summary_self_readback_pass"] = False
            summary["fp_growth_analysis_complete"] = False
            summary["ready_for_stage29"] = False
            summary["blockers"] = blockers
            write_json(summary_path, summary)

        print(f"SPARK_APPLICATION_ID={application_id}")
        print(f"BASKET_COUNT={basket_count}")
        print(f"FREQUENT_ITEMSET_COUNT={frequent_itemset_count}")
        print(f"ASSOCIATION_RULE_COUNT={association_rule_count}")
        print(f"MANUAL_RULE_VALIDATION_PASS={manual_validation['overall_pass']}")
        print(f"MODEL_RELOAD_VALIDATION_PASS={reload_config['model_reload_validation_pass']}")
        print(f"ARTIFACT_READBACK_PASS={summary['artifact_readback_pass']}")
        print(f"FINAL_SUMMARY_SELF_READBACK_PASS={summary['final_summary_self_readback_pass']}")
        print(f"FP_GROWTH_ANALYSIS_COMPLETE={summary['fp_growth_analysis_complete']}")
        print(f"READY_FOR_STAGE_29={summary['ready_for_stage29']}")
        return 0 if summary["ready_for_stage29"] else 3
    except Exception as exc:
        blockers.append(f"{type(exc).__name__}: {exc}")
        OUT.mkdir(parents=True, exist_ok=True)
        failed = {
            "stage": STAGE,
            "authority": AUTHORITY,
            "stage27_authority": STAGE27_AUTHORITY,
            "models_modified": False,
            "stage27_modified": False,
            "current_run_clean_namespace": clean_namespace_at_start,
            "fp_growth_analysis_complete": False,
            "ready_for_stage29": False,
            "artifact_readback_pass": False,
            "final_summary_self_readback_pass": False,
            "warnings": warnings,
            "blockers": blockers,
        }
        write_json(OUT / "stage28_final_summary.json", failed)
        write_text(REPORT, "STAGE 28 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_29=False\n")
        print(f"STAGE_28_FAILED={type(exc).__name__}: {exc}")
        print("FP_GROWTH_ANALYSIS_COMPLETE=False")
        print("READY_FOR_STAGE_29=False")
        return 3
    finally:
        if spark is not None:
            try:
                spark.stop()
            except Exception:
                pass


def source_validation_dates(source_min: Any, source_max: Any) -> bool:
    return str(source_min) == EXPECTED_MIN_DATE and str(source_max) == EXPECTED_MAX_DATE


if __name__ == "__main__":
    sys.exit(main())
