#!/usr/bin/env python3
"""Stage 27: Spark SQL analytical layer and semantic validation.

This stage consumes locked authorities only.  It does not fit models, create
predictions, benchmark runtimes, or persist/cache the analytical inputs.
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
import urllib.error
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    abs as spark_abs,
    avg,
    col,
    count,
    countDistinct,
    expr,
    lit,
    max as spark_max,
    min as spark_min,
    percentile_approx,
    sum as spark_sum,
    when,
)


STAGE = 27
AUTHORITY = "SPARK_SQL_ANALYTICAL_LAYER_V1"
STAGE26_AUTHORITY = "SPARK_EXECUTION_EVIDENCE_V1"
EXPECTED_ROWS = 31_788_324
EXPECTED_CUSTOMERS = 1_362_281
EXPECTED_MIN_DATE = "2018-09-20"
EXPECTED_MAX_DATE = "2020-09-22"
ANALYSIS_DATE = "2020-09-23"
Q2_SUM = 298167.136781439
TOTAL_SUM = 884645.9740508008
TOLERANCE_FACTOR = 1e-8
EXPECTED_K = 5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE26_SUMMARY = PROJECT_ROOT / "output" / "analysis" / "stage26" / "stage26_final_summary.json"
TRANSACTIONS_PATH = PROJECT_ROOT / "output" / "clean" / "transactions"
CUSTOMERS_PATH = PROJECT_ROOT / "output" / "clean" / "customers"
ARTICLES_PATH = PROJECT_ROOT / "output" / "clean" / "articles"
CUSTOMER_FEATURES_PATH = PROJECT_ROOT / "output" / "features" / "customer_features"
RFM_PATH = PROJECT_ROOT / "output" / "features" / "rfm"
PREDICTIONS_PATH = PROJECT_ROOT / "output" / "final" / "enhanced_kmeans_predictions"
STAGE16_DIR = PROJECT_ROOT / "output" / "analysis" / "stage16"
OUT = PROJECT_ROOT / "output" / "sql" / "stage27"
SQL_DIR = PROJECT_ROOT / "sql" / "stage27"
REPORT = PROJECT_ROOT / "reports" / "stage27_spark_sql_analytical_layer.txt"
RUN_LOG = PROJECT_ROOT / "reports" / "stage27_run.log"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_json_value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float):
        return float(value)
    return value


def rows_as_dicts(rows: list[Any]) -> list[dict[str, Any]]:
    return [{key: as_json_value(value) for key, value in row.asDict().items()} for row in rows]


def tolerance(reference: float) -> float:
    return TOLERANCE_FACTOR * max(1.0, abs(reference))


def close_enough(actual: Any, reference: Any) -> tuple[float | None, float | None, bool]:
    if actual is None or reference is None:
        return None, None, False
    difference = abs(float(actual) - float(reference))
    allowed = tolerance(float(reference))
    return difference, allowed, difference <= allowed


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


def capture_explain(df: DataFrame) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        df.explain(mode="formatted")
    return buffer.getvalue()


def safe_ident(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def choose_column(columns: list[str], candidates: list[str], label: str) -> str:
    lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise RuntimeError(f"No authoritative {label} column found. Available columns: {columns}")


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=8) as response:
        return json.loads(response.read().decode("utf-8"))


def latest_sql_query_id(ui_url: str | None, seen_ids: set[int]) -> int | None:
    if not ui_url:
        return None
    try:
        application_id = None
        # Spark's UI URL does not expose app id; callers may provide the API base
        # through the environment variable set by run_stage.
        base = os.environ.get("STAGE27_REST_BASE")
        if not base:
            return None
        response = fetch_json(base.rstrip("/") + "/sql")
        candidates = response if isinstance(response, list) else []
        ids = [int(item["id"]) for item in candidates if "id" in item]
        new_ids = [item for item in ids if item not in seen_ids]
        return max(new_ids) if new_ids else (max(ids) if ids else None)
    except Exception:
        return None


def gate_stage26() -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    try:
        summary = read_json(STAGE26_SUMMARY)
    except Exception as exc:
        return [f"Cannot read Stage 26 summary: {type(exc).__name__}: {exc}"], {}
    checks = {
        "stage": (summary.get("stage"), 26),
        "authority": (summary.get("authority"), STAGE26_AUTHORITY),
        "source_row_count": (summary.get("source_row_count"), EXPECTED_ROWS),
        "execution_evidence_complete": (summary.get("execution_evidence_complete"), True),
        "artifact_readback_pass": (summary.get("artifact_readback_pass"), True),
        "final_summary_self_readback_pass": (summary.get("final_summary_self_readback_pass"), True),
        "current_run_clean_namespace": (summary.get("current_run_clean_namespace"), True),
        "ready_for_stage27": (summary.get("ready_for_stage27"), True),
        "blockers": (summary.get("blockers"), []),
        "models_modified": (summary.get("models_modified"), False),
    }
    for key, (actual, expected) in checks.items():
        if actual != expected:
            blockers.append(f"Stage26 gate mismatch {key}: actual={actual!r}, expected={expected!r}")
    return blockers, summary


def validate_artifacts(paths: dict[str, Path], spark: SparkSession | None = None) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for label, path in paths.items():
        if path.suffix.lower() == ".parquet":
            # Spark writes Parquet as a directory containing part files and
            # _SUCCESS, not as one filesystem file.
            valid = path.is_file() or (path.is_dir() and any(path.iterdir()))
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
        if valid and path.suffix.lower() == ".parquet" and spark is not None:
            try:
                valid = spark.read.parquet(str(path)).limit(1).count() >= 0
            except Exception:
                valid = False
        result[label] = valid
    return result


def sql_file_texts(product_group: str, product_name: str, age_column: str) -> dict[str, str]:
    pg = safe_ident(product_group)
    pn = safe_ident(product_name)
    age = safe_ident(age_column)
    return {
        "01_dataset_kpi_overview.sql": """SELECT\n  COUNT(*) AS transaction_item_rows,\n  COUNT(DISTINCT customer_id) AS distinct_customers,\n  COUNT(DISTINCT article_id) AS distinct_articles,\n  COUNT(DISTINCT t_dat) AS distinct_purchase_dates,\n  MIN(t_dat) AS min_purchase_date,\n  MAX(t_dat) AS max_purchase_date,\n  SUM(price) AS monetary_sum,\n  AVG(price) AS average_price_per_item_row,\n  MIN(price) AS minimum_price,\n  MAX(price) AS maximum_price\nFROM transactions_clean\n""",
        "02_monthly_purchase_trend.sql": """SELECT\n  DATE_TRUNC('month', t_dat) AS month,\n  COUNT(*) AS purchase_item_rows,\n  COUNT(DISTINCT customer_id) AS distinct_customers,\n  COUNT(DISTINCT article_id) AS distinct_articles,\n  SUM(price) AS monetary_sum,\n  AVG(price) AS average_price_per_item_row\nFROM transactions_clean\nGROUP BY DATE_TRUNC('month', t_dat)\nORDER BY month\n""",
        "03_customer_rfm_sql.sql": """WITH customer_rollup AS (\n  SELECT\n    customer_id,\n    MAX(t_dat) AS LastPurchaseDate,\n    DATEDIFF(DATE '2020-09-23', MAX(t_dat)) AS Recency,\n    COUNT(DISTINCT t_dat) AS Frequency,\n    SUM(price) AS Monetary,\n    COUNT(*) AS TotalItems,\n    COUNT(DISTINCT article_id) AS ProductVariety\n  FROM transactions_clean\n  GROUP BY customer_id\n)\nSELECT\n  customer_id, LastPurchaseDate, Recency, Frequency, Monetary, TotalItems,\n  ProductVariety, CAST(TotalItems AS DOUBLE) / Frequency AS AvgBasketSize\nFROM customer_rollup\n""",
        "04_product_performance.sql": f"""SELECT\n  a.{pg} AS product_group,\n  COUNT(*) AS purchase_item_rows,\n  COUNT(DISTINCT t.customer_id) AS distinct_customers,\n  COUNT(DISTINCT t.article_id) AS distinct_articles,\n  SUM(t.price) AS monetary_sum,\n  AVG(t.price) AS average_price_per_item_row\nFROM transactions_clean t\nJOIN articles_clean a ON t.article_id = a.article_id\nGROUP BY a.{pg}\nORDER BY monetary_sum DESC\n""",
        "05_customer_age_band_analysis.sql": f"""WITH age_banded AS (\n  SELECT\n    CASE\n      WHEN c.{age} IS NULL THEN 'UNKNOWN'\n      WHEN CAST(c.{age} AS DOUBLE) < 25 THEN 'UNDER_25'\n      WHEN CAST(c.{age} AS DOUBLE) BETWEEN 25 AND 34 THEN 'AGE_25_34'\n      WHEN CAST(c.{age} AS DOUBLE) BETWEEN 35 AND 44 THEN 'AGE_35_44'\n      WHEN CAST(c.{age} AS DOUBLE) BETWEEN 45 AND 54 THEN 'AGE_45_54'\n      WHEN CAST(c.{age} AS DOUBLE) >= 55 THEN 'AGE_55_PLUS'\n      ELSE 'UNKNOWN'\n    END AS age_band,\n    t.customer_id, t.article_id, t.price\n  FROM transactions_clean t\n  JOIN customers_clean c ON t.customer_id = c.customer_id\n)\nSELECT\n  age_band, COUNT(DISTINCT customer_id) AS customer_count,\n  COUNT(*) AS purchase_item_rows, COUNT(DISTINCT article_id) AS distinct_articles,\n  SUM(price) AS monetary_sum, AVG(price) AS average_price_per_item_row,\n  SUM(price) / COUNT(DISTINCT customer_id) AS average_monetary_per_customer,\n  COUNT(*) / COUNT(DISTINCT customer_id) AS average_purchase_rows_per_customer\nFROM age_banded\nGROUP BY age_band\nORDER BY age_band\n""",
        "06_m2_segment_analytics.sql": """WITH labeled AS (\n  SELECT\n    p.customer_id, CAST(p.prediction AS INT) AS prediction,\n    r.Recency, r.Frequency, r.Monetary, r.AvgBasketSize, r.ProductVariety,\n    CASE CAST(p.prediction AS INT)\n      WHEN 0 THEN 'Khách hàng mua ít nhưng giỏ lớn'\n      WHEN 1 THEN 'Khách hàng hoạt động giá trị cao'\n      WHEN 2 THEN 'Khách hàng lâu chưa quay lại'\n      WHEN 3 THEN 'Khách hàng ít tương tác'\n      WHEN 4 THEN 'Khách hàng hoạt động thường xuyên'\n    END AS segment_name_vi,\n    CASE CAST(p.prediction AS INT)\n      WHEN 0 THEN 'Infrequent Large-Basket Customers'\n      WHEN 1 THEN 'Active High-Value Customers'\n      WHEN 2 THEN 'Lapsed Customers'\n      WHEN 3 THEN 'Low-Engagement Customers'\n      WHEN 4 THEN 'Active Regular Customers'\n    END AS segment_name_en\n  FROM m2_predictions p\n  JOIN rfm_customer r ON p.customer_id = r.customer_id\n)\nSELECT\n  prediction, segment_name_vi, segment_name_en, COUNT(*) AS customer_count,\n  COUNT(*) / CAST(SUM(COUNT(*)) OVER () AS DOUBLE) AS customer_ratio,\n  AVG(Recency) AS mean_recency, percentile_approx(Recency, 0.5) AS median_recency,\n  AVG(Frequency) AS mean_frequency, percentile_approx(Frequency, 0.5) AS median_frequency,\n  AVG(Monetary) AS mean_monetary, percentile_approx(Monetary, 0.5) AS median_monetary,\n  AVG(AvgBasketSize) AS mean_avg_basket_size, percentile_approx(AvgBasketSize, 0.5) AS median_avg_basket_size,\n  AVG(ProductVariety) AS mean_product_variety, percentile_approx(ProductVariety, 0.5) AS median_product_variety\nFROM labeled\nGROUP BY prediction, segment_name_vi, segment_name_en\nORDER BY prediction\n""",
        "07_top_articles_by_product_group.sql": f"""WITH article_stats AS (\n  SELECT\n    t.article_id, a.{pg} AS product_group, a.{pn} AS product_name,\n    COUNT(*) AS purchase_item_rows, COUNT(DISTINCT t.customer_id) AS distinct_customers,\n    SUM(t.price) AS monetary_sum\n  FROM transactions_clean t\n  JOIN articles_clean a ON t.article_id = a.article_id\n  GROUP BY t.article_id, a.{pg}, a.{pn}\n), ranked AS (\n  SELECT *, ROW_NUMBER() OVER (PARTITION BY product_group ORDER BY monetary_sum DESC, article_id) AS product_group_rank\n  FROM article_stats\n)\nSELECT * FROM ranked WHERE product_group_rank <= 5\nORDER BY product_group, product_group_rank\n""",
    }


def main() -> int:
    clean_namespace_at_start = not OUT.exists() and not SQL_DIR.exists()
    blockers, stage26_summary = gate_stage26()
    warnings: list[str] = []
    spark: SparkSession | None = None
    OUT.mkdir(parents=True, exist_ok=True)
    SQL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if not clean_namespace_at_start:
            blockers.append("Stage 27 output or SQL namespace was not clean at process start")
        if blockers:
            raise RuntimeError("FAIL CLOSED: " + "; ".join(blockers))
        paths = [TRANSACTIONS_PATH, CUSTOMERS_PATH, ARTICLES_PATH, CUSTOMER_FEATURES_PATH, RFM_PATH, PREDICTIONS_PATH]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise RuntimeError("Authoritative source path missing: " + ", ".join(missing))

        spark = (
            SparkSession.builder.appName("Stage27_Spark_SQL_Analytical_Layer")
            .master("local[4]")
            .config("spark.sql.shuffle.partitions", "64")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.parquet.enableVectorizedReader", "false")
            .config("spark.sql.parquet.columnarReaderBatchSize", "512")
            .getOrCreate()
        )
        enable_windows_java_filesystem_fallback(spark)
        spark.sparkContext.setLogLevel("WARN")
        application_id = spark.sparkContext.applicationId
        ui_url = getattr(spark.sparkContext, "uiWebUrl", None)
        rest_base = None
        if ui_url:
            rest_base = str(ui_url).rstrip("/") + "/api/v1/applications/" + application_id
            os.environ["STAGE27_REST_BASE"] = rest_base
        environment = {
            "stage": STAGE,
            "authority": AUTHORITY,
            "spark_version": spark.version,
            "python_version": platform.python_version(),
            "java_version": str(spark._jvm.java.lang.System.getProperty("java.version")),
            "os": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "ram_bytes": None,
            "master": spark.sparkContext.master,
            "spark_application_id": application_id,
            "spark_ui_url": ui_url,
            "rest_base_url": rest_base,
            "spark_sql_shuffle_partitions": 64,
            "spark_sql_adaptive_enabled": True,
            "spark_sql_parquet_vectorized_reader": False,
            "spark_sql_parquet_columnar_reader_batch_size": 512,
            "analysis_date": ANALYSIS_DATE,
            "event_log_enabled": spark.conf.get("spark.eventLog.enabled", "false"),
        }
        try:
            import psutil  # type: ignore
            environment["ram_bytes"] = psutil.virtual_memory().total
        except Exception:
            pass
        write_json(OUT / "environment.json", environment)

        transactions = spark.read.parquet(str(TRANSACTIONS_PATH)).select("t_dat", "customer_id", "article_id", "price")
        customers = spark.read.parquet(str(CUSTOMERS_PATH))
        articles = spark.read.parquet(str(ARTICLES_PATH))
        rfm_authority = spark.read.parquet(str(CUSTOMER_FEATURES_PATH))
        rfm_authority_simple = spark.read.parquet(str(RFM_PATH))
        predictions = spark.read.parquet(str(PREDICTIONS_PATH)).select("customer_id", "prediction")
        customer_columns = customers.columns
        article_columns = articles.columns
        age_column = choose_column(customer_columns, ["age", "Age"], "customer age")
        product_group_column = choose_column(article_columns, ["product_group_name", "garment_group_name", "department_name", "section_name", "index_group_name"], "article product group")
        product_name_column = choose_column(article_columns, ["prod_name", "product_name", "detail_desc", "article_id"], "article product name")

        transactions.createOrReplaceTempView("transactions_clean")
        customers.createOrReplaceTempView("customers_clean")
        articles.createOrReplaceTempView("articles_clean")
        rfm_authority.createOrReplaceTempView("rfm_customer")
        predictions.createOrReplaceTempView("m2_predictions")
        registered_views = {
            "transactions_clean": {"path": str(TRANSACTIONS_PATH), "columns": transactions.columns},
            "customers_clean": {"path": str(CUSTOMERS_PATH), "columns": customers.columns},
            "articles_clean": {"path": str(ARTICLES_PATH), "columns": articles.columns},
            "rfm_customer": {"path": str(CUSTOMER_FEATURES_PATH), "columns": rfm_authority.columns},
            "m2_predictions": {"path": str(PREDICTIONS_PATH), "columns": predictions.columns},
            "selected_schema_columns": {"age_column": age_column, "product_group_column": product_group_column, "product_name_column": product_name_column},
        }
        write_json(OUT / "registered_views.json", registered_views)

        tx_stats = transactions.agg(
            count("*").alias("row_count"), countDistinct("customer_id").alias("distinct_customers"),
            spark_min("t_dat").alias("min_date"), spark_max("t_dat").alias("max_date"),
            spark_sum(when(col("customer_id").isNull(), 1).otherwise(0)).alias("null_customer_id"),
            spark_sum(when(col("article_id").isNull(), 1).otherwise(0)).alias("null_article_id"),
            spark_sum(when(col("t_dat").isNull(), 1).otherwise(0)).alias("null_t_dat"),
            spark_sum(when(col("price").isNull(), 1).otherwise(0)).alias("null_price"),
        ).collect()[0].asDict()
        missing_customer_refs = transactions.join(customers.select("customer_id").dropDuplicates(), "customer_id", "left_anti").count()
        missing_article_refs = transactions.join(articles.select("article_id").dropDuplicates(), "article_id", "left_anti").count()
        source_validation = {key: as_json_value(value) for key, value in tx_stats.items()}
        source_validation.update({
            "customer_authority_path": str(CUSTOMERS_PATH), "article_authority_path": str(ARTICLES_PATH),
            "rfm_authority_path": str(CUSTOMER_FEATURES_PATH), "prediction_authority_path": str(PREDICTIONS_PATH),
            "missing_customer_references": missing_customer_refs, "missing_article_references": missing_article_refs,
            "age_column": age_column, "product_group_column": product_group_column, "product_name_column": product_name_column,
        })
        source_validation["pass"] = (
            source_validation["row_count"] == EXPECTED_ROWS and source_validation["distinct_customers"] == EXPECTED_CUSTOMERS
            and source_validation["min_date"] == EXPECTED_MIN_DATE and source_validation["max_date"] == EXPECTED_MAX_DATE
            and all(source_validation[key] == 0 for key in ["null_customer_id", "null_article_id", "null_t_dat", "null_price", "missing_customer_references", "missing_article_references"])
        )
        write_json(OUT / "source_authority_validation.json", source_validation)
        if not source_validation["pass"]:
            raise RuntimeError(f"Source authority validation failed: {source_validation}")

        sql_texts = sql_file_texts(product_group_column, product_name_column, age_column)
        for filename, sql_text in sql_texts.items():
            write_text(SQL_DIR / filename, sql_text)

        query_manifest: dict[str, Any] = {}
        seen_sql_ids: set[int] = set()

        def run_query(query_id: str, query_name: str, sql_filename: str, sql_text: str, output_artifact: str | None = None, plan_artifact: str | None = None, write_csv_path: Path | None = None, csv_rows: list[dict[str, Any]] | None = None) -> tuple[DataFrame, list[dict[str, Any]], int | None]:
            nonlocal seen_sql_ids
            df = spark.sql(sql_text)
            if plan_artifact:
                write_text(OUT / plan_artifact, capture_explain(df))
            if output_artifact:
                df.write.mode("overwrite").parquet(str(OUT / output_artifact))
                row_count = df.count()
            else:
                rows = rows_as_dicts(df.collect())
                row_count = len(rows)
            if write_csv_path is not None and csv_rows is not None:
                write_csv(write_csv_path, csv_rows)
            current_sql_id = latest_sql_query_id(ui_url, seen_sql_ids)
            if current_sql_id is not None:
                seen_sql_ids.add(current_sql_id)
            query_manifest[query_id] = {
                "query_id": current_sql_id,
                "query_name": query_name,
                "sql_file": f"sql/stage27/{sql_filename}",
                "input_views": ["transactions_clean"],
                "output_artifact": output_artifact or "inline_json_artifact",
                "output_row_count": row_count,
                "validation_pass": None,
                "physical_plan_artifact": f"output/sql/stage27/{plan_artifact}" if plan_artifact else None,
                "runtime_sec_context_only": None,
            }
            return df, rows_as_dicts(df.limit(1000).collect()) if output_artifact and row_count <= 1000 else [], current_sql_id

        # Q1
        q1_sql = sql_texts["01_dataset_kpi_overview.sql"]
        q1_df = spark.sql(q1_sql)
        q1 = rows_as_dicts(q1_df.collect())[0]
        q1_diff, q1_tol, q1_sum_pass = close_enough(q1["monetary_sum"], TOTAL_SUM)
        q1.update({"transaction_item_rows_pass": q1["transaction_item_rows"] == EXPECTED_ROWS, "distinct_customers_pass": q1["distinct_customers"] == EXPECTED_CUSTOMERS, "date_range_pass": q1["min_purchase_date"] == EXPECTED_MIN_DATE and q1["max_purchase_date"] == EXPECTED_MAX_DATE, "monetary_difference": q1_diff, "monetary_tolerance": q1_tol, "monetary_sum_pass": q1_sum_pass, "validation_pass": q1["transaction_item_rows"] == EXPECTED_ROWS and q1["distinct_customers"] == EXPECTED_CUSTOMERS and q1["min_purchase_date"] == EXPECTED_MIN_DATE and q1["max_purchase_date"] == EXPECTED_MAX_DATE and q1_sum_pass})
        write_json(OUT / "q1_dataset_kpi.json", q1)
        query_manifest["Q1"] = {"query_id": latest_sql_query_id(ui_url, seen_sql_ids), "query_name": "DATASET_KPI_OVERVIEW", "sql_file": "sql/stage27/01_dataset_kpi_overview.sql", "input_views": ["transactions_clean"], "output_artifact": "output/sql/stage27/q1_dataset_kpi.json", "output_row_count": 1, "validation_pass": q1["validation_pass"], "physical_plan_artifact": None, "runtime_sec_context_only": None}

        # Q2
        q2_sql = sql_texts["02_monthly_purchase_trend.sql"]
        q2_df = spark.sql(q2_sql)
        q2_df.write.mode("overwrite").parquet(str(OUT / "q2_monthly_purchase_trend.parquet"))
        q2_rows = rows_as_dicts(q2_df.collect())
        write_csv(OUT / "q2_monthly_purchase_trend.csv", q2_rows)
        monthly_totals = q2_df.agg(count("*").alias("months"), spark_sum("purchase_item_rows").alias("rows"), spark_sum("monetary_sum").alias("monetary_sum")).collect()[0].asDict()
        q2_row_diff, q2_row_tol, q2_row_pass = close_enough(monthly_totals["rows"], EXPECTED_ROWS)
        q2_sum_diff, q2_sum_tol, q2_sum_pass = close_enough(monthly_totals["monetary_sum"], TOTAL_SUM)
        q2_validation = {"monthly_row_count": len(q2_rows), "monthly_purchase_item_rows": monthly_totals["rows"], "monthly_monetary_sum": monthly_totals["monetary_sum"], "row_difference": q2_row_diff, "row_tolerance": q2_row_tol, "monetary_difference": q2_sum_diff, "monetary_tolerance": q2_sum_tol, "purchase_item_rows_reconcile": q2_row_pass, "monetary_reconcile": q2_sum_pass, "pass": q2_row_pass and q2_sum_pass}
        write_json(OUT / "q2_validation.json", q2_validation)
        query_manifest["Q2"] = {"query_id": latest_sql_query_id(ui_url, seen_sql_ids), "query_name": "MONTHLY_PURCHASE_TREND", "sql_file": "sql/stage27/02_monthly_purchase_trend.sql", "input_views": ["transactions_clean"], "output_artifact": "output/sql/stage27/q2_monthly_purchase_trend.parquet", "output_row_count": len(q2_rows), "validation_pass": q2_validation["pass"], "physical_plan_artifact": None, "runtime_sec_context_only": None}

        # Q3
        q3_sql = sql_texts["03_customer_rfm_sql.sql"]
        q3_df = spark.sql(q3_sql)
        q3_df.write.mode("overwrite").parquet(str(OUT / "q3_customer_rfm_sql.parquet"))
        q3_count = q3_df.count()
        rfm_fields = ["Recency", "Frequency", "TotalItems", "ProductVariety", "Monetary", "AvgBasketSize"]
        comparison = q3_df.alias("s").join(rfm_authority.alias("a"), "customer_id", "full")
        compare_agg = comparison.agg(
            count(when(col("s.customer_id").isNull(), 1)).alias("missing_sql_customers"),
            count(when(col("a.customer_id").isNull(), 1)).alias("missing_stage5_customers"),
            count(when(col("s.Recency") != col("a.Recency"), 1)).alias("recency_mismatch_count"),
            count(when(col("s.Frequency") != col("a.Frequency"), 1)).alias("frequency_mismatch_count"),
            count(when(col("s.TotalItems") != col("a.TotalItems"), 1)).alias("total_items_mismatch_count"),
            count(when(col("s.ProductVariety") != col("a.ProductVariety"), 1)).alias("product_variety_mismatch_count"),
            spark_max(spark_abs(col("s.Monetary") - col("a.Monetary"))).alias("monetary_max_abs_diff"),
            avg(spark_abs(col("s.Monetary") - col("a.Monetary"))).alias("monetary_mean_abs_diff"),
            spark_max(spark_abs(col("s.AvgBasketSize") - col("a.AvgBasketSize"))).alias("avg_basket_max_abs_diff"),
            avg(spark_abs(col("s.AvgBasketSize") - col("a.AvgBasketSize"))).alias("avg_basket_mean_abs_diff"),
        ).collect()[0].asDict()
        sql_customer_count = q3_df.select("customer_id").distinct().count()
        monetary_diff, monetary_tol, monetary_pass = close_enough(compare_agg["monetary_max_abs_diff"] or 0.0, 0.0)
        avg_diff, avg_tol, avg_pass = close_enough(compare_agg["avg_basket_max_abs_diff"] or 0.0, 0.0)
        q3_equivalence = {key: as_json_value(value) for key, value in compare_agg.items()}
        q3_equivalence.update({"sql_customer_count": sql_customer_count, "stage5_customer_count": rfm_authority.select("customer_id").distinct().count(), "monetary_tolerance": monetary_tol, "avg_basket_tolerance": avg_tol, "monetary_pass": monetary_pass, "avg_basket_pass": avg_pass, "customer_universe_pass": compare_agg["missing_sql_customers"] == 0 and compare_agg["missing_stage5_customers"] == 0, "exact_feature_pass": all(compare_agg[key] == 0 for key in ["recency_mismatch_count", "frequency_mismatch_count", "total_items_mismatch_count", "product_variety_mismatch_count"]), "pass": compare_agg["missing_sql_customers"] == 0 and compare_agg["missing_stage5_customers"] == 0 and all(compare_agg[key] == 0 for key in ["recency_mismatch_count", "frequency_mismatch_count", "total_items_mismatch_count", "product_variety_mismatch_count"]) and monetary_pass and avg_pass})
        write_json(OUT / "q3_rfm_authority_comparison.json", q3_equivalence)
        write_text(OUT / "q3_rfm_physical_plan.txt", capture_explain(q3_df))
        query_manifest["Q3"] = {"query_id": latest_sql_query_id(ui_url, seen_sql_ids), "query_name": "CUSTOMER_RFM_SQL_RECONSTRUCTION", "sql_file": "sql/stage27/03_customer_rfm_sql.sql", "input_views": ["transactions_clean"], "output_artifact": "output/sql/stage27/q3_customer_rfm_sql.parquet", "output_row_count": q3_count, "validation_pass": q3_equivalence["pass"], "physical_plan_artifact": "output/sql/stage27/q3_rfm_physical_plan.txt", "runtime_sec_context_only": None}

        # Q4 product analysis
        q4_df = spark.sql(sql_texts["04_product_performance.sql"])
        q4_df.write.mode("overwrite").parquet(str(OUT / "q4_product_performance.parquet"))
        q4_rows = rows_as_dicts(q4_df.collect())
        write_csv(OUT / "q4_top20_product_groups.csv", q4_rows[:20])
        product_join_rows = transactions.join(articles.select("article_id").dropDuplicates(), "article_id", "inner").count()
        q4_validation = {"valid_article_join_rows": product_join_rows, "expected_rows": EXPECTED_ROWS, "pass": product_join_rows == EXPECTED_ROWS}
        write_json(OUT / "q4_validation.json", q4_validation)
        write_text(OUT / "q4_product_join_physical_plan.txt", capture_explain(q4_df))
        query_manifest["Q4"] = {"query_id": latest_sql_query_id(ui_url, seen_sql_ids), "query_name": "PRODUCT_PERFORMANCE", "sql_file": "sql/stage27/04_product_performance.sql", "input_views": ["transactions_clean", "articles_clean"], "output_artifact": "output/sql/stage27/q4_product_performance.parquet", "output_row_count": len(q4_rows), "validation_pass": q4_validation["pass"], "physical_plan_artifact": "output/sql/stage27/q4_product_join_physical_plan.txt", "runtime_sec_context_only": None}

        # Q5 customer demographics
        q5_df = spark.sql(sql_texts["05_customer_age_band_analysis.sql"])
        q5_df.write.mode("overwrite").parquet(str(OUT / "q5_age_band_analysis.parquet"))
        q5_rows = rows_as_dicts(q5_df.collect())
        write_csv(OUT / "q5_age_band_analysis.csv", q5_rows)
        customer_join_rows = transactions.join(customers.select("customer_id").dropDuplicates(), "customer_id", "inner").count()
        age_rows_sum = sum(int(row["purchase_item_rows"]) for row in q5_rows)
        q5_validation = {"valid_customer_join_rows": customer_join_rows, "age_band_purchase_item_rows": age_rows_sum, "expected_rows": EXPECTED_ROWS, "pass": customer_join_rows == EXPECTED_ROWS and age_rows_sum == EXPECTED_ROWS}
        write_json(OUT / "q5_validation.json", q5_validation)
        query_manifest["Q5"] = {"query_id": latest_sql_query_id(ui_url, seen_sql_ids), "query_name": "CUSTOMER_AGE_BAND_ANALYSIS", "sql_file": "sql/stage27/05_customer_age_band_analysis.sql", "input_views": ["transactions_clean", "customers_clean"], "output_artifact": "output/sql/stage27/q5_age_band_analysis.parquet", "output_row_count": len(q5_rows), "validation_pass": q5_validation["pass"], "physical_plan_artifact": None, "runtime_sec_context_only": None}

        # Q6 final M2 segment analytics
        stage16_summary = read_json(STAGE16_DIR / "final_enhanced_cluster_profile_summary.json")
        stage16_labels = stage16_summary["segment_mapping"]
        expected_counts = {int(row["cluster_id"]): int(row["count"]) for row in stage16_summary["cluster_distribution"]}
        q6_df = spark.sql(sql_texts["06_m2_segment_analytics.sql"])
        q6_df.write.mode("overwrite").parquet(str(OUT / "q6_m2_segment_analytics.parquet"))
        q6_rows = rows_as_dicts(q6_df.collect())
        write_csv(OUT / "q6_m2_segment_analytics.csv", q6_rows)
        prediction_counts = {int(row["prediction"]): int(row["customer_count"]) for row in q6_rows}
        prediction_total = predictions.count()
        prediction_distinct = predictions.select("customer_id").distinct().count()
        missing_rfm_predictions = predictions.join(rfm_authority.select("customer_id").dropDuplicates(), "customer_id", "left_anti").count()
        q6_validation = {"prediction_rows": prediction_total, "prediction_distinct_customers": prediction_distinct, "expected_prediction_rows": EXPECTED_CUSTOMERS, "expected_k": EXPECTED_K, "observed_cluster_counts": prediction_counts, "expected_cluster_counts": expected_counts, "stage16_labels": stage16_labels, "missing_rfm_predictions": missing_rfm_predictions, "prediction_count_pass": prediction_total == EXPECTED_CUSTOMERS and prediction_distinct == EXPECTED_CUSTOMERS, "k_pass": len(prediction_counts) == EXPECTED_K, "membership_authority_match": prediction_counts == expected_counts and missing_rfm_predictions == 0, "pass": prediction_total == EXPECTED_CUSTOMERS and prediction_distinct == EXPECTED_CUSTOMERS and len(prediction_counts) == EXPECTED_K and prediction_counts == expected_counts and missing_rfm_predictions == 0}
        write_json(OUT / "q6_m2_segment_validation.json", q6_validation)
        write_text(OUT / "q6_m2_segment_join_physical_plan.txt", capture_explain(q6_df))
        query_manifest["Q6"] = {"query_id": latest_sql_query_id(ui_url, seen_sql_ids), "query_name": "FINAL_M2_SEGMENT_ANALYTICS", "sql_file": "sql/stage27/06_m2_segment_analytics.sql", "input_views": ["m2_predictions", "rfm_customer"], "output_artifact": "output/sql/stage27/q6_m2_segment_analytics.parquet", "output_row_count": len(q6_rows), "validation_pass": q6_validation["pass"], "physical_plan_artifact": "output/sql/stage27/q6_m2_segment_join_physical_plan.txt", "runtime_sec_context_only": None}

        # Q7 window function analysis
        q7_df = spark.sql(sql_texts["07_top_articles_by_product_group.sql"])
        q7_df.write.mode("overwrite").parquet(str(OUT / "q7_top_articles_by_product_group.parquet"))
        q7_rows = rows_as_dicts(q7_df.collect())
        write_csv(OUT / "q7_top_articles_by_product_group.csv", q7_rows)
        write_text(OUT / "q7_window_physical_plan.txt", capture_explain(q7_df))
        query_manifest["Q7"] = {"query_id": latest_sql_query_id(ui_url, seen_sql_ids), "query_name": "TOP_ARTICLES_BY_PRODUCT_GROUP", "sql_file": "sql/stage27/07_top_articles_by_product_group.sql", "input_views": ["transactions_clean", "articles_clean"], "output_artifact": "output/sql/stage27/q7_top_articles_by_product_group.parquet", "output_row_count": len(q7_rows), "validation_pass": bool(q7_rows), "physical_plan_artifact": "output/sql/stage27/q7_window_physical_plan.txt", "runtime_sec_context_only": None}

        # Independent DataFrame API validations.
        df_q1 = transactions.agg(count("*").alias("rows"), countDistinct("customer_id").alias("customers"), spark_min("t_dat").alias("min_date"), spark_max("t_dat").alias("max_date"), spark_sum("price").alias("sum_price")).collect()[0].asDict()
        df_q2 = transactions.groupBy(expr("date_trunc('month', t_dat)").alias("month")).agg(count("*").alias("purchase_item_rows"), spark_sum("price").alias("monetary_sum"))
        df_q2_total = df_q2.agg(spark_sum("purchase_item_rows").alias("rows"), spark_sum("monetary_sum").alias("sum_price")).collect()[0].asDict()
        df_q6_counts = {int(row["prediction"]): int(row["count"]) for row in predictions.groupBy("prediction").count().withColumnRenamed("count", "count").collect()}
        df_q1_diff, df_q1_tol, df_q1_sum_pass = close_enough(df_q1["sum_price"], TOTAL_SUM)
        df_q2_diff, df_q2_tol, df_q2_sum_pass = close_enough(df_q2_total["sum_price"], TOTAL_SUM)
        sql_dataframe_validation = {"Q1": {"dataframe_rows": df_q1["rows"], "dataframe_customers": df_q1["customers"], "dataframe_min_date": as_json_value(df_q1["min_date"]), "dataframe_max_date": as_json_value(df_q1["max_date"]), "monetary_difference": df_q1_diff, "monetary_tolerance": df_q1_tol, "pass": df_q1["rows"] == EXPECTED_ROWS and df_q1["customers"] == EXPECTED_CUSTOMERS and str(df_q1["min_date"]) == EXPECTED_MIN_DATE and str(df_q1["max_date"]) == EXPECTED_MAX_DATE and df_q1_sum_pass}, "Q2": {"dataframe_monthly_rows": df_q2_total["rows"], "dataframe_monthly_sum": df_q2_total["sum_price"], "monetary_difference": df_q2_diff, "monetary_tolerance": df_q2_tol, "pass": df_q2_total["rows"] == EXPECTED_ROWS and df_q2_sum_pass}, "Q6": {"dataframe_cluster_counts": df_q6_counts, "sql_cluster_counts": prediction_counts, "pass": df_q6_counts == prediction_counts}, "pass": False}
        sql_dataframe_validation["pass"] = all(sql_dataframe_validation[key]["pass"] for key in ["Q1", "Q2", "Q6"])
        write_json(OUT / "sql_dataframe_validation.json", sql_dataframe_validation)

        plan_texts = {name: (OUT / name).read_text(encoding="utf-8") for name in ["q3_rfm_physical_plan.txt", "q4_product_join_physical_plan.txt", "q6_m2_segment_join_physical_plan.txt", "q7_window_physical_plan.txt"]}
        physical_plan_findings = {
            "q3": {"contains_scan_parquet": "Scan parquet" in plan_texts["q3_rfm_physical_plan.txt"], "contains_hash_aggregate": "HashAggregate" in plan_texts["q3_rfm_physical_plan.txt"], "contains_exchange": "Exchange" in plan_texts["q3_rfm_physical_plan.txt"]},
            "q4": {"contains_scan_parquet": "Scan parquet" in plan_texts["q4_product_join_physical_plan.txt"], "join_operator_observed": next((operator for operator in ["SortMergeJoin", "BroadcastHashJoin", "ShuffledHashJoin"] if operator in plan_texts["q4_product_join_physical_plan.txt"]), None)},
            "q6": {"contains_scan_parquet": "Scan parquet" in plan_texts["q6_m2_segment_join_physical_plan.txt"], "join_operator_observed": next((operator for operator in ["SortMergeJoin", "BroadcastHashJoin", "ShuffledHashJoin"] if operator in plan_texts["q6_m2_segment_join_physical_plan.txt"]), None)},
            "q7": {"contains_window": "Window" in plan_texts["q7_window_physical_plan.txt"], "contains_sort": "Sort" in plan_texts["q7_window_physical_plan.txt"], "contains_row_number": "row_number" in plan_texts["q7_window_physical_plan.txt"].lower()},
        }
        physical_plan_evidence_complete = all(item.get("contains_scan_parquet", False) for item in [physical_plan_findings["q3"], physical_plan_findings["q4"], physical_plan_findings["q6"]]) and all(physical_plan_findings["q7"].values())

        write_json(OUT / "query_execution_manifest.json", query_manifest)
        monthly_top = max(q2_rows, key=lambda row: float(row["monetary_sum"])) if q2_rows else {}
        product_top = q4_rows[0] if q4_rows else {}
        age_top = max(q5_rows, key=lambda row: float(row["monetary_sum"])) if q5_rows else {}
        m2_top = max(q6_rows, key=lambda row: float(row["customer_count"])) if q6_rows else {}
        article_top = q7_rows[0] if q7_rows else {}
        findings = {
            "dataset": [{"metric": "transaction_item_rows", "value": EXPECTED_ROWS, "source_query": "Q1"}, {"metric": "distinct_customers", "value": EXPECTED_CUSTOMERS, "source_query": "Q1"}],
            "monthly_trend": [{"metric": "highest_monthly_monetary_sum", "value": monthly_top, "source_query": "Q2"}],
            "products": [{"metric": "top_product_group_by_monetary_sum", "value": product_top, "source_query": "Q4"}],
            "customer_demographics": [{"metric": "top_age_band_by_monetary_sum", "value": age_top, "source_query": "Q5"}],
            "m2_segments": [{"metric": "largest_m2_segment_by_customer_count", "value": m2_top, "source_query": "Q6"}],
            "window_analysis": [{"metric": "top_article_within_first_product_group", "value": article_top, "source_query": "Q7"}],
        }
        write_json(OUT / "spark_sql_findings.json", findings)

        core_pass = all([q1["validation_pass"], q2_validation["pass"], q3_equivalence["pass"], q4_validation["pass"], q5_validation["pass"], q6_validation["pass"], bool(q7_rows), sql_dataframe_validation["pass"], physical_plan_evidence_complete])
        if not q3_equivalence["pass"]:
            blockers.append("SQL RFM differs from Stage 5 authority")
        if not q6_validation["pass"]:
            blockers.append("M2 membership differs from Stage 16 authority")
        if not q4_validation["pass"]:
            blockers.append("Product join lost transaction rows")
        if not q5_validation["pass"]:
            blockers.append("Customer join or age-band totals failed")
        if not sql_dataframe_validation["pass"]:
            blockers.append("SQL/DataFrame validation failed")
        if not physical_plan_evidence_complete:
            blockers.append("Required physical-plan evidence incomplete")
        if not core_pass:
            blockers.append("One or more Stage 27 query validations failed")

        required_output = [
            "environment.json", "source_authority_validation.json", "registered_views.json", "q1_dataset_kpi.json",
            "q2_monthly_purchase_trend.parquet", "q2_monthly_purchase_trend.csv", "q2_validation.json", "q3_customer_rfm_sql.parquet", "q3_rfm_authority_comparison.json",
            "q4_product_performance.parquet", "q4_top20_product_groups.csv", "q4_validation.json", "q5_age_band_analysis.parquet", "q5_age_band_analysis.csv", "q5_validation.json",
            "q6_m2_segment_analytics.parquet", "q6_m2_segment_analytics.csv", "q6_m2_segment_validation.json", "q7_top_articles_by_product_group.parquet", "q7_top_articles_by_product_group.csv",
            "sql_dataframe_validation.json", "query_execution_manifest.json", "spark_sql_findings.json", "q3_rfm_physical_plan.txt", "q4_product_join_physical_plan.txt", "q6_m2_segment_join_physical_plan.txt", "q7_window_physical_plan.txt",
        ]
        required_files = {f"output\\sql\\stage27\\{name}": OUT / name for name in required_output}
        required_files.update({f"sql\\stage27\\{name}": SQL_DIR / name for name in sql_texts})
        required_files.update({"reports\\stage27_spark_sql_analytical_layer.txt": REPORT, "reports\\stage27_run.log": RUN_LOG})
        report_text = "\n".join([
            "STAGE 27 SPARK SQL ANALYTICAL LAYER", "A. Data authorities", json.dumps(registered_views, ensure_ascii=False),
            "B. SQL views", ", ".join(registered_views.keys()), "C. Dataset overview", json.dumps(q1, ensure_ascii=False),
            "D. Monthly purchase trend", json.dumps(q2_validation, ensure_ascii=False), "E. SQL RFM reconstruction", json.dumps(q3_equivalence, ensure_ascii=False),
            "F. Product analysis", json.dumps(q4_rows[:20], ensure_ascii=False), "G. Customer age-band analysis", json.dumps(q5_rows, ensure_ascii=False),
            "H. M2 segment analysis", json.dumps(q6_rows, ensure_ascii=False), "I. Window-function analysis", json.dumps(q7_rows[:20], ensure_ascii=False),
            "J. Physical-plan findings", json.dumps(physical_plan_findings, ensure_ascii=False), "K. Validation", json.dumps({"source": source_validation, "sql_dataframe": sql_dataframe_validation}, ensure_ascii=False),
            "L. Business findings", json.dumps(findings, ensure_ascii=False), "NO_RUNTIME_RANKING=True", "NO_MODEL_TRAINING=True",
        ]) + "\n"
        write_text(REPORT, report_text)
        artifact_readback = validate_artifacts(required_files, spark)
        artifact_readback_pass = all(artifact_readback.values())
        if not artifact_readback_pass:
            blockers.append("Stage 27 artifact read-back failed")

        summary = {
            "stage": STAGE, "authority": AUTHORITY, "stage26_authority": STAGE26_AUTHORITY,
            "transaction_row_count": EXPECTED_ROWS, "distinct_transaction_customers": EXPECTED_CUSTOMERS,
            "sql_views_registered": list(registered_views.keys())[:5], "queries_completed": {key: True for key in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"]},
            "rfm_sql_stage5_semantic_equivalence": q3_equivalence["pass"], "m2_membership_authority_match": q6_validation["pass"], "sql_dataframe_validation_pass": sql_dataframe_validation["pass"],
            "product_join_validation_pass": q4_validation["pass"], "customer_join_validation_pass": q5_validation["pass"], "physical_plan_evidence_complete": physical_plan_evidence_complete,
            "models_modified": False, "stage22_modified": False, "stage23_modified": False, "stage24_modified": False, "stage25_modified": False, "stage26_modified": False,
            "spark_application_id": application_id, "spark_ui_url": ui_url, "source_paths": {"transactions": str(TRANSACTIONS_PATH), "customers": str(CUSTOMERS_PATH), "articles": str(ARTICLES_PATH), "rfm_customer": str(CUSTOMER_FEATURES_PATH), "m2_predictions": str(PREDICTIONS_PATH)},
            "current_run_clean_namespace": clean_namespace_at_start, "artifact_readback": artifact_readback, "artifact_readback_pass": artifact_readback_pass,
            "spark_sql_analytical_layer_complete": False, "ready_for_stage28": False, "warnings": warnings, "blockers": blockers,
        }
        write_json(OUT / "stage27_final_summary.json", summary)
        try:
            parsed = read_json(OUT / "stage27_final_summary.json")
            self_readback = bool((OUT / "stage27_final_summary.json").stat().st_size > 0 and parsed.get("stage") == STAGE and parsed.get("authority") == AUTHORITY and parsed.get("artifact_readback_pass") is True)
        except Exception:
            self_readback = False
        if not self_readback:
            blockers.append("Final Stage 27 summary self-readback failed")
        summary["final_summary_self_readback_pass"] = self_readback
        summary["spark_sql_analytical_layer_complete"] = not blockers and artifact_readback_pass and self_readback
        summary["ready_for_stage28"] = not blockers and artifact_readback_pass and self_readback
        summary["blockers"] = blockers
        write_json(OUT / "stage27_final_summary.json", summary)
        final_self = False
        try:
            final_parsed = read_json(OUT / "stage27_final_summary.json")
            final_self = final_parsed.get("final_summary_self_readback_pass") is True and final_parsed.get("artifact_readback_pass") is True
        except Exception:
            final_self = False
        if not final_self:
            blockers.append("Final Stage 27 summary final reopen failed")
            summary["final_summary_self_readback_pass"] = False
            summary["spark_sql_analytical_layer_complete"] = False
            summary["ready_for_stage28"] = False
            summary["blockers"] = blockers
            write_json(OUT / "stage27_final_summary.json", summary)

        print(f"SPARK_APPLICATION_ID={application_id}")
        print(f"Q1_PASS={q1['validation_pass']}")
        print(f"Q2_PASS={q2_validation['pass']}")
        print(f"Q3_RFM_PASS={q3_equivalence['pass']}")
        print(f"Q4_PASS={q4_validation['pass']}")
        print(f"Q5_PASS={q5_validation['pass']}")
        print(f"Q6_PASS={q6_validation['pass']}")
        print(f"Q7_ROWS={len(q7_rows)}")
        print(f"ARTIFACT_READBACK_PASS={artifact_readback_pass}")
        print(f"FINAL_SUMMARY_SELF_READBACK_PASS={final_self}")
        print(f"SPARK_SQL_ANALYTICAL_LAYER_COMPLETE={summary['spark_sql_analytical_layer_complete']}")
        print(f"READY_FOR_STAGE_28={summary['ready_for_stage28']}")
        return 0 if summary["ready_for_stage28"] else 3
    except Exception as exc:
        blockers.append(f"{type(exc).__name__}: {exc}")
        failed = {"stage": STAGE, "authority": AUTHORITY, "stage26_authority": STAGE26_AUTHORITY, "models_modified": False, "stage26_modified": False, "current_run_clean_namespace": clean_namespace_at_start, "artifact_readback_pass": False, "final_summary_self_readback_pass": False, "spark_sql_analytical_layer_complete": False, "ready_for_stage28": False, "warnings": warnings, "blockers": blockers}
        write_json(OUT / "stage27_final_summary.json", failed)
        write_text(REPORT, "STAGE 27 FAILED CLOSED\n" + "\n".join(blockers) + "\nREADY_FOR_STAGE_28=False\n")
        print(f"STAGE_27_FAILED={type(exc).__name__}: {exc}")
        print("SPARK_SQL_ANALYTICAL_LAYER_COMPLETE=False")
        print("READY_FOR_STAGE_28=False")
        return 3
    finally:
        if spark is not None:
            try:
                spark.stop()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
