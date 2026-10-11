from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    to_date,
    count,
    countDistinct,
    min as spark_min,
    max as spark_max,
    avg,
    stddev,
    sum as spark_sum,
    when,
    trim
)


# ============================================================
# 1. TAO SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("RFM-Data-Profiling")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# ============================================================
# HAM HO TRO
# ============================================================

def title(text):
    print("\n")
    print("=" * 80)
    print(text)
    print("=" * 80)


def show_null_counts(df, columns):
    expressions = []

    for c in columns:
        expressions.append(
            spark_sum(
                when(
                    col(c).isNull()
                    | (trim(col(c).cast("string")) == ""),
                    1
                ).otherwise(0)
            ).alias(c)
        )

    df.agg(*expressions).show(truncate=False)


# ============================================================
# DUONG DAN
# ============================================================

transactions_path = r"D:\BigData\RFM_Spark\data\transactions_train.csv"
customers_path = r"D:\BigData\RFM_Spark\data\customers.csv"
articles_path = r"D:\BigData\RFM_Spark\data\articles.csv"


# ============================================================
# 2. PROFILE TRANSACTIONS
# ============================================================

title("1. TRANSACTIONS_TRAIN.CSV")


transactions_raw = (
    spark.read
    .option("header", True)
    .csv(transactions_path)
)


transactions = transactions_raw.select(
    to_date(col("t_dat"), "yyyy-MM-dd").alias("t_dat"),
    col("customer_id"),
    col("article_id"),
    col("price").cast("double").alias("price"),
    col("sales_channel_id").cast("int").alias("sales_channel_id")
)


print("\n--- Schema ---")
transactions.printSchema()


print("\n--- Tong quan giao dich ---")

transactions.agg(

    count("*").alias("total_rows"),

    countDistinct("customer_id").alias("unique_customers"),

    countDistinct("article_id").alias("unique_articles"),

    spark_min("t_dat").alias("min_date"),

    spark_max("t_dat").alias("max_date"),

    spark_min("price").alias("min_price"),

    spark_max("price").alias("max_price"),

    avg("price").alias("avg_price"),

    stddev("price").alias("std_price"),

    spark_sum(
        when(col("price") <= 0, 1).otherwise(0)
    ).alias("price_le_0")

).show(truncate=False)


print("\n--- NULL / rong ---")

show_null_counts(
    transactions,
    [
        "t_dat",
        "customer_id",
        "article_id",
        "price",
        "sales_channel_id"
    ]
)


print("\n--- Phan bo sales_channel_id ---")

transactions.groupBy(
    "sales_channel_id"
).count().orderBy(
    "sales_channel_id"
).show(truncate=False)


# ============================================================
# 3. PROFILE CUSTOMERS
# ============================================================

title("2. CUSTOMERS.CSV")


customers_raw = (
    spark.read
    .option("header", True)
    .csv(customers_path)
)


customers = customers_raw.withColumn(
    "age_numeric",
    col("age").cast("double")
)


print("\n--- Schema ---")
customers.printSchema()


print("\n--- Tong quan customers ---")

customers.agg(

    count("*").alias("total_rows"),

    countDistinct("customer_id").alias("unique_customers"),

    spark_min("age_numeric").alias("min_age"),

    spark_max("age_numeric").alias("max_age"),

    avg("age_numeric").alias("avg_age"),

    stddev("age_numeric").alias("std_age"),

    spark_sum(
        when(
            (col("age_numeric") < 0)
            | (col("age_numeric") > 100),
            1
        ).otherwise(0)
    ).alias("invalid_age")

).show(truncate=False)


print("\n--- NULL / rong ---")

show_null_counts(
    customers,
    [
        "customer_id",
        "FN",
        "Active",
        "club_member_status",
        "fashion_news_frequency",
        "age",
        "postal_code"
    ]
)


print("\n--- club_member_status ---")

customers.groupBy(
    "club_member_status"
).count().orderBy(
    col("count").desc()
).show(20, truncate=False)


print("\n--- fashion_news_frequency ---")

customers.groupBy(
    "fashion_news_frequency"
).count().orderBy(
    col("count").desc()
).show(20, truncate=False)


# ============================================================
# 4. PROFILE ARTICLES
# ============================================================

title("3. ARTICLES.CSV")


articles = (
    spark.read
    .option("header", True)
    .csv(articles_path)
)


print("\n--- Schema ---")
articles.printSchema()


print("\n--- Tong quan articles ---")

articles.agg(

    count("*").alias("total_rows"),

    countDistinct("article_id").alias("unique_articles"),

    countDistinct("product_type_name").alias("product_types"),

    countDistinct("product_group_name").alias("product_groups"),

    countDistinct("section_name").alias("sections"),

    countDistinct("garment_group_name").alias("garment_groups"),

    countDistinct("colour_group_name").alias("colour_groups")

).show(truncate=False)


print("\n--- NULL tren cac cot quan trong ---")

show_null_counts(
    articles,
    [
        "article_id",
        "prod_name",
        "product_type_name",
        "product_group_name",
        "colour_group_name",
        "department_name",
        "index_name",
        "section_name",
        "garment_group_name"
    ]
)


print("\n--- Top Product Group ---")

articles.groupBy(
    "product_group_name"
).count().orderBy(
    col("count").desc()
).show(10, truncate=False)


print("\n--- Top Garment Group ---")

articles.groupBy(
    "garment_group_name"
).count().orderBy(
    col("count").desc()
).show(10, truncate=False)


print("\n--- Top Section ---")

articles.groupBy(
    "section_name"
).count().orderBy(
    col("count").desc()
).show(10, truncate=False)


print("\n--- Top Colour ---")

articles.groupBy(
    "colour_group_name"
).count().orderBy(
    col("count").desc()
).show(10, truncate=False)


# ============================================================
# KET THUC
# ============================================================

title("DATA PROFILING HOAN TAT")

spark.stop()