from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    max as spark_max,
    min as spark_min,
    sum as spark_sum,
    count,
    datediff,
    date_add,
    lit,
    avg,
    when
)


# ============================================================
# 1. SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .master("local[4]")
    .appName("RFM-eRFM-Feature-Engineering")
    .config("spark.sql.shuffle.partitions", "64")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.parquet.columnarReaderBatchSize", "1024")
    .config("spark.local.dir", r"D:\BigData\spark_tmp")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# ============================================================
# 2. PATHS
# ============================================================

transactions_path = (
    r"D:\BigData\RFM_Spark\output\clean\transactions"
)

features_path = (
    r"D:\BigData\RFM_Spark\output\features\customer_features"
)

rfm_path = (
    r"D:\BigData\RFM_Spark\output\features\rfm"
)

erfm_path = (
    r"D:\BigData\RFM_Spark\output\features\erfm"
)


def title(text):
    print("\n")
    print("=" * 90)
    print(text)
    print("=" * 90)


# ============================================================
# 3. READ ONLY NECESSARY COLUMNS
# ============================================================

title("1. LOAD TRANSACTIONS")

transactions = (
    spark.read
    .parquet(transactions_path)
    .select(
        "t_dat",
        "customer_id",
        "article_id",
        "price"
    )
)

transactions.printSchema()


# ============================================================
# 4. ANALYSIS DATE
# ============================================================

title("2. ANALYSIS DATE")

date_info = (
    transactions
    .agg(
        spark_min("t_dat").alias("min_date"),
        spark_max("t_dat").alias("max_date")
    )
    .first()
)

min_date = date_info["min_date"]
max_date = date_info["max_date"]

analysis_date = (
    spark.createDataFrame(
        [(max_date,)],
        ["max_date"]
    )
    .select(
        date_add(
            col("max_date"),
            1
        ).alias("analysis_date")
    )
    .first()["analysis_date"]
)

print("Min transaction date :", min_date)
print("Max transaction date :", max_date)
print("Analysis date        :", analysis_date)


# ============================================================
# 5. CORE FEATURES
# ============================================================

title("3. CORE CUSTOMER AGGREGATION")

core = (
    transactions
    .groupBy("customer_id")
    .agg(
        spark_max("t_dat").alias(
            "last_purchase_date"
        ),

        spark_sum("price").alias(
            "Monetary"
        ),

        count("*").alias(
            "TotalItems"
        )
    )
)


# ============================================================
# 6. FREQUENCY
# ============================================================

title("4. BUILD FREQUENCY")

frequency = (
    transactions
    .select(
        "customer_id",
        "t_dat"
    )
    .dropDuplicates(
        ["customer_id", "t_dat"]
    )
    .groupBy("customer_id")
    .count()
    .withColumnRenamed(
        "count",
        "Frequency"
    )
)


# ============================================================
# 7. PRODUCT VARIETY
# ============================================================

title("5. BUILD PRODUCT VARIETY")

product_variety = (
    transactions
    .select(
        "customer_id",
        "article_id"
    )
    .dropDuplicates(
        ["customer_id", "article_id"]
    )
    .groupBy("customer_id")
    .count()
    .withColumnRenamed(
        "count",
        "ProductVariety"
    )
)


# ============================================================
# 8. JOIN FEATURES
# ============================================================

title("6. JOIN FEATURES")

customer_features = (
    core
    .join(
        frequency,
        on="customer_id",
        how="inner"
    )
    .join(
        product_variety,
        on="customer_id",
        how="inner"
    )
)


# ============================================================
# 9. RECENCY + AVG BASKET
# ============================================================

customer_features = (
    customer_features
    .withColumn(
        "Recency",
        datediff(
            lit(analysis_date),
            col("last_purchase_date")
        )
    )
    .withColumn(
        "AvgBasketSize",
        col("TotalItems") / col("Frequency")
    )
)


customer_features = customer_features.select(
    "customer_id",
    "last_purchase_date",
    "Recency",
    "Frequency",
    "Monetary",
    "TotalItems",
    "AvgBasketSize",
    "ProductVariety"
)


# ============================================================
# 10. WRITE CUSTOMER FEATURES FIRST
# ============================================================

title("7. SAVE CUSTOMER FEATURES")

customer_features.write \
    .mode("overwrite") \
    .parquet(features_path)

print("Customer features saved.")


# ============================================================
# 11. RELOAD RESULT
# ============================================================

features = spark.read.parquet(
    features_path
)

customer_count = features.count()

print(
    "Customer count:",
    customer_count
)


# ============================================================
# 12. SAMPLE
# ============================================================

title("8. SAMPLE FEATURES")

features.show(
    20,
    truncate=False
)


# ============================================================
# 13. VALIDATION
# ============================================================

title("9. VALIDATE FEATURES")

features.agg(

    spark_sum(
        when(
            col("Recency").isNull(),
            1
        ).otherwise(0)
    ).alias("null_recency"),

    spark_sum(
        when(
            col("Frequency").isNull(),
            1
        ).otherwise(0)
    ).alias("null_frequency"),

    spark_sum(
        when(
            col("Monetary").isNull(),
            1
        ).otherwise(0)
    ).alias("null_monetary"),

    spark_sum(
        when(
            col("AvgBasketSize").isNull(),
            1
        ).otherwise(0)
    ).alias("null_avg_basket"),

    spark_sum(
        when(
            col("ProductVariety").isNull(),
            1
        ).otherwise(0)
    ).alias("null_product_variety"),

    spark_sum(
        when(
            col("Recency") < 0,
            1
        ).otherwise(0)
    ).alias("invalid_recency"),

    spark_sum(
        when(
            col("Frequency") <= 0,
            1
        ).otherwise(0)
    ).alias("invalid_frequency"),

    spark_sum(
        when(
            col("Monetary") <= 0,
            1
        ).otherwise(0)
    ).alias("invalid_monetary"),

    spark_sum(
        when(
            col("AvgBasketSize") <= 0,
            1
        ).otherwise(0)
    ).alias("invalid_avg_basket"),

    spark_sum(
        when(
            col("ProductVariety") <= 0,
            1
        ).otherwise(0)
    ).alias("invalid_product_variety")

).show(truncate=False)


# ============================================================
# 14. SUMMARY
# ============================================================

title("10. FEATURE SUMMARY")

features.select(
    "Recency",
    "Frequency",
    "Monetary",
    "AvgBasketSize",
    "ProductVariety"
).summary(
    "count",
    "mean",
    "stddev",
    "min",
    "25%",
    "50%",
    "75%",
    "max"
).show(
    truncate=False
)


# ============================================================
# 15. RFM
# ============================================================

title("11. BUILD RFM")

rfm = features.select(
    "customer_id",
    "Recency",
    "Frequency",
    "Monetary"
)

rfm.write \
    .mode("overwrite") \
    .parquet(rfm_path)

print("RFM saved.")


# ============================================================
# 16. eRFM
# ============================================================

title("12. BUILD eRFM")

erfm = features.select(
    "customer_id",
    "Recency",
    "Frequency",
    "Monetary",
    "AvgBasketSize",
    "ProductVariety"
)

erfm.write \
    .mode("overwrite") \
    .parquet(erfm_path)

print("eRFM saved.")


# ============================================================
# 17. FINISH
# ============================================================

title("BUILD RFM/eRFM HOAN TAT")

print("Analysis Date :", analysis_date)
print("Customer Count:", customer_count)

print(
    "Customer Features:",
    features_path
)

print(
    "RFM:",
    rfm_path
)

print(
    "eRFM:",
    erfm_path
)

spark.stop()