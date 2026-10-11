from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    trim,
    upper,
    to_date,
    when,
    count,
    countDistinct,
    min as spark_min,
    max as spark_max,
    avg,
    sum as spark_sum
)


# ============================================================
# 1. SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("RFM-Data-Cleaning")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Parquet compression
spark.conf.set("spark.sql.parquet.compression.codec", "snappy")


# ============================================================
# 2. DUONG DAN
# ============================================================

transactions_path = r"D:\BigData\RFM_Spark\data\transactions_train.csv"
customers_path = r"D:\BigData\RFM_Spark\data\customers.csv"
articles_path = r"D:\BigData\RFM_Spark\data\articles.csv"

transactions_output = r"D:\BigData\RFM_Spark\output\clean\transactions"
customers_output = r"D:\BigData\RFM_Spark\output\clean\customers"
articles_output = r"D:\BigData\RFM_Spark\output\clean\articles"


def title(text):
    print("\n")
    print("=" * 80)
    print(text)
    print("=" * 80)


# ============================================================
# 3. CLEAN TRANSACTIONS
# ============================================================

title("1. CLEAN TRANSACTIONS")


transactions_raw = (
    spark.read
    .option("header", True)
    .csv(transactions_path)
)


transactions = (
    transactions_raw

    # Ngay giao dich
    .withColumn(
        "t_dat",
        to_date(trim(col("t_dat")), "yyyy-MM-dd")
    )

    # Customer ID
    .withColumn(
        "customer_id",
        trim(col("customer_id"))
    )

    # Article ID
    .withColumn(
        "article_id",
        trim(col("article_id"))
    )

    # Price
    .withColumn(
        "price",
        col("price").cast("double")
    )

    # Sales channel
    .withColumn(
        "sales_channel_id",
        col("sales_channel_id").cast("int")
    )
)


# ------------------------------------------------------------
# Chi giu cac giao dich co thong tin bat buoc hop le
# ------------------------------------------------------------

transactions_clean = transactions.filter(

    col("t_dat").isNotNull()

    & col("customer_id").isNotNull()
    & (col("customer_id") != "")

    & col("article_id").isNotNull()
    & (col("article_id") != "")

    & col("price").isNotNull()
    & (col("price") > 0)

    & col("sales_channel_id").isin(1, 2)
)


print("\nSchema sau cleaning:")

transactions_clean.printSchema()


print("\nThong ke transactions sau cleaning:")

transactions_clean.agg(

    count("*").alias("total_rows"),

    countDistinct("customer_id").alias("unique_customers"),

    countDistinct("article_id").alias("unique_articles"),

    spark_min("t_dat").alias("min_date"),

    spark_max("t_dat").alias("max_date"),

    spark_min("price").alias("min_price"),

    spark_max("price").alias("max_price"),

    avg("price").alias("avg_price")

).show(truncate=False)


# ============================================================
# 4. CLEAN CUSTOMERS
# ============================================================

title("2. CLEAN CUSTOMERS")


customers_raw = (
    spark.read
    .option("header", True)
    .csv(customers_path)
)


customers_clean = (
    customers_raw

    .withColumn(
        "customer_id",
        trim(col("customer_id"))
    )

    # Age thanh so
    .withColumn(
        "age",
        col("age").cast("double")
    )

    # Neu age nam ngoai khoang hop ly -> NULL
    .withColumn(
        "age",
        when(
            (col("age") >= 16)
            & (col("age") <= 99),
            col("age")
        ).otherwise(None)
    )

    # Club member
    .withColumn(
        "club_member_status",
        when(
            col("club_member_status").isNull()
            | (trim(col("club_member_status")) == ""),
            "UNKNOWN"
        ).otherwise(
            upper(trim(col("club_member_status")))
        )
    )

    # Fashion news frequency
    .withColumn(
        "fashion_news_frequency",
        when(
            col("fashion_news_frequency").isNull()
            | (trim(col("fashion_news_frequency")) == ""),
            "UNKNOWN"
        ).otherwise(
            upper(trim(col("fashion_news_frequency")))
        )
    )

    # Postal code
    .withColumn(
        "postal_code",
        trim(col("postal_code"))
    )
)


# Customer ID la khoa bat buoc
customers_clean = customers_clean.filter(

    col("customer_id").isNotNull()
    & (col("customer_id") != "")
)


print("\nSchema customers sau cleaning:")

customers_clean.printSchema()


print("\nThong ke customers sau cleaning:")

customers_clean.agg(

    count("*").alias("total_rows"),

    countDistinct("customer_id").alias("unique_customers"),

    spark_min("age").alias("min_age"),

    spark_max("age").alias("max_age"),

    avg("age").alias("avg_age")

).show(truncate=False)


print("\nClub Member Status:")

customers_clean.groupBy(
    "club_member_status"
).count().orderBy(
    col("count").desc()
).show(truncate=False)


print("\nFashion News Frequency:")

customers_clean.groupBy(
    "fashion_news_frequency"
).count().orderBy(
    col("count").desc()
).show(truncate=False)


# ============================================================
# 5. CLEAN ARTICLES
# ============================================================

title("3. CLEAN ARTICLES")


articles_raw = (
    spark.read
    .option("header", True)
    .csv(articles_path)
)


articles_clean = articles_raw


# Trim tat ca cac cot string
for column_name in articles_clean.columns:

    articles_clean = articles_clean.withColumn(
        column_name,
        trim(col(column_name))
    )


# Article ID la khoa bat buoc
articles_clean = articles_clean.filter(

    col("article_id").isNotNull()
    & (col("article_id") != "")
)


print("\nSchema articles sau cleaning:")

articles_clean.printSchema()


print("\nThong ke articles sau cleaning:")

articles_clean.agg(

    count("*").alias("total_rows"),

    countDistinct("article_id").alias("unique_articles"),

    countDistinct("product_type_name").alias("product_types"),

    countDistinct("product_group_name").alias("product_groups"),

    countDistinct("section_name").alias("sections"),

    countDistinct("garment_group_name").alias("garment_groups"),

    countDistinct("colour_group_name").alias("colour_groups")

).show(truncate=False)


# ============================================================
# 6. KIEM TRA THAM CHIEU CUSTOMER
# ============================================================

title("4. VALIDATE CUSTOMER REFERENCES")


transaction_customers = (
    transactions_clean
    .select("customer_id")
    .distinct()
)


customer_master = (
    customers_clean
    .select("customer_id")
    .distinct()
)


missing_customer = (
    transaction_customers
    .join(
        customer_master,
        on="customer_id",
        how="left_anti"
    )
)


missing_customer_count = missing_customer.count()


print(
    "So customer co transaction nhung khong co trong customers.csv:",
    missing_customer_count
)


# ============================================================
# 7. KIEM TRA THAM CHIEU ARTICLE
# ============================================================

title("5. VALIDATE ARTICLE REFERENCES")


transaction_articles = (
    transactions_clean
    .select("article_id")
    .distinct()
)


article_master = (
    articles_clean
    .select("article_id")
    .distinct()
)


missing_article = (
    transaction_articles
    .join(
        article_master,
        on="article_id",
        how="left_anti"
    )
)


missing_article_count = missing_article.count()


print(
    "So article co transaction nhung khong co trong articles.csv:",
    missing_article_count
)


# ============================================================
# 8. LUU PARQUET
# ============================================================

title("6. SAVE CLEAN DATA AS PARQUET")


print("Dang luu transactions...")

transactions_clean.write \
    .mode("overwrite") \
    .parquet(transactions_output)


print("Transactions saved.")


print("Dang luu customers...")

customers_clean.write \
    .mode("overwrite") \
    .parquet(customers_output)


print("Customers saved.")


print("Dang luu articles...")

articles_clean.write \
    .mode("overwrite") \
    .parquet(articles_output)


print("Articles saved.")


# ============================================================
# 9. HOAN TAT
# ============================================================

title("DATA CLEANING HOAN TAT")

print("Transactions:", transactions_output)

print("Customers:", customers_output)

print("Articles:", articles_output)


spark.stop()