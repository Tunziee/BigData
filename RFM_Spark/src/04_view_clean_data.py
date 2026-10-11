from pyspark.sql import SparkSession


# ============================================================
# 1. TAO SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("View-Clean-Parquet")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# ============================================================
# 2. DUONG DAN PARQUET
# ============================================================

transactions_path = r"D:\BigData\RFM_Spark\output\clean\transactions"
customers_path = r"D:\BigData\RFM_Spark\output\clean\customers"
articles_path = r"D:\BigData\RFM_Spark\output\clean\articles"


def print_title(text):
    print("\n")
    print("=" * 100)
    print(text)
    print("=" * 100)


def inspect_dataset(name, path, rows=10):
    print_title(name)

    df = spark.read.parquet(path)

    print("\n--- SCHEMA ---")
    df.printSchema()

    print("\n--- CAC COT ---")
    print(df.columns)

    print(f"\n--- {rows} DONG DAU ---")
    df.show(rows, truncate=False)

    print("\n--- SO LUONG COT ---")
    print(len(df.columns))


# ============================================================
# 3. XEM TRANSACTIONS
# ============================================================

inspect_dataset(
    "1. CLEAN TRANSACTIONS",
    transactions_path,
    rows=10
)


# ============================================================
# 4. XEM CUSTOMERS
# ============================================================

inspect_dataset(
    "2. CLEAN CUSTOMERS",
    customers_path,
    rows=10
)


# ============================================================
# 5. XEM ARTICLES
# ============================================================

inspect_dataset(
    "3. CLEAN ARTICLES",
    articles_path,
    rows=5
)


# ============================================================
# 6. HOAN TAT
# ============================================================

print_title("VIEW CLEAN DATA HOAN TAT")

spark.stop()