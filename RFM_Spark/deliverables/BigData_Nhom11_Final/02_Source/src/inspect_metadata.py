from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("Inspect-Metadata")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# ==============================
# CUSTOMERS
# ==============================

customers_path = r"D:\BigData\RFM_Spark\data\customers.csv"

customers = (
    spark.read
    .option("header", True)
    .csv(customers_path)
)

print("\n" + "=" * 60)
print("CUSTOMERS.CSV")
print("=" * 60)

print("\nCac cot:")
print(customers.columns)

print("\nSchema:")
customers.printSchema()

print("\n5 dong dau:")
customers.show(5, truncate=False)


# ==============================
# ARTICLES
# ==============================

articles_path = r"D:\BigData\RFM_Spark\data\articles.csv"

articles = (
    spark.read
    .option("header", True)
    .csv(articles_path)
)

print("\n" + "=" * 60)
print("ARTICLES.CSV")
print("=" * 60)

print("\nCac cot:")
print(articles.columns)

print("\nSchema:")
articles.printSchema()

print("\n5 dong dau:")
articles.show(5, truncate=False)


spark.stop()