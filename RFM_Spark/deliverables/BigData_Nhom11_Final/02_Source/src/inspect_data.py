from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("RFM-Inspect-Data")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

file_path = r"D:\BigData\RFM_Spark\data\transactions_train.csv"

df = (
    spark.read
    .option("header", True)
    .option("mode", "PERMISSIVE")
    .csv(file_path)
)

print("\n===== DANH SACH COT =====")
print(df.columns)

print("\n===== SCHEMA =====")
df.printSchema()

print("\n===== 10 DONG DAU =====")
df.show(10, truncate=False)

print("\n===== SO LUONG COT =====")
print(len(df.columns))

spark.stop()