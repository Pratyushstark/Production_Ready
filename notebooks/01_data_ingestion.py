# Databricks notebook source
# notebooks/01_data_ingestion.py

from sklearn.model_selection import train_test_split

# ----------------------------
# Environment
# ----------------------------
dbutils.widgets.text("env", "dev")
ENV = dbutils.widgets.get("env")

CATALOG = f"mlops_{ENV}"
SCHEMA = "raw"

# ----------------------------
# Read raw data
# ----------------------------
df = spark.read.csv(
    "/Volumes/workspace/default/raw_data",
    header=True,
    inferSchema=True
)

df = df.withColumn("Class", df["Class"].cast("int"))

pdf = df.toPandas()

# ----------------------------
# Stratified split
# ----------------------------
train_pdf, temp_pdf = train_test_split(
    pdf,
    test_size=0.30,
    stratify=pdf["Class"],
    random_state=42
)

val_pdf, test_pdf = train_test_split(
    temp_pdf,
    test_size=0.50,
    stratify=temp_pdf["Class"],
    random_state=42
)

# ----------------------------
# Pandas → Spark
# ----------------------------
train_df = spark.createDataFrame(train_pdf)
val_df   = spark.createDataFrame(val_pdf)
test_df  = spark.createDataFrame(test_pdf)

# ----------------------------
# Create catalog + schema (DEV ONLY)
# ----------------------------
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# ----------------------------
# Write Delta tables
# ----------------------------
train_df.write.mode("overwrite").format("delta").saveAsTable(
    f"{CATALOG}.{SCHEMA}.train"
)

val_df.write.mode("overwrite").format("delta").saveAsTable(
    f"{CATALOG}.{SCHEMA}.val"
)

test_df.write.mode("overwrite").format("delta").saveAsTable(
    f"{CATALOG}.{SCHEMA}.test"
)

print("DEV data ingestion completed")
