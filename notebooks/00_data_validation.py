# Databricks notebook source
# notebooks/00_data_validation.py

# ----------------------------
# Environment
# ----------------------------
dbutils.widgets.text("env", "dev")
ENV = dbutils.widgets.get("env")

CATALOG = f"mlops_{ENV}"
SCHEMA = "raw"

# ----------------------------
# Load data
# ----------------------------
train_df = spark.table(f"{CATALOG}.{SCHEMA}.train")

# ----------------------------
# Gate 1: Non-empty dataset
# ----------------------------
row_count = train_df.count()
if row_count == 0:
    raise ValueError("❌ Data validation failed: train dataset is empty")

# ----------------------------
# Gate 2: Required columns
# ----------------------------
required_columns = {"Class"}
missing_cols = required_columns - set(train_df.columns)
if missing_cols:
    raise ValueError(f"❌ Missing required columns: {missing_cols}")

# ----------------------------
# Gate 3: Target validity
# ----------------------------
class_counts = (
    train_df
    .groupBy("Class")
    .count()
    .collect()
)

if len(class_counts) < 2:
    raise ValueError("❌ Only one class present in training data")

# ----------------------------
# Gate 4: Null ratio
# ----------------------------
total_rows = train_df.count()

for col in train_df.columns:
    nulls = train_df.filter(train_df[col].isNull()).count()
    if nulls / total_rows > 0.2:
        raise ValueError(f"❌ Column {col} has >20% nulls")

print("✅ Data validation passed")
