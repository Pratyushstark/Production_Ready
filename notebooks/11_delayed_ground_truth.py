# Databricks notebook source
# from pyspark.sql.functions import current_timestamp

# CATALOG = "mlops_prod"
# SCHEMA = "raw"

# preds = spark.table(f"{CATALOG}.{SCHEMA}.batch_predictions")

# labels = spark.table(f"{CATALOG}.{SCHEMA}.test") \
#     .select("Class")

# joined = preds.withColumn("true_label", labels["Class"]) \
#               .withColumn("label_time", current_timestamp())

# joined.write.mode("overwrite").saveAsTable(
#     f"{CATALOG}.{SCHEMA}.delayed_labels"
# )

# print("Delayed ground truth joined")

# from pyspark.sql.functions import sha2, concat_ws, col, current_timestamp

# # ----------------------------
# # Config
# # ----------------------------
# CATALOG = "mlops_prod"
# SCHEMA = "raw"

# ID_COLUMNS = [
#     "Time",
#     "Amount",
#     *[f"V{i}" for i in range(1, 29)]
# ]

# # ----------------------------
# # Load tables
# # ----------------------------
# preds = spark.table(f"{CATALOG}.{SCHEMA}.batch_predictions")
# labels = spark.table(f"{CATALOG}.{SCHEMA}.test")

# # ----------------------------
# # Create deterministic record_id
# # ----------------------------
# def add_record_id(df):
#     return df.withColumn(
#         "record_id",
#         sha2(concat_ws("||", *ID_COLUMNS), 256)
#     )

# preds = add_record_id(preds)
# labels = add_record_id(labels)

# # ----------------------------
# # Validate record_id
# # ----------------------------
# if preds.filter(col("record_id").isNull()).count() > 0:
#     raise ValueError("Null record_id found in batch_predictions")

# if labels.filter(col("record_id").isNull()).count() > 0:
#     raise ValueError("Null record_id found in test table")

# if preds.groupBy("record_id").count().filter("count > 1").count() > 0:
#     raise ValueError("Duplicate record_id found in batch_predictions")

# if labels.groupBy("record_id").count().filter("count > 1").count() > 0:
#     raise ValueError("Duplicate record_id found in test table")

# # ----------------------------
# # Join delayed ground truth
# # ----------------------------
# delayed_labels = (
#     preds.join(
#         labels.select("record_id", "Class"),
#         on="record_id",
#         how="left"
#     )
#     .withColumnRenamed("Class", "true_label")
#     .withColumn("label_time", current_timestamp())
# )

# # ----------------------------
# # Persist result
# # ----------------------------
# delayed_labels.write.mode("overwrite").saveAsTable(
#     f"{CATALOG}.{SCHEMA}.delayed_labels"
# )

# print("Delayed ground truth table created successfully")


from pyspark.sql.functions import (
    sha2, concat_ws, col, current_timestamp, row_number
)
from pyspark.sql.window import Window

CATALOG = "mlops_prod"
SCHEMA = "raw"

ID_COLUMNS = [
    "Time",
    "Amount",
    *[f"V{i}" for i in range(1, 29)]
]

# ----------------------------
# Load tables
# ----------------------------
preds = spark.table(f"{CATALOG}.{SCHEMA}.batch_predictions")
labels = spark.table(f"{CATALOG}.{SCHEMA}.test")

# ----------------------------
# Base hash (feature identity)
# ----------------------------
def add_base_hash(df):
    return df.withColumn(
        "base_hash",
        sha2(concat_ws("||", *ID_COLUMNS), 256)
    )

preds = add_base_hash(preds)
labels = add_base_hash(labels)

# ----------------------------
# Disambiguate duplicates safely
# ----------------------------
window = Window.partitionBy("base_hash").orderBy("base_hash")

preds = preds.withColumn(
    "record_id",
    sha2(
        concat_ws("||", col("base_hash"), row_number().over(window)),
        256
    )
)

labels = labels.withColumn(
    "record_id",
    sha2(
        concat_ws("||", col("base_hash"), row_number().over(window)),
        256
    )
)

# ----------------------------
# Join delayed ground truth
# ----------------------------
delayed_labels = (
    preds.join(
        labels.select("record_id", "Class"),
        on="record_id",
        how="left"
    )
    .withColumnRenamed("Class", "true_label")
    .withColumn("label_time", current_timestamp())
)

# ----------------------------
# Persist
# ----------------------------
delayed_labels.write.mode("overwrite").saveAsTable(
    f"{CATALOG}.{SCHEMA}.delayed_labels"
)

print("Delayed ground truth table created successfully")

