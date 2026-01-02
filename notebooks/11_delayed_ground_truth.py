# Databricks notebook source
from pyspark.sql.functions import current_timestamp
from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG = "mlops_prod"
SCHEMA = "raw"

preds = spark.table(f"{CATALOG}.{SCHEMA}.batch_predictions")

labels_test = spark.table(f"{CATALOG}.{SCHEMA}.test") \
    .select("Class")
labels_train = spark.table(f"{CATALOG}.{SCHEMA}.train") \
    .select("Class")
labels_val = spark.table(f"{CATALOG}.{SCHEMA}.val") \
    .select("Class")
labels = labels_test.union(labels_train).union(labels_val)


w = Window.orderBy(F.monotonically_increasing_id())

preds_idx = preds.withColumn("row_id", F.row_number().over(w))
labels_idx = labels.withColumn("row_id", F.row_number().over(w))

joined = (
    preds_idx
    .join(labels_idx, on="row_id")
    .drop("row_id")
    .withColumnRenamed("Class", "true_label")
    .withColumn("label_time", F.current_timestamp())
)

joined.write.mode("overwrite").saveAsTable(
    f"{CATALOG}.{SCHEMA}.delayed_labels"
)

print("Delayed ground truth joined")
