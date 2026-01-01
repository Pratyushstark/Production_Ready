# Databricks notebook source
from pyspark.sql.functions import current_timestamp

CATALOG = "mlops_prod"
SCHEMA = "raw"

preds = spark.table(f"{CATALOG}.{SCHEMA}.batch_predictions")

labels = spark.table(f"{CATALOG}.{SCHEMA}.test") \
    .select("Class")

joined = preds.withColumn("true_label", labels["Class"]) \
              .withColumn("label_time", current_timestamp())

joined.write.mode("overwrite").saveAsTable(
    f"{CATALOG}.{SCHEMA}.delayed_labels"
)

print("Delayed ground truth joined")
