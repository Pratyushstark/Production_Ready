# Databricks notebook source
from pyspark.sql.functions import avg, stddev, count

CATALOG = "mlops_prod"
SCHEMA = "raw"

df = spark.table(f"{CATALOG}.{SCHEMA}.batch_predictions")

stats = (
    df
    .groupBy("model_version", "data_phase")
    .agg(
        count("*").alias("num_preds"),
        avg("prediction").alias("mean_pred"),
        stddev("prediction").alias("std_pred")
    )
)

stats.write.mode("overwrite").saveAsTable(
    f"{CATALOG}.{SCHEMA}.monitoring_stats"
)

print("Monitoring stats computed")
