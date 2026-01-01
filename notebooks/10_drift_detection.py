# Databricks notebook source
from pyspark.sql.functions import col, abs as sql_abs

CATALOG = "mlops_prod"
SCHEMA = "raw"

stats = spark.table(f"{CATALOG}.{SCHEMA}.monitoring_stats")

rows = stats.orderBy("data_phase").collect()

drift_flags = []

for i in range(1, len(rows)):
    prev = rows[i-1]
    curr = rows[i]

    mean_shift = abs(curr["mean_pred"] - prev["mean_pred"])
    std_shift = abs(curr["std_pred"] - prev["std_pred"])

    drift = mean_shift > 0.15 or std_shift > 0.15
    drift_flags.append(drift)

final_drift = any(drift_flags)

dbutils.jobs.taskValues.set("drift_detected", final_drift)

print(f"Drift detected: {final_drift}")
