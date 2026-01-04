# Databricks notebook source
# notebooks/09a_monitoring_stats.py

from pyspark.sql import functions as F
from datetime import timedelta

dbutils.widgets.text("env", "prod")
ENV = dbutils.widgets.get("env")

CATALOG = f"mlops_{ENV}"
SCHEMA = "raw"

PRED_TABLE = f"{CATALOG}.{SCHEMA}.batch_predictions"
MON_TABLE  = f"{CATALOG}.{SCHEMA}.model_monitoring_metrics"

# ---------------------------------------
# Time window (aligned with batch cadence)
# ---------------------------------------
now = spark.sql("SELECT current_timestamp()").collect()[0][0]
window_start = now - timedelta(minutes=5)

preds = (
    spark.table(PRED_TABLE)
    .filter(F.col("inference_time") >= F.lit(window_start))
)

# ---------------------------------------
# Guard: no new predictions
# ---------------------------------------
if preds.count() == 0:
    print("No new predictions in this window")
    dbutils.notebook.exit("OK")

# ---------------------------------------
# Metrics
# ---------------------------------------
volume = preds.count()

stats = preds.agg(
    F.mean("prediction").alias("mean_pred"),
    F.stddev("prediction").alias("std_pred"),
    F.expr("percentile(prediction, 0.95)").alias("p95_pred"),
    F.mean((F.col("prediction") > 0.9).cast("double")).alias("fraud_rate")
).collect()[0]

model_version = (
    preds.select("model_version")
    .limit(1)
    .collect()[0][0]
)

metrics = [
    ("inference_volume", volume),
    ("mean_prediction", stats["mean_pred"]),
    ("std_prediction", stats["std_pred"]),
    ("p95_prediction", stats["p95_pred"]),
    ("fraud_rate", stats["fraud_rate"]),
]

rows = [
    (window_start, now, name, float(value), model_version)
    for name, value in metrics
]

spark.createDataFrame(
    rows,
    ["window_start", "window_end", "metric_name", "metric_value", "model_version"]
).write.mode("append").saveAsTable(MON_TABLE)

print("Monitoring metrics appended successfully")
