# Databricks notebook source
from sklearn.metrics import f1_score
import pandas as pd

df = spark.table("mlops_prod.raw.delayed_labels").toPandas()

f1 = f1_score(df["true_label"], df["prediction"] > 0.5)

spark.createDataFrame(
    [(f1,)],
    ["f1_score"]
).write.mode("overwrite").saveAsTable(
    "mlops_prod.raw.performance_metrics"
)

print(f"Delayed F1 score: {f1}")
