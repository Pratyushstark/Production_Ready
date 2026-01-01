# Databricks notebook source
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "mlops_dev.raw.fraud_model"
client = MlflowClient()

drift = dbutils.jobs.taskValues.get(
    taskKey="drift_detection",
    key="drift_detected"
)

perf = spark.table("mlops_prod.raw.performance_metrics").collect()[0]["f1_score"]

if not drift and perf >= 0.65:
    print("Model healthy. No rollback.")
    dbutils.notebook.exit("OK")

champion = client.get_model_version_by_alias(
    name=MODEL_NAME,
    alias="Champion"
)

versions = client.search_model_versions(f"name='{MODEL_NAME}'")
previous = sorted(
    [v for v in versions if int(v.version) < int(champion.version)],
    key=lambda v: int(v.version),
    reverse=True
)[0]

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="Champion",
    version=previous.version
)

print(f"Rolled back to version {previous.version}")
