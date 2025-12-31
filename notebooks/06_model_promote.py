# Databricks notebook source
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score

# Environment
# ----------------------------
dbutils.widgets.text("env", "stag")
ENV = dbutils.widgets.get("env")

CATALOG = f"mlops_{ENV}"
SCHEMA = "raw"
value = dbutils.jobs.taskValues.get(taskKey = "model_evaluation", key="f1_score", default= 0)

if value < 0.8:
    raise RuntimeError(
        f"Model failed staging gate. "
        f"Accuracy {value:.4f} < {0.8}"
    )
