# Databricks notebook source
# notebooks/04_register_model_dev.py

import mlflow
from mlflow.tracking import MlflowClient

# ----------------------------
# Environment
# ----------------------------
dbutils.widgets.text("env", "dev")
ENV = dbutils.widgets.get("env")

CATALOG = f"mlops_{ENV}"
SCHEMA = "raw"

# ----------------------------
# Config
# ----------------------------
DEV_EXPERIMENT = "/Shared/mlops_dev"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.fraud_model"
ARTIFACT_PATH = "classifier_pipeline"
TAGS = {
    "team": "mlops",
    "project": "fraud_classification",
    "source_env": "dev",
    "candidate": "true"
}

client = MlflowClient()

# ----------------------------
# Get best DEV run
# ----------------------------
runs = mlflow.search_runs(
    experiment_names=[DEV_EXPERIMENT],
    order_by=["metrics.f1_score DESC"],
    max_results=1,
)

if runs.empty:
    raise RuntimeError("No DEV runs found")

run_id = runs.iloc[0].run_id
model_uri = f"runs:/{run_id}/{ARTIFACT_PATH}"

# ----------------------------
# Register model (DEV responsibility ENDS HERE)
# ----------------------------
model_version = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME,
    tags=TAGS
)

# ----------------------------
# Attach metrics for traceability
# ----------------------------
best_score = runs["metrics.f1_score"].values[0]

client.set_model_version_tag(
    name=model_version.name,
    version=model_version.version,
    key="f1_score",
    value=f"{round(best_score, 4)}"
)

print(f"✅ DEV registered model version {model_version.version} as candidate")
