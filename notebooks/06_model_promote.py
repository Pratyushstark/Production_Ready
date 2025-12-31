# Databricks notebook source
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score

# Environment
# ----------------------------
dbutils.widgets.text("env", "stag")
ENV = dbutils.widgets.get("env")

MIN_ACCURACY = 0.8

CATALOG = f"mlops_{ENV}"
SCHEMA = "raw"
MODEL_NAME = "mlops_dev.raw.fraud_model"

client = mlflow.tracking.MlflowClient()

value = dbutils.jobs.taskValues.get(taskKey = "model_evaluation", key="results")

model_uri = value["model_uri"]
version = value["model_version"]
f1_score = value["f1_score"]

model = mlflow.sklearn.load_model(model_uri)

# Hard gate
# ----------------------------
if f1_score < MIN_ACCURACY:
    print(f"current f1 score is:{f1_score}. Discarding the current dev model")
else:
    print(f"current f1 score is :{f1_score} better than the current staging model f1 score.")
    # ----------------------------
    # Transition to STAGING
    # ----------------------------
    client.set_registered_model_alias(
        name = MODEL_NAME,
        alias = "staging",
        version = version
    )

    print(f"✅ Alias 'latest-model' set to version {version}")

    # ----------------------------
    # Setting F1 Score as a tag
    # ----------------------------
    # Provide more details on this specific model version
    best_score = f1_score

    TAGS = {
    "promoted_by": "stag_pipeline",
    "f1_score": f"{round(best_score, 4)}"
}

    for key, value in TAGS.items():
        client.set_model_version_tag(
            name = MODEL_NAME,
            version = version,
            key = key,
            value = value
        )


print("Staging evaluation passed")
