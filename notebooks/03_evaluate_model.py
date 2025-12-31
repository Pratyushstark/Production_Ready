# Databricks notebook source
# notebooks/03_evaluate_model.py

import mlflow
from sklearn.metrics import accuracy_score

# ----------------------------
# Environment
# ----------------------------
dbutils.widgets.text("env", "dev")
ENV = dbutils.widgets.get("env")

CATALOG = f"mlops_{ENV}"
SCHEMA = "raw"

# ----------------------------
# Load test data
# ----------------------------
test_df = spark.table(f"{CATALOG}.{SCHEMA}.test")
test_pdf = test_df.toPandas()

X_test = test_pdf.drop(columns=["Class"])
y_test = test_pdf["Class"]

# ----------------------------
# Load latest model
# ----------------------------
mlflow.set_experiment("/Shared/mlops_dev")
runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)

model_uri = f"runs:/{runs.iloc[0].run_id}/classifier_pipeline"
model = mlflow.sklearn.load_model(model_uri)

# ----------------------------
# Evaluate
# ----------------------------
test_preds = model.predict(X_test)
test_acc = accuracy_score(y_test, test_preds)

print(f"DEV test accuracy: {test_acc}")
