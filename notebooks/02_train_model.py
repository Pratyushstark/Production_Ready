# Databricks notebook source
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlflow.models import infer_signature

# ----------------------------
# Environment
# ----------------------------
dbutils.widgets.text("env", "dev")
ENV = dbutils.widgets.get("env")

CATALOG = f"mlops_{ENV}"
SCHEMA = "raw"

# ----------------------------
# Load data
# ----------------------------
train_df = spark.table(f"{CATALOG}.{SCHEMA}.train")
val_df   = spark.table(f"{CATALOG}.{SCHEMA}.val")

train_pdf = train_df.toPandas()
val_pdf   = val_df.toPandas()

X_train = train_pdf.drop(columns=["Class"])
y_train = train_pdf["Class"]

X_val = val_pdf.drop(columns=["Class"])
y_val = val_pdf["Class"]

# ----------------------------
# Train model
# ----------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ----------------------------
# Evaluate
# ----------------------------
val_preds = model.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)

# ----------------------------
# MLflow logging with experiment tracking
# ----------------------------
EXPERIMENT_NAME = "/Shared/mlops_dev"
mlflow.set_experiment(EXPERIMENT_NAME)

# Start a new MLflow run for the experiment
with mlflow.start_run(run_name="logreg_dev") as run:
    run_id = run.info.run_id

    # ----------------------------
    # Model signature
    # ----------------------------
    signature = infer_signature(
        model_input=X_train,
        model_output=model.predict(X_train)
    )

    # ----------------------------
    # Log Unity Catalog datasets
    # ----------------------------
    train_dataset = mlflow.data.from_spark(
        train_df,
        table_name=f"{CATALOG}.{SCHEMA}.train"
    )
    mlflow.log_input(train_dataset, context="training")

    val_dataset = mlflow.data.from_spark(
        val_df,
        table_name=f"{CATALOG}.{SCHEMA}.val"
    )
    mlflow.log_input(val_dataset, context="validation")

    # ----------------------------
    # Log model
    # ----------------------------
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="classifier_pipeline",
        signature=signature,
        input_example=X_val.iloc[:1]
    )

    # ----------------------------
    # Evaluate the model
    # ----------------------------
    eval_data = X_val.copy()
    eval_data['Class'] = y_val

    result = mlflow.models.evaluate(
        model=model_info.model_uri,
        data=eval_data,
        targets='Class',
        model_type="classifier",
        evaluators=["default"]
    )

    metrics = result.metrics
    mlflow.log_metric("val_accuracy", val_acc)

    # ----------------------------
    # Print Metrics & Log Final Results
    # ----------------------------
    print(f"Run ID: {run_id}")
    print(f"DEV training completed | Val accuracy: {val_acc}")
    print(f"Evaluation Results: {metrics}")

    # Log the metrics to MLflow
    mlflow.log_metric("val_accuracy", val_acc)

# ----------------------------
# Final Output
# ----------------------------
print(f"Model training and logging completed with experiment: {EXPERIMENT_NAME}")
