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

MODEL_NAME = "mlops_dev.raw.fraud_model"
ARTIFACT_PATH = "classifier_pipeline"
# TAGS = {
#     "team": "mlops",
#     "project": "fraud_classification",
#     "source_env": "stag",
#     "candidate": "true",
#     "promoted_by": "stag_pipeline"
# }

# Metric thresholds (GATES)
# ----------------------------
MIN_ACCURACY = 0.66

# Load test data (read-only)
# ----------------------------
test_df = spark.table(f"{CATALOG}.{SCHEMA}.test")
test_pdf = test_df.toPandas()
val_df = spark.table(f"{CATALOG}.{SCHEMA}.val")
val_pdf = val_df.toPandas()

df = test_pdf.merge(val_pdf,how="outer")
df = df.reset_index(drop=True)
print(df.shape)

X_test = df.drop(columns=["Class"])
y_test = df["Class"]


client = mlflow.tracking.MlflowClient()

# Fetch candidate models registered by DEV
versions = client.search_model_versions(
    f"name='{MODEL_NAME}'"
)

if not versions:
    raise RuntimeError("No DEV candidate model found for promotion")

eligible = []

for v in versions:
    mv = client.get_model_version(
        name=v.name,
        version=v.version
    )

    if (
        mv.tags.get("candidate") == "true"
        and mv.tags.get("source_env") == "dev"
        and mv.current_stage is None
    ):
        eligible.append(mv)

# Choose latest version
latest = max(versions, key=lambda v: int(v.version))

model_uri = f"models:/{MODEL_NAME}/{latest.version}"
model = mlflow.sklearn.load_model(model_uri)

# Evaluate
# ----------------------------
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
f1_score = f1_score(y_test, preds)

print(f"STAGING evaluation accuracy: {f1_score}")

# Hard gate
# ----------------------------
if f1_score < MIN_ACCURACY:
    # raise RuntimeError(
    #     f"Model failed staging gate. "
    #     f"Accuracy {f1_score:.4f} < {MIN_ACCURACY}"
    # )
    print(f"current f1 score is:{f1_score}. Discarding the current dev model")
else:
    print("If current f1 score is better than the current staging model f1 score.")
    # ----------------------------
    # Transition to STAGING
    # ----------------------------
    client.set_registered_model_alias(
        name = MODEL_NAME,
        alias = "staging",
        version = latest.version
    )

    print(f"✅ Alias 'latest-model' set to version {latest.version}")

    # ----------------------------
    # Setting F1 Score as a tag
    # ----------------------------
    # Provide more details on this specific model version
    best_score = f1_score

    # We can also tag the model version with the F1 score for visibility. This will add f1 score as a tag
    # TAGS = {
    #     "promoted_by": "stag_pipeline",
    #     "f1_score": f"{round(best_score,4)}"
    # }

    # client.set_model_version_tag(
    #         name = latest.name,
    #         version = latest.version,
    #         # tags = TAGS
    #         key = "promoted_by",
    #         value = "stag_pipeline"
    #     )

    TAGS = {
    "promoted_by": "stag_pipeline",
    "f1_score": f"{round(best_score, 4)}"
}

    for key, value in TAGS.items():
        client.set_model_version_tag(
            name=latest.name,
            version=latest.version,
            key=key,
            value=value
        )


print("Staging evaluation passed")
dbutils.jobs.taskValues.set(key="f1_score",value=f1_score)

# # Promote to Staging
# client.transition_model_version_stage(
#     name=MODEL_NAME,
#     version=latest.version,
#     stage="Staging",
#     archive_existing_versions=True
# )

# # Optional but recommended
# client.set_model_version_tag(
#     name=MODEL_NAME,
#     version=latest.version,
#     key="promoted_by",
#     value="staging_pipeline"
# )


