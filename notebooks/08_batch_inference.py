# Databricks notebook source
import mlflow
from pyspark.sql.functions import struct, current_timestamp, lit
from mlflow.tracking import MlflowClient

# dbutils.widgets.text("phase", "test")  # test | train | val
# PHASE = dbutils.widgets.get("phase")

CATALOG = "mlops_prod"
SCHEMA = "raw"

MODEL_NAME = "mlops_dev.raw.fraud_model"
client = MlflowClient()

# INPUT_TABLE = f"{CATALOG}.{SCHEMA}.{PHASE}"
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.batch_predictions"

model_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri=f"models:/{MODEL_NAME}@Champion",
    result_type="double"
)

# df = spark.table(INPUT_TABLE)
df_test = spark.table(f"{CATALOG}.{SCHEMA}.test")
df_train = spark.table(f"{CATALOG}.{SCHEMA}.train")
df_val = spark.table(f"{CATALOG}.{SCHEMA}.val")
df = df_test.union(df_train).union(df_val)

feature_cols = [c for c in df.columns if c != "Class"]

pred_df = (
    df
    .withColumn("prediction", model_udf(struct(*feature_cols)))
    .withColumn("data_phase", "All")
    .withColumn("model_version", lit(
        client.get_model_version_by_alias(MODEL_NAME, "Champion").version
    ))
    .withColumn("inference_time", current_timestamp())
)

pred_df.write.mode("overwrite").format("delta").saveAsTable(OUTPUT_TABLE)

print(f"Batch inference completed: ")
