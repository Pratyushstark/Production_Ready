# Databricks notebook source
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "mlops_dev.raw.fraud_model"
client = MlflowClient()

staging = client.get_model_version_by_alias(
    name=MODEL_NAME,
    alias="staging"
)

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="Champion",
    version=staging.version
)

print(f"Champion set to version {staging.version}")
