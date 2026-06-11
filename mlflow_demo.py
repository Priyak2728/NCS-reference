import mlflow

with mlflow.start_run() as run:

    mlflow.log_param(
        "algorithm",
        "LinearRegression"
    )

    mlflow.log_metric(
        "prediction_error",
        30
    )

    print("Run ID:", run.info.run_id)

print("Experiment Logged")