"""
Promote the latest version of the registered model to Production
if the weighted F1 score from the most recent evaluation run exceeds
the required threshold.

Run:
    python scripts/promote.py
"""

import os

import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["promote"]


def get_latest_f1(client: MlflowClient, experiment_name: str) -> tuple[str, float]:
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise RuntimeError(f"Experiment not found: {experiment_name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if not runs or "f1_weighted" not in runs[0].data.metrics:
        raise RuntimeError("No completed evaluation run with f1_weighted found")

    run = runs[0]
    return run.info.run_id, run.data.metrics["f1_weighted"]


def promote(client: MlflowClient, model_name: str) -> str:
    versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        raise RuntimeError(f"No registered versions found for: {model_name}")

    latest_version = max(versions, key=lambda version: int(version.version))

    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=latest_version.version,
    )

    print(
        f"Model '{model_name}' version {latest_version.version} "
        "promoted to @production."
    )

    return latest_version.version


def main():
    params = load_params()
    threshold = float(params["f1_threshold"])

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "financial-news-sentiment",
    )
    model_name = os.getenv("MODEL_NAME", "finbert-financial-sentiment")

    client = MlflowClient(
        tracking_uri=tracking_uri,
        registry_uri=tracking_uri,
    )

    run_id, f1_score = get_latest_f1(client, experiment_name)

    print(f"Latest evaluation run: {run_id}")
    print(f"F1 weighted: {f1_score:.4f}")
    print(f"Required threshold: {threshold:.4f}")

    if f1_score >= threshold:
        promote(client, model_name)
    else:
        print(
            f"Model not promoted: F1 {f1_score:.4f} "
            f"is below threshold {threshold:.4f}."
        )


if __name__ == "__main__":
    main()
