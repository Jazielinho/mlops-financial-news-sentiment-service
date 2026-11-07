"""
Production monitoring stream for FinBERT sentiment API.

Reads headlines from data/stream.csv, sends them to the /predict endpoint,
and logs aggregated metrics to MLflow every WINDOW_SIZE predictions.

Each observation window logs:
    - Sentiment distribution (% positive, % negative, % neutral)
    - Average confidence score
    - Average latency (ms)

Run from the project root:
    python monitoring/stream.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "finbert-evaluation")
WINDOW_SIZE = 50  # number of predictions per observation window
SLEEP_MS = 100  # delay between requests to simulate real traffic (ms)


def predict(text: str) -> dict:
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": text},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def log_window(window: list[dict], window_idx: int) -> None:
    """Log aggregated metrics for one observation window to MLflow."""
    total = len(window)

    metrics = {
        "pct_positive": 100
        * sum(item["sentiment"] == "positive" for item in window)
        / total,
        "pct_negative": 100
        * sum(item["sentiment"] == "negative" for item in window)
        / total,
        "pct_neutral": 100
        * sum(item["sentiment"] == "neutral" for item in window)
        / total,
        "avg_confidence": sum(item["confidence"] for item in window) / total,
        "avg_latency_ms": sum(item["latency_ms"] for item in window) / total,
    }

    mlflow.log_metrics(metrics, step=window_idx)
    print(f"Window {window_idx}: {metrics}")


def main():
    stream_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "stream.csv",
    )
    headlines = pd.read_csv(stream_path)["text"].dropna().astype(str).tolist()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    window = []
    window_idx = 0

    with mlflow.start_run(run_name="production-monitoring"):
        for text in headlines:
            try:
                window.append(predict(text))
            except requests.RequestException as error:
                print(f"Prediction failed: {error}")
                continue

            if len(window) == WINDOW_SIZE:
                window_idx += 1
                log_window(window, window_idx)
                window.clear()

            time.sleep(SLEEP_MS / 1000)

        if window:
            window_idx += 1
            log_window(window, window_idx)


if __name__ == "__main__":
    main()
