"""
Model quality check (drift) using deepchecks

Run:
    python scripts/run_deepchecks.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import yaml
from app.utils import load_classifier
from deepchecks.nlp import TextData
from deepchecks.nlp.checks import PredictionDrift, PropertyDrift
from dotenv import load_dotenv

load_dotenv()


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["deepchecks"]


def run_predictions(classifier, texts: list[str]) -> list[str]:
    results = classifier(texts, batch_size=32)
    return [r["label"] for r in results]


def main():
    params = load_params()
    property_drift_threshold = params["property_drift_threshold"]
    prediction_drift_threshold = params["prediction_drift_threshold"]

    print("Loading production model...")

    classifier = load_classifier()

    stream_df = pd.read_csv("data/stream.csv")
    test_df = pd.read_csv("data/test.csv")

    stream_texts = stream_df["text"].tolist()
    test_texts = test_df["text"].tolist()

    stream_dataset = TextData(
        raw_text=stream_texts,
        task_type="text_classification",
    )

    test_dataset = TextData(
        raw_text=test_texts,
        task_type="text_classification",
    )

    stream_dataset.calculate_builtin_properties(
        include_properties=["Sentiment"]
    )
    test_dataset.calculate_builtin_properties(
        include_properties=["Sentiment"]
    )

    property_result = PropertyDrift(properties=["Sentiment"]).run(
        train_dataset=test_dataset,
        test_dataset=stream_dataset,
        with_display=False,
    )

    property_drift_score = property_result.value["Sentiment"]["Drift score"]

    print(f"Property drift score: {property_drift_score:.4f}")

    drift_failed = property_drift_score > property_drift_threshold

    if drift_failed:
        print(
            f"Property drift exceeded threshold: "
            f"{property_drift_score:.4f} > {property_drift_threshold:.4f}"
        )

    test_predictions = run_predictions(classifier, test_texts)
    stream_predictions = run_predictions(classifier, stream_texts)

    prediction_result = PredictionDrift(drift_mode="prediction").run(
        train_dataset=test_dataset,
        test_dataset=stream_dataset,
        train_predictions=test_predictions,
        test_predictions=stream_predictions,
        model_classes=["negative", "neutral", "positive"],
        with_display=False,
    )

    prediction_drift_score = prediction_result.value["Drift score"]

    print(f"Prediction drift score: {prediction_drift_score:.4f}")

    if prediction_drift_score > prediction_drift_threshold:
        drift_failed = True
        print(
            f"Prediction drift exceeded threshold: "
            f"{prediction_drift_score:.4f} > {prediction_drift_threshold:.4f}"
        )

    if drift_failed:
        sys.exit(1)

    print("Deepchecks drift gate passed.")


if __name__ == "__main__":
    main()
