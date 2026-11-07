"""
Integration tests for the Sentiment Analysis API.
"""

import json
import logging

import pytest
from fastapi.testclient import TestClient

from app.main import app, log


@pytest.mark.parametrize(
    "text",
    [
        "Markets rise after strong earnings",
        "Shares fall after disappointing results",
        "The company published its quarterly report",
    ],
)
def test_predict_returns_valid_response(client: TestClient, text: str):
    response = client.post("/predict", json={"text": text})

    assert response.status_code == 200

    result = response.json()

    assert set(result) == {
        "text",
        "sentiment",
        "confidence",
        "latency_ms",
    }
    assert result["text"] == text
    assert result["sentiment"] in {"positive", "negative", "neutral"}
    assert 0 <= result["confidence"] <= 1
    assert result["latency_ms"] >= 0


def test_predict_batch(client: TestClient):
    texts = [
        "Markets rise after strong earnings",
        "Shares fall after disappointing results",
    ]

    response = client.post(
        "/predict/batch",
        json={"texts": texts},
    )

    assert response.status_code == 200

    results = response.json()

    assert len(results) == len(texts)

    for text, result in zip(texts, results):
        assert result["text"] == text
        assert set(result) == {
            "text",
            "sentiment",
            "confidence",
            "latency_ms",
        }
        assert result["sentiment"] in {"positive", "negative", "neutral"}
        assert 0 <= result["confidence"] <= 1
        assert result["latency_ms"] >= 0


def test_predict_batch_empty_list_returns_422(client: TestClient):
    response = client.post(
        "/predict/batch",
        json={"texts": []},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Texts list cannot be empty"}


def test_metrics_exposes_prediction_metrics():
    response = TestClient(app).get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "prediction_requests_total" in response.text
    assert "prediction_latency_ms" in response.text
    assert "prediction_errors_total" in response.text


def test_log_emits_structured_json(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.INFO, logger="finbert-api"):
        log("INFO", "Test log", sentiment="positive")

    entry = json.loads(caplog.records[-1].message)

    assert entry["level"] == "INFO"
    assert entry["message"] == "Test log"
    assert entry["sentiment"] == "positive"
    assert "timestamp" in entry
