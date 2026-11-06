"""
Sentiment Prediction API.

Endpoints:
    GET  /health          — service health status
    POST /predict         — single headline sentiment
    POST /predict/batch   — batch headline sentiment
    GET  /metrics         — Prometheus metrics
"""

import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.utils import load_classifier

# TODO (Task 6)
# Configure the Python logger using basicConfig.


# TODO (Task 6)
# Implement a structured logging helper that emits each log entry as a JSON
# object containing at least a timestamp, level, and message field.
def log(level: str, message: str, **kwargs) -> None:
    pass


# TODO (Task 6)
# Define the following Prometheus metrics:
# 1. A Counter for the total number of prediction requests, labelled by sentiment.
# 2. A Histogram for prediction latency in milliseconds.
# 3. A Counter for the total number of prediction errors.
# Then instrument run_predictions() to update each metric accordingly.
# Documentation: https://prometheus.io/docs/concepts/metric_types/
PREDICTION_REQUESTS = None
PREDICTION_LATENCY = None
PREDICTION_ERRORS = None

load_dotenv()

classifiers = {}


class PredictRequest(BaseModel):
    text: str


class PredictBatchRequest(BaseModel):
    texts: list[str]


class PredictionResult(BaseModel):
    text: str
    sentiment: str
    confidence: float
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    log("INFO", "Loading model...")
    classifiers["sentiment"] = load_classifier()
    log("INFO", "Model loaded successfully")
    yield
    classifiers.clear()
    log("INFO", "Model unloaded")


app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)


def run_predictions(texts: list[str]) -> list[PredictionResult]:
    try:
        if any(len(text) > 2000 for text in texts):
            log("WARNING", "Input text exceeds 2000 characters")

        start_time = time.perf_counter()

        raw_predictions = classifiers["sentiment"](
            texts,
            truncation=True,
            max_length=512,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # TODO (Task 6): Observe latency_ms on PREDICTION_LATENCY and increment
        # PREDICTION_REQUESTS (labelled by sentiment) for each prediction.

        # TODO (Task 6): Log each prediction using log() with sentiment, confidence,
        # and latency_ms fields.

        return [
            PredictionResult(
                text=text,
                sentiment=str(prediction["label"]).lower(),
                confidence=float(prediction["score"]),
                latency_ms=latency_ms,
            )
            for text, prediction in zip(texts, raw_predictions)
        ]
    except Exception as e:
        # TODO (Task 6): Increment PREDICTION_ERRORS, log the error using log(),
        # then re-raise.
        raise e


# TODO (Task 6): Implement the /metrics endpoint.
# Return the Prometheus metrics in the correct format using generate_latest()
# and CONTENT_TYPE_LATEST.
# Documentation: https://prometheus.io/docs/instrumenting/exposition_formats/
@app.get("/metrics")
def metrics():
    raise NotImplementedError


@app.get("/health")
def health():
    if "sentiment" not in classifiers:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResult)
def predict(request: PredictRequest):
    if "sentiment" not in classifiers:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    if not request.text.strip():
        raise HTTPException(
            status_code=422,
            detail="Text cannot be empty",
        )

    return run_predictions([request.text])[0]


@app.post("/predict/batch", response_model=list[PredictionResult])
def predict_batch(request: PredictBatchRequest):
    if "sentiment" not in classifiers:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    if not request.texts:
        raise HTTPException(
            status_code=422,
            detail="Texts list cannot be empty",
        )

    if any(not text.strip() for text in request.texts):
        raise HTTPException(
            status_code=422,
            detail="Texts cannot contain empty values",
        )

    return run_predictions(request.texts)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
    )
