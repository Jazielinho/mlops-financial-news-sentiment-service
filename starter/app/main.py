"""
Sentiment Prediction API.

Endpoints:
    GET  /health          — service health status
    POST /predict         — single headline sentiment
    POST /predict/batch   — batch headline sentiment
    GET  /metrics         — Prometheus metrics
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from app.utils import load_classifier

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("finbert-api")


def log(level: str, message: str, **kwargs) -> None:
    normalized_level = level.upper()
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "level": normalized_level,
        "message": message,
        **kwargs,
    }
    logger.log(
        getattr(logging, normalized_level, logging.INFO),
        json.dumps(entry, default=str),
    )


PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total number of predictions by sentiment",
    ["sentiment"],
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_ms",
    "Prediction latency in milliseconds",
    buckets=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
)
PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total number of prediction errors",
)

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
    if any(len(text) > 2000 for text in texts):
        log("WARNING", "Input text exceeds 2000 characters")

    try:
        start_time = time.perf_counter()

        raw_predictions = classifiers["sentiment"](
            texts,
            truncation=True,
            max_length=512,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000
        PREDICTION_LATENCY.observe(latency_ms)

        results = []
        for text, prediction in zip(texts, raw_predictions):
            result = PredictionResult(
                text=text,
                sentiment=str(prediction["label"]).lower(),
                confidence=float(prediction["score"]),
                latency_ms=latency_ms,
            )
            PREDICTION_REQUESTS.labels(sentiment=result.sentiment).inc()
            log(
                "INFO",
                "Prediction completed",
                sentiment=result.sentiment,
                confidence=result.confidence,
                latency_ms=result.latency_ms,
            )
            results.append(result)

        return results
    except Exception as error:
        PREDICTION_ERRORS.inc()
        log("ERROR", "Prediction failed", error=str(error))
        raise


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


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
        port=int(os.getenv("API_PORT", "8000")),
    )
