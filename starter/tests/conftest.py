import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    """Single TestClient shared across the entire test session.

    The lifespan runs once — FinBERT is loaded once and held for all tests,
    avoiding repeated load/unload cycles that cause OOM on CI runners.
    """
    with TestClient(app) as c:
        yield c
