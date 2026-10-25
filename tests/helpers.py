"""Shared test helpers; the model is trained once per test run."""
from functools import lru_cache

import pandas as pd

from classifier import LogClassifier
from data_validation import clean
from duplicate_detector import add_duplicate_info
from generate_logs import build


@lru_cache(maxsize=1)
def datasets():
    return build()


@lru_cache(maxsize=1)
def model():
    return LogClassifier.train(clean(datasets()["train"]))


def logs(*rows):
    """Raw log frame from partial row dicts."""
    base = {"timestamp": "2026-10-12 10:00:00", "level": "INFO", "service": "api-gateway",
            "message": "Health check OK", "request_id": "", "endpoint": "", "status_code": "", "latency_ms": ""}
    return pd.DataFrame([{**base, **r} for r in rows])


def classify(*rows, threshold=0.6):
    return add_duplicate_info(model().predict(clean(logs(*rows)), threshold))
