"""Shared constants and the one text-normalisation helper used everywhere."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SAMPLE_PATH = DATA_DIR / "sample_logs.csv"
TRAIN_PATH = DATA_DIR / "training_logs.csv"
EVAL_PATH = DATA_DIR / "evaluation_logs.csv"
MODEL_PATH = ROOT / "models" / "loglens_model.joblib"

KEEP, SUPPRESS, REVIEW = "Keep", "Suppress Candidate", "Review Required"
CLASSES = [KEEP, SUPPRESS, REVIEW]

REQUIRED_COLUMNS = ["timestamp", "level", "service", "message"]
INPUT_COLUMNS = REQUIRED_COLUMNS + ["request_id", "endpoint", "status_code", "latency_ms"]

LEVEL_ALIASES = {"DEBUG": "DEBUG", "TRACE": "DEBUG", "INFO": "INFO", "WARN": "WARN",
                 "WARNING": "WARN", "ERROR": "ERROR", "ERR": "ERROR", "CRITICAL": "CRITICAL",
                 "FATAL": "CRITICAL"}
LEVEL_ORDER = ["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL", "UNKNOWN"]
PROTECTED_LEVELS = ["ERROR", "CRITICAL"]          # never suppressible
KNOWN_STATUS = {200, 201, 202, 204, 301, 302, 304, 400, 401, 402, 403, 404, 408, 409, 422,
                429, 500, 501, 502, 503, 504}

_PATTERNS = [
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), "<uuid>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<ip>"),
    (re.compile(r"[$€£]\s?\d[\d,]*(?:\.\d+)?"), "<amt>"),
    (re.compile(r"\b\d+(?:\.\d+)?\s?(?:ms|s|sec|seconds?)\b"), "<dur>"),
    (re.compile(r"\b[a-z]{2,}[-_#][0-9a-z]*\d[0-9a-z]*\b"), "<id>"),   # ORD-10293, user_4412
    (re.compile(r"\b\d{4,}\b"), "<num>"),
]


def normalize_message(message) -> str:
    """Lower-case a message and mask volatile values (ids, IPs, durations, money).

    Status codes, error names, endpoints and service names are kept. The original
    message is never modified; this only builds a derived string.
    """
    if message is None or str(message).strip().lower() in ("", "nan"):
        return ""
    text = str(message).strip().lower()
    for pattern, token in _PATTERNS:
        text = pattern.sub(token, text)
    return re.sub(r"\s+", " ", text).strip()
