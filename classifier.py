"""Log classifier: TF-IDF + Logistic Regression, plus a few plain safety rules.

    message + level + service + status  ->  TF-IDF  ->  Logistic Regression
                                                    ->  rules (can only send a record to review)

Rules (a record becomes Review Required when any applies):
  1. the message is empty
  2. the model says "Suppress Candidate" but the record is ERROR/CRITICAL or HTTP 5xx
  3. the HTTP status code is not a recognised one
  4. model confidence is below the threshold (default 0.60)

`confidence` is the model's top class probability. It is NOT a calibrated probability.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from common import (CLASSES, KEEP, KNOWN_STATUS, MODEL_PATH, PROTECTED_LEVELS, REVIEW, SUPPRESS,
                    TRAIN_PATH, normalize_message)
from data_validation import clean, load_and_clean

DEFAULT_THRESHOLD = 0.60


def model_text(df: pd.DataFrame) -> pd.Series:
    """Text given to TF-IDF: level / service / status tokens + the normalised message."""
    status = df["status_clean"].astype("object").where(df["status_clean"].notna(), "none")
    return ("lvl_" + df["level_clean"].str.lower()
            + " svc_" + df["service"].str.strip().str.lower().str.replace("-", "_", regex=False)
            + " sc_" + status.astype(str) + " " + df["message"].map(normalize_message))


class LogClassifier:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    @classmethod
    def train(cls, df: pd.DataFrame, label_col: str = "label") -> "LogClassifier":
        """Fit on a labeled frame (labels must be Keep / Suppress Candidate / Review Required)."""
        data = df if "level_clean" in df.columns else clean(df)
        data = data[data["is_valid"]]
        labels = data[label_col].str.strip()
        if not set(labels) <= set(CLASSES) or labels.nunique() < 2:
            raise ValueError("Training labels must be a subset of the 3 classes and include 2+.")
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(token_pattern=r"[a-z0-9_]+(?:[-.][a-z0-9_]+)*",
                                      ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", LogisticRegression(C=10.0, max_iter=2000, class_weight="balanced")),
        ])
        return cls(pipe.fit(model_text(data), labels))

    def predict(self, df: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
        """Return a copy of df with model_prediction, confidence, predicted_class, reason."""
        out = (df if "level_clean" in df.columns else clean(df)).copy()
        if out.empty:
            for c in ("model_prediction", "confidence", "predicted_class", "reason"):
                out[c] = pd.Series(dtype="object")
            return out
        proba = self.pipeline.predict_proba(model_text(out))
        pred = pd.Series(np.array(self.pipeline.classes_)[proba.argmax(axis=1)], index=out.index)
        conf = pd.Series(proba.max(axis=1), index=out.index)

        status = out["status_clean"].fillna(0).astype(int)
        rules = [
            (~out["is_valid"], "message is empty"),
            ((pred == SUPPRESS) & (out["level_clean"].isin(PROTECTED_LEVELS) | (status >= 500)),
             "error-level or HTTP 5xx records are never suppressed"),
            (out["status_clean"].notna() & ~status.isin(KNOWN_STATUS), "unrecognised HTTP status code"),
            (conf < threshold, f"confidence below {threshold:.2f}"),
        ]
        why = "Model: " + pred + " (confidence " + conf.round(2).astype(str) + ")"
        applies = (pred != REVIEW) | ~out["is_valid"]     # nothing to change if already Review
        done = pd.Series(False, index=out.index)
        for mask, text in rules:                          # first matching rule gives the reason
            hit = mask & applies & ~done
            why = why.where(~hit, "Review Required: " + text + " (model said " + pred + ")")
            done |= hit
        final = pred.where(~done, REVIEW)
        out["model_prediction"] = pred.where(out["is_valid"], REVIEW)
        out["confidence"] = conf.where(out["is_valid"]).round(3)
        out["predicted_class"] = final
        out["reason"] = why.where(out["is_valid"], "Review Required: message is empty")
        return out

    def save(self, path: Path = MODEL_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "LogClassifier":
        """Load a saved model (only load files you trained yourself; joblib uses pickle)."""
        if not Path(path).exists():
            raise FileNotFoundError(f"No model at {path}. Run `python train_model.py`.")
        return cls(joblib.load(path))


def load_or_train(path: Path = MODEL_PATH) -> LogClassifier:
    """Load the saved model, or train one from data/training_logs.csv if none exists."""
    if Path(path).exists():
        return LogClassifier.load(path)
    clf = LogClassifier.train(load_and_clean(TRAIN_PATH))
    try:
        clf.save(path)
    except OSError:
        pass
    return clf
