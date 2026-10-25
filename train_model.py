"""Train the classifier, evaluate it on the held-out set, and save the model.  Usage: python train_model.py"""
from common import EVAL_PATH, MODEL_PATH, TRAIN_PATH, normalize_message
from classifier import DEFAULT_THRESHOLD, LogClassifier
from data_validation import load_and_clean
from metrics import evaluate


def heldout_eval(clf: LogClassifier, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Evaluate a trained model on data/evaluation_logs.csv (ground-truth column `label`)."""
    ev = load_and_clean(EVAL_PATH)
    pred = clf.predict(ev, threshold)
    return evaluate(ev["label"], pred["predicted_class"])


def main():
    train, ev = load_and_clean(TRAIN_PATH), load_and_clean(EVAL_PATH)
    overlap = set(train["message"].map(normalize_message)) & set(ev["message"].map(normalize_message))
    if overlap:
        raise SystemExit(f"Train/evaluation leakage detected: {sorted(overlap)[:5]}")
    print("leakage check: no message appears in both training and evaluation data")
    clf = LogClassifier.train(train)
    clf.save(MODEL_PATH)
    print(f"trained on {len(train)} records, saved {MODEL_PATH.name}")
    m = heldout_eval(clf)
    print(f"held-out evaluation (n={m['n']}): accuracy {m['accuracy']:.3f}, macro F1 {m['macro_f1']:.3f}, "
          f"false suppressions {m['false_suppressions']} of {m['keep_records']} Keep records")
    print(m["per_class"].round(3).to_string(index=False))
    print("confusion matrix (rows = actual, columns = predicted):")
    print(m["confusion"].to_string())


if __name__ == "__main__":
    main()
