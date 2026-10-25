from classifier import LogClassifier
from common import KEEP, REVIEW, SUPPRESS
from data_validation import clean
from generate_logs import build
from tests.helpers import classify, datasets, logs, model


def test_critical_events_are_kept():
    out = classify({"level": "CRITICAL", "service": "database-service", "message": "Database connection refused by host db-primary"},
                   {"level": "ERROR", "service": "payment-service", "status_code": "402",
                    "message": "Payment declined for order ORD-12345: issuer response code 51"},
                   {"level": "ERROR", "status_code": "504", "message": "Upstream timeout after 9000ms calling inventory-service"})
    assert (out["predicted_class"] == KEEP).all()


def test_routine_info_is_suppress_candidate():
    out = classify({"message": "Health check passed: payment-service responded 200 in 12ms", "status_code": "200"})
    assert out.loc[0, "predicted_class"] == SUPPRESS


def test_repeated_errors_are_never_suppressed():
    rows = [{"level": "ERROR", "message": "Health check OK", "status_code": "200"}] * 20
    rows += [{"level": "CRITICAL", "message": "Heartbeat received from api-gateway"}] * 20
    rows += [{"level": "INFO", "status_code": "503", "message": "Health check OK"}] * 10
    assert (classify(*rows)["predicted_class"] != SUPPRESS).all()


def test_no_error_level_record_is_suppressed_in_sample():
    out = model().predict(clean(datasets()["sample"]))
    assert not ((out["level_clean"].isin(["ERROR", "CRITICAL"])) & (out["predicted_class"] == SUPPRESS)).any()


def test_unknown_status_code_goes_to_review():
    out = classify({"status_code": "418"}, {"status_code": "999"})
    assert (out["predicted_class"] == REVIEW).all()


def test_unfamiliar_message_goes_to_review_at_high_threshold():
    out = classify({"message": "Quantum flux realigned in sector seven"}, threshold=1.0)
    assert out.loc[0, "predicted_class"] == REVIEW and "confidence below" in out.loc[0, "reason"]


def test_empty_message_goes_to_review_and_missing_fields_do_not_crash():
    out = classify({"message": ""}, {"timestamp": "bad", "message": "Request routed through fallback path"})
    assert out.loc[0, "predicted_class"] == REVIEW and "empty" in out.loc[0, "reason"]
    assert out.loc[1, "predicted_class"] in (KEEP, SUPPRESS, REVIEW)


def test_every_record_gets_one_class_and_a_reason_and_message_is_unchanged():
    sample = clean(datasets()["sample"])
    out = model().predict(sample)
    assert len(out) == len(sample) and out["predicted_class"].isin([KEEP, SUPPRESS, REVIEW]).all()
    assert (out["reason"].str.len() > 0).all() and out["message"].tolist() == sample["message"].tolist()


def test_predictions_are_consistent_and_threshold_is_monotonic():
    sample = clean(datasets()["sample"])
    a, b = model().predict(sample), model().predict(sample)
    assert a["predicted_class"].tolist() == b["predicted_class"].tolist()
    review = [(model().predict(sample, t)["predicted_class"] == REVIEW).sum() for t in (0.3, 0.6, 0.9)]
    assert review == sorted(review)


def test_empty_frame_is_handled():
    assert model().predict(clean(logs({})).iloc[0:0]).empty


def test_save_and_load_roundtrip(tmp_path):
    model().save(tmp_path / "m.joblib")
    sample = clean(datasets()["sample"]).head(50)
    assert LogClassifier.load(tmp_path / "m.joblib").predict(sample)["predicted_class"].tolist() == \
        model().predict(sample)["predicted_class"].tolist()


def test_train_and_eval_templates_do_not_overlap():
    sets = build()
    from common import normalize_message
    assert not (set(sets["train"]["message"].map(normalize_message)) & set(sets["eval"]["message"].map(normalize_message)))
