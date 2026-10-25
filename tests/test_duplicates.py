from common import normalize_message
from tests.helpers import classify


def test_normalisation_masks_ids_but_keeps_status_codes():
    assert normalize_message("Payment declined for ORD-10293 code 51") == normalize_message("Payment declined for ORD-99999 code 51")
    assert normalize_message("Received HTTP 502") != normalize_message("Received HTTP 503")
    assert normalize_message(None) == ""


def test_repeated_message_forms_one_group():
    out = classify(*[{"message": f"Session created for user_{i}000"} for i in range(1, 5)])
    assert out["dup_group_id"].nunique() == 1
    assert out["occurrence_index"].tolist() == [1, 2, 3, 4] and out["is_duplicate"].sum() == 3
    assert len(out) == 4                                   # nothing removed


def test_different_level_service_or_status_are_separate_groups():
    out = classify({"message": "Operation done"}, {"message": "Operation done", "level": "ERROR"},
                   {"message": "Operation done", "service": "auth-service"},
                   {"message": "Operation done", "status_code": "500"})
    assert out["dup_group_id"].nunique() == 4


def test_similar_but_different_messages_not_merged():
    out = classify({"message": "Payment declined for card"}, {"message": "Payment declined for account"})
    assert out["dup_group_id"].nunique() == 2


def test_first_last_seen_and_invalid_records():
    out = classify({"message": "ping", "timestamp": "2026-10-12 12:00:00"},
                   {"message": "ping", "timestamp": "2026-10-12 08:00:00"}, {"message": ""})
    assert out["occurrence_index"].tolist()[:2] == [2, 1]
    assert (out["group_first_seen"].iloc[:2] == out["ts"].iloc[1]).all()
    assert out["dup_group_id"].isna().iloc[2]


def test_records_without_status_code_are_not_lumped_together():
    out = classify({"message": "alpha event"}, {"message": "beta event"}, {"message": "alpha event"})
    assert out["dup_group_id"].tolist() == ["G0001", "G0002", "G0001"]
