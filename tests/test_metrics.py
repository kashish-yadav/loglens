import pandas as pd
import pytest

from common import KEEP as K, REVIEW as R, SUPPRESS as S
from metrics import collapsible_mask, evaluate, impact, pct
from report_generator import record_csv, summary_markdown
from tests.helpers import classify, datasets, model
from data_validation import clean
from duplicate_detector import add_duplicate_info


def frame(rows):
    """rows of (class, level, group) -> minimal classified frame; group None = invalid record."""
    return pd.DataFrame([{"record_id": i + 1, "predicted_class": c, "level_clean": lv, "dup_group_id": g,
                          "is_valid": g is not None, "data_issues": ""} for i, (c, lv, g) in enumerate(rows)])


def test_pct_handles_zero_denominator():
    assert pct(5, 0) == 0.0 and pct(1, 4) == 25.0


def test_evaluate_hand_computed():
    m = evaluate([K, K, S, S, S, R], [K, S, S, S, R, R])
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["per_class"].set_index("Class").loc[K, "Recall"] == 0.5
    assert m["confusion"].values.tolist() == [[1, 1, 0], [0, 2, 1], [0, 0, 1]]
    assert m["false_suppressions"] == 1 and m["keep_records"] == 2


def test_evaluate_empty_input():
    assert evaluate([], [])["n"] == 0


def test_grouping_policy_and_before_after_counts():
    df = frame([(S, "INFO", "G1")] * 3 + [(S, "INFO", "G2")] + [(K, "ERROR", "G3")] * 3
               + [(R, "INFO", "G4")] * 2 + [(S, "INFO", None)])
    m = impact(df)
    assert (m["total"], m["valid"]) == (10, 9)
    assert m["eligible_records"] == 3 and m["collapsible_groups"] == 1      # only the repeated suppress group
    assert m["display_rows"] == 10 - 3 + 1 and m["display_reduction_pct"] == 20.0
    assert m["unique_groups"] == 4 and m["duplicate_occurrences"] == 5
    assert m["coverage_pct"] == pct(7, 9) and m["noise_reduction_pct"] == pct(3, 9)
    assert m["errors_collapsed"] == 0 and m["review"] == 2 and m["keep"] == 3


def test_mixed_group_is_not_collapsible():
    assert not collapsible_mask(frame([(S, "INFO", "G1"), (K, "INFO", "G1")])).any()


def test_time_saved_formula():
    m = impact(frame([(S, "INFO", "G1")] * 4 + [(K, "ERROR", "G2")]), seconds_per_record=60)
    assert m["minutes_before"] == 5.0 and m["minutes_saved"] == 3.0       # (4 eligible - 1 group) x 60s


def test_zero_records():
    m = impact(frame([(S, "INFO", "G1")]).iloc[0:0])
    assert m["total"] == 0 and m["coverage_pct"] == 0.0 and m["display_rows"] == 0


def test_impact_follows_filters_on_real_data():
    data = add_duplicate_info(model().predict(clean(datasets()["sample"])))
    keep_only = impact(data[data["predicted_class"] == K])
    assert keep_only["total"] == impact(data)["keep"] and keep_only["eligible_records"] == 0
    assert impact(data)["errors_collapsed"] == 0


def test_reports_reflect_the_dataset_they_are_given():
    a = add_duplicate_info(model().predict(clean(datasets()["sample"])))
    b = add_duplicate_info(model().predict(clean(datasets()["eval"])))
    ma, mb = summary_markdown(a, "a.csv"), summary_markdown(b, "b.csv")
    assert ma != mb and f"{len(b):,}" in mb and "b.csv" in mb
    assert "not reported" in ma and "estimate" in ma.lower()
    assert len(pd.read_csv(pd.io.common.BytesIO(record_csv(a)))) == len(a)


def test_report_with_model_eval_and_empty_data():
    a = add_duplicate_info(model().predict(clean(datasets()["sample"])))
    ev = evaluate([K, S], [K, S])
    assert "accuracy 1.000" in summary_markdown(a, "x", model_eval=ev)
    assert "Records: 0" in summary_markdown(a.iloc[0:0], "empty")
