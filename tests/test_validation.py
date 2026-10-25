import pandas as pd
import pytest

from data_validation import ValidationError, clean, load_csv
from tests.helpers import logs


def test_empty_and_header_only_files_rejected():
    for data in (b"", b"  \n"):
        with pytest.raises(ValidationError):
            load_csv(data)
    with pytest.raises(ValidationError):
        load_csv(b"timestamp,level,service,message\n")


def test_non_csv_name_rejected():
    with pytest.raises(ValidationError):
        load_csv(b"a,b\n1,2\n", filename="logs.xlsx")


def test_missing_required_column_rejected():
    with pytest.raises(ValidationError):
        clean(pd.DataFrame({"message": ["x"], "level": ["INFO"]}))


def test_missing_optional_columns_are_added():
    df = clean(pd.DataFrame({"timestamp": ["2026-10-12 10:00:00"], "level": ["info"],
                             "service": ["s"], "message": ["m"]}))
    assert {"status_code", "latency_ms", "request_id", "endpoint"} <= set(df.columns)


def test_bad_values_are_flagged_but_records_kept():
    df = clean(logs({"timestamp": "garbage"}, {"message": "  "}, {"status_code": "abc"}, {"level": "VERBOSE"}, {}))
    assert len(df) == 5
    assert "bad_timestamp" in df.loc[0, "data_issues"]
    assert not df.loc[1, "is_valid"] and "empty_message" in df.loc[1, "data_issues"]
    assert "bad_status_code" in df.loc[2, "data_issues"] and df.loc[3, "level_clean"] == "UNKNOWN"
    assert df.loc[4, "data_issues"] == ""


def test_original_columns_untouched():
    df = clean(logs({"message": "Keep ME  as-is", "level": "warning"}))
    assert df.loc[0, "message"] == "Keep ME  as-is" and df.loc[0, "level_clean"] == "WARN"
