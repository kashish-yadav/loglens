"""Duplicate detection. Independent of the classifier: it only decides which records
are the same event repeated. Nothing is removed; every record keeps its group info."""
import pandas as pd

from common import normalize_message


def add_duplicate_info(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized_message, dup_group_id, group_size, occurrence_index,
    is_duplicate, group_first_seen, group_last_seen.

    Records are the same group when service, severity, status code and the
    normalised message match. Invalid (empty-message) records are not grouped.
    """
    out = df.copy()
    out["normalized_message"] = out["message"].map(normalize_message)
    valid = out["is_valid"].astype(bool)
    status = out["status_clean"].astype("object").where(out["status_clean"].notna(), "none").astype(str)
    key = (out["service"].str.strip() + "|" + out["level_clean"] + "|" + status + "|"
           + out["normalized_message"])
    codes, _ = pd.factorize(key[valid])
    out["dup_group_id"] = pd.NA
    out.loc[valid, "dup_group_id"] = [f"G{c + 1:04d}" for c in codes]

    v = out[valid].sort_values(["ts", "record_id"], na_position="last")
    out["occurrence_index"] = 1
    out.loc[v.index, "occurrence_index"] = v.groupby("dup_group_id").cumcount() + 1
    out["group_size"] = 1
    out.loc[valid, "group_size"] = out[valid].groupby("dup_group_id")["record_id"].transform("size")
    out["is_duplicate"] = (out["occurrence_index"] > 1) & valid
    out["group_first_seen"] = out.groupby("dup_group_id")["ts"].transform("min")
    out["group_last_seen"] = out.groupby("dup_group_id")["ts"].transform("max")
    return out
