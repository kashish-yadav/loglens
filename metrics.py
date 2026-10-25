"""Model-quality metrics and the Impact & Efficiency numbers.

Kept separate on purpose:
  * model quality (accuracy / precision / recall / F1) needs ground-truth labels
  * operational numbers (coverage, duplicates, grouping) say nothing about correctness
  * time saved is an ESTIMATE from one assumption (seconds per record)

Grouping policy: a duplicate group is collapsed into ONE display row only if it has 2+ records
and ALL of them are Suppress Candidate. Repeated errors are never collapsed. Nothing is deleted.
"""
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from common import CLASSES, KEEP, PROTECTED_LEVELS, REVIEW, SUPPRESS


def pct(part: float, whole: float) -> float:
    """Percentage rounded to 1 dp; 0.0 when the denominator is zero."""
    return round(100.0 * part / whole, 1) if whole else 0.0


def evaluate(y_true, y_pred) -> dict:
    """Accuracy, per-class precision/recall/F1, macro F1, confusion matrix, false suppressions."""
    yt, yp = pd.Series(list(y_true)), pd.Series(list(y_pred))
    if yt.empty:
        return {"n": 0}
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=CLASSES, zero_division=0)
    keep_n = int((yt == KEEP).sum())
    wrong = int(((yt == KEEP) & (yp == SUPPRESS)).sum())
    return {
        "n": len(yt), "accuracy": float((yt == yp).mean()), "macro_f1": float(f.mean()),
        "per_class": pd.DataFrame({"Class": CLASSES, "Precision": p, "Recall": r, "F1": f,
                                   "Support": s}),
        "confusion": pd.DataFrame(confusion_matrix(yt, yp, labels=CLASSES), index=CLASSES, columns=CLASSES),
        "false_suppressions": wrong, "keep_records": keep_n,
    }


def collapsible_mask(df: pd.DataFrame) -> pd.Series:
    """True for records in a duplicate group of 2+ whose members are all Suppress Candidate.

    Computed on the frame given, so it follows any filters applied to it.
    """
    mask = pd.Series(False, index=df.index)
    valid = df[df["is_valid"]]
    if valid.empty:
        return mask
    size = valid.groupby("dup_group_id")["record_id"].transform("size")
    all_suppress = (valid["predicted_class"] == SUPPRESS).groupby(valid["dup_group_id"]).transform("all")
    mask.loc[valid.index] = ((size >= 2) & all_suppress).to_numpy()
    return mask


def impact(df: pd.DataFrame, seconds_per_record: float = 20.0) -> dict:
    """Impact & Efficiency numbers for the records in df (pass the filtered frame)."""
    total = len(df)
    valid = df[df["is_valid"]] if total else df
    n_valid = len(valid)
    counts = valid["predicted_class"].value_counts() if n_valid else pd.Series(dtype=int)
    keep, suppress = int(counts.get(KEEP, 0)), int(counts.get(SUPPRESS, 0))
    groups = int(valid["dup_group_id"].nunique()) if n_valid else 0
    mask = collapsible_mask(df) if total else pd.Series(dtype=bool)
    eligible = int(mask.sum())
    collapsed = int(df.loc[mask, "dup_group_id"].nunique()) if eligible else 0
    display = total - eligible + collapsed
    errors = valid["level_clean"].isin(PROTECTED_LEVELS) if n_valid else pd.Series(dtype=bool)
    return {
        "total": total, "valid": n_valid, "keep": keep, "suppress": suppress,
        "review": int((df["predicted_class"] == REVIEW).sum()) if total else 0,
        "coverage_pct": pct(keep + suppress, n_valid),
        "unique_groups": groups, "duplicate_occurrences": n_valid - groups,
        "duplicate_rate_pct": pct(n_valid - groups, n_valid),
        "eligible_records": eligible, "collapsible_groups": collapsed,
        "noise_reduction_pct": pct(eligible, n_valid),
        "display_rows": display, "display_reduction_pct": pct(total - display, total),
        "error_records": int(errors.sum()) if n_valid else 0,
        "errors_collapsed": int((mask & df["level_clean"].isin(PROTECTED_LEVELS)).sum()) if eligible else 0,
        "seconds_per_record": seconds_per_record,
        "minutes_before": round(n_valid * seconds_per_record / 60, 1),
        "minutes_saved": round((eligible - collapsed) * seconds_per_record / 60, 1),
    }


FORMULAS = {
    "Classification coverage": "(Keep + Suppress Candidate) / valid records x 100  (Review Required is not 'automatic')",
    "Potential noise reduction": "records in collapsible groups / valid records x 100",
    "Duplicate rate": "(valid records - unique groups) / valid records x 100",
    "Display-row reduction": "(raw rows - display rows) / raw rows x 100, where display rows = raw - eligible + collapsible groups",
    "Estimated time saved": "(eligible records - collapsible groups) x seconds per record  -- an ESTIMATE from your assumption, not a measurement",
}
