"""Build the two downloadable reports from whatever dataset is passed in (never cached sample numbers).

    python report_generator.py     # writes reports/impact_report.csv and impact_summary.md for the sample data
"""
import pandas as pd

from common import INPUT_COLUMNS, ROOT, SAMPLE_PATH
from metrics import FORMULAS, impact

REPORT_COLUMNS = INPUT_COLUMNS + ["record_id", "predicted_class", "confidence", "reason", "dup_group_id",
                                  "group_size", "occurrence_index", "is_duplicate", "data_issues"]


def record_csv(df: pd.DataFrame) -> bytes:
    """Record-level report: original fields + prediction, explanation and duplicate-group fields."""
    return df[[c for c in REPORT_COLUMNS if c in df.columns]].to_csv(index=False).encode("utf-8")


def summary_markdown(df: pd.DataFrame, dataset_name: str, seconds_per_record: float = 20.0,
                     model_eval: dict | None = None, synthetic: bool = False) -> str:
    """Markdown summary for the records in df."""
    m = impact(df, seconds_per_record)
    L = ["# LogLens Impact Summary", "",
         f"- Dataset: **{dataset_name}** ({'synthetic' if synthetic else 'user-provided'} data)",
         f"- Records: {m['total']:,} ({m['valid']:,} valid; empty-message records are sent to review)", "",
         "## Classification", "", "| Class | Records |", "|---|---:|",
         f"| Keep | {m['keep']:,} |", f"| Suppress Candidate | {m['suppress']:,} |",
         f"| Review Required | {m['review']:,} |", "",
         "## Duplicates and grouping", "",
         "These counts overlap; do not add them together.", "",
         f"- Unique message groups: {m['unique_groups']:,}",
         f"- Duplicate occurrences beyond the first: {m['duplicate_occurrences']:,} ({m['duplicate_rate_pct']}%)",
         f"- Records eligible for grouping: {m['eligible_records']:,} ({m['noise_reduction_pct']}% potential noise reduction)",
         f"- Rows when collapsed: {m['display_rows']:,} instead of {m['total']:,} ({m['display_reduction_pct']}% fewer)",
         f"- ERROR/CRITICAL records collapsed (must be 0): {m['errors_collapsed']}",
         "- No records are deleted; the CSV report keeps every row with its group id.", "",
         f"## Classification coverage: {m['coverage_pct']}%", "",
         "## Estimated review time (ESTIMATE)", "",
         f"Assumption: {seconds_per_record:g} seconds to inspect one record.",
         f"Manual review before: {m['minutes_before']} min. Estimated saved by collapsing repeats: "
         f"**{m['minutes_saved']} min**. This is an estimate from the assumption above, not a measured result.", "",
         "## Model quality", ""]
    if model_eval and model_eval.get("n"):
        L += [f"Held-out labeled data, n = {model_eval['n']}: accuracy {model_eval['accuracy']:.3f}, "
              f"macro F1 {model_eval['macro_f1']:.3f}; false suppressions "
              f"{model_eval['false_suppressions']} of {model_eval['keep_records']} Keep records.", "",
              "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
        L += [f"| {r.Class} | {r.Precision:.3f} | {r.Recall:.3f} | {r.F1:.3f} | {r.Support} |"
              for r in model_eval["per_class"].itertuples()]
    else:
        L.append("No ground-truth labels available, so accuracy / precision / recall / F1 are not reported.")
    L += ["", "## Formulas", ""] + [f"- **{k}:** {v}" for k, v in FORMULAS.items()]
    issues = int((df["data_issues"] != "").sum()) if len(df) else 0
    L += ["", "## Limitations", "",
          f"- {issues} record(s) have data-quality issues (kept and still classified).",
          "- Synthetic, small data: results show the pipeline works, not real-world accuracy.",
          "- Confidence is a model score, not a calibrated probability."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    from classifier import load_or_train
    from data_validation import load_and_clean
    from duplicate_detector import add_duplicate_info
    from train_model import heldout_eval

    clf = load_or_train()
    data = add_duplicate_info(clf.predict(load_and_clean(SAMPLE_PATH)))
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "impact_report.csv").write_bytes(record_csv(data))
    (out / "impact_summary.md").write_text(
        summary_markdown(data, SAMPLE_PATH.name, model_eval=heldout_eval(clf), synthetic=True), encoding="utf-8")
    print("wrote reports/impact_report.csv and reports/impact_summary.md")
