# LogLens Impact Summary

- Dataset: **sample_logs.csv** (synthetic data)
- Records: 309 (308 valid; empty-message records are sent to review)

## Classification

| Class | Records |
|---|---:|
| Keep | 101 |
| Suppress Candidate | 142 |
| Review Required | 66 |

## Duplicates and grouping

These counts overlap; do not add them together.

- Unique message groups: 228
- Duplicate occurrences beyond the first: 80 (26.0%)
- Records eligible for grouping: 76 (24.7% potential noise reduction)
- Rows when collapsed: 263 instead of 309 (14.9% fewer)
- ERROR/CRITICAL records collapsed (must be 0): 0
- No records are deleted; the CSV report keeps every row with its group id.

## Classification coverage: 78.9%

## Estimated review time (ESTIMATE)

Assumption: 20 seconds to inspect one record.
Manual review before: 102.7 min. Estimated saved by collapsing repeats: **15.3 min**. This is an estimate from the assumption above, not a measured result.

## Model quality

Held-out labeled data, n = 151: accuracy 0.914, macro F1 0.894; false suppressions 0 of 50 Keep records.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Keep | 1.000 | 0.860 | 0.925 | 50 |
| Suppress Candidate | 0.986 | 0.932 | 0.958 | 74 |
| Review Required | 0.684 | 0.963 | 0.800 | 27 |

## Formulas

- **Classification coverage:** (Keep + Suppress Candidate) / valid records x 100  (Review Required is not 'automatic')
- **Potential noise reduction:** records in collapsible groups / valid records x 100
- **Duplicate rate:** (valid records - unique groups) / valid records x 100
- **Display-row reduction:** (raw rows - display rows) / raw rows x 100, where display rows = raw - eligible + collapsible groups
- **Estimated time saved:** (eligible records - collapsible groups) x seconds per record  -- an ESTIMATE from your assumption, not a measurement

## Limitations

- 5 record(s) have data-quality issues (kept and still classified).
- Synthetic, small data: results show the pipeline works, not real-world accuracy.
- Confidence is a model score, not a calibrated probability.
