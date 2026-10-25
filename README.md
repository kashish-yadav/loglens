# LogLens

A small **proof of concept**: classify application log messages, group repeated ones without deleting
anything, flag uncertain ones for a human, and show how much that might help in a Streamlit dashboard.

> All data is **synthetic** (no real logs or company information). Nothing here is deployed or measured
> in production. "Time saved" is an **estimate** from one editable assumption.

## What it does
* **Classifies** each log as **Keep** (important), **Suppress Candidate** (routine) or **Review Required** (unsure),
  using TF-IDF + Logistic Regression, and shows a one-line reason per record.
* **Safety rules:** ERROR/CRITICAL or HTTP 5xx records are never suppressed; empty messages, unknown HTTP status
  codes and low-confidence predictions go to Review Required.
* **Finds duplicates** (same service, severity, status and message after masking ids/numbers). Records are never
  removed. Only groups of 2+ records that are *all* Suppress Candidate collapse into one display row, so repeated
  errors always stay visible.
* **Dashboard** (3 tabs): Overview charts, Log Explorer (filters, search, CSV download), Impact & Efficiency
  (coverage, noise reduction, duplicate rate, before/after, model quality, downloadable reports).
* **Upload your own CSV** (required: `timestamp, level, service, message`; optional: `request_id, endpoint,
  status_code, latency_ms`). Bad values are flagged, not dropped.

## Run it
```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate      Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python generate_logs.py      # optional: data is already included
python train_model.py        # trains + prints held-out results
streamlit run app.py
pytest -q
```
If you skip training, the app trains the model itself on first start.

## Files
```
app.py                dashboard             classifier.py        model + safety rules
data_validation.py    CSV loading/cleaning   duplicate_detector.py duplicate groups
metrics.py            quality + impact maths report_generator.py  CSV / Markdown reports
generate_logs.py      synthetic data         train_model.py       train + evaluate
common.py             constants, text masking tests/               34 tests
data/                 sample, training and evaluation CSVs
```

## Data
Synthetic, seed 42: `sample_logs.csv` (309 rows for the dashboard; includes 10 exact duplicates and 5 deliberately
bad values), `training_logs.csv` (299), `evaluation_logs.csv` (151). Each event type has several wordings; training and
evaluation use **different wordings** so the evaluation tests unseen phrasing. `train_model.py` stops if any message
appears in both.

## Results (one run; reproduce with `python train_model.py`)
Held-out synthetic evaluation, n = 151:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Keep | 1.000 | 0.860 | 0.925 | 50 |
| Suppress Candidate | 0.986 | 0.932 | 0.958 | 74 |
| Review Required | 0.684 | 0.963 | 0.800 | 27 |

Accuracy 0.914, macro-F1 0.894. **0 of 50 Keep records wrongly marked suppressible.** All errors are
conservative: 7 Keep and 5 Suppress records were sent to review; 1 Review record was called routine.

Dashboard numbers on the sample (default 20 s/record): 78.9% classification coverage, 24.7% potential noise
reduction (76 records), 26.0% duplicate rate, 309 rows -> 263 when collapsed, ~15 min estimated saved of ~103.
0 ERROR/CRITICAL records collapsed. See `reports/impact_summary.md`.

These come from small, template-generated data: they show the pipeline works, **not** real-world accuracy.

## Limits
Synthetic data and one train/eval split; confidence is a model score, not a calibrated probability; duplicate
matching is text-based (a blank status code puts a record in its own group); no review queue or model retraining
(Review Required records can be filtered in the Explorer); don't upload sensitive logs to a shared deployment.

## Verification note
Built in an offline environment: data generation, training, evaluation and report generation were run for real; the 34 tests
passed using a minimal stand-in for `pytest`, and the dashboard code was executed against stand-ins for Streamlit and Plotly.
**Real `pytest -q` and `streamlit run app.py` have not been run by the author - please do so.**

MIT License.
