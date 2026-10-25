"""LogLens dashboard.  Run:  streamlit run app.py"""
import pandas as pd
import plotly.express as px
import streamlit as st

from classifier import DEFAULT_THRESHOLD, load_or_train
from common import CLASSES, KEEP, LEVEL_ORDER, PROTECTED_LEVELS, REVIEW, SAMPLE_PATH, SUPPRESS
from data_validation import ValidationError, load_and_clean
from duplicate_detector import add_duplicate_info
from metrics import FORMULAS, collapsible_mask, impact
from report_generator import record_csv, summary_markdown
from train_model import heldout_eval

st.set_page_config(page_title="LogLens", page_icon="🔎", layout="wide")
COLORS = {KEEP: "#2f6fed", SUPPRESS: "#9aa5b1", REVIEW: "#f5a623"}


def show_df(data, **kw):
    try:
        st.dataframe(data, width="stretch", hide_index=True, **kw)
    except TypeError:                                   # older Streamlit versions
        st.dataframe(data, use_container_width=True, hide_index=True, **kw)


def show_fig(fig):
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), height=320, legend_title_text="")
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


@st.cache_resource
def get_model():
    return load_or_train()


@st.cache_data
def process(data: bytes, name: str, threshold: float):
    df = add_duplicate_info(get_model().predict(load_and_clean(data, name), threshold))
    return df


@st.cache_data
def model_quality(threshold: float):
    return heldout_eval(get_model(), threshold)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🔎 LogLens")
st.sidebar.caption("Proof of concept · synthetic data")
source = st.sidebar.radio("Data source", ["Synthetic sample", "Upload CSV"])
if source == "Upload CSV":
    up = st.sidebar.file_uploader("Log CSV", type=["csv"])
    st.sidebar.caption("Required columns: timestamp, level, service, message. Optional: request_id, "
                       "endpoint, status_code, latency_ms. Processed in memory only.")
    if up is None:
        st.info("Upload a log CSV in the sidebar, or switch to the synthetic sample.")
        st.stop()
    raw, name, synthetic = up.getvalue(), up.name, False
else:
    raw, name, synthetic = SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name, True

threshold = st.sidebar.slider("Confidence threshold", 0.3, 0.95, DEFAULT_THRESHOLD, 0.05,
                              help="Predictions below this confidence go to Review Required. "
                                   "Confidence is a model score, not a calibrated probability.")
try:
    data = process(raw, name, threshold)
except ValidationError as exc:
    st.error(f"Could not use this file: {exc}")
    st.stop()

st.sidebar.subheader("Filters")
classes = st.sidebar.multiselect("Classification", CLASSES, default=CLASSES)
levels_all = [l for l in LEVEL_ORDER if l in set(data["level_clean"])]
levels = st.sidebar.multiselect("Severity", levels_all, default=levels_all)
services_all = sorted(data["service"].unique())
services = st.sidebar.multiselect("Service", services_all, default=services_all)
search = st.sidebar.text_input("Search message")

df = data[data["predicted_class"].isin(classes) & data["level_clean"].isin(levels)
          & data["service"].isin(services)]
if search.strip():
    df = df[df["message"].str.contains(search.strip(), case=False, regex=False)]

st.title("LogLens")
st.caption(f"{'Synthetic sample' if synthetic else name} · {len(data):,} records loaded · "
           f"{len(df):,} shown after filters")
if (data["data_issues"] != "").any():
    st.warning(f"{(data['data_issues'] != '').sum()} record(s) have data issues (bad timestamp, "
               "empty message, ...). They are kept and still classified.")

tab_over, tab_explore, tab_impact = st.tabs(["Overview", "Log Explorer", "Impact & Efficiency"])

# ---------------------------------------------------------------- overview
with tab_over:
    if df.empty:
        st.info("No records match the filters.")
    else:
        c = st.columns(6)
        c[0].metric("Total logs", f"{len(df):,}")
        c[1].metric("Keep", f"{(df.predicted_class == KEEP).sum():,}")
        c[2].metric("Suppress Candidate", f"{(df.predicted_class == SUPPRESS).sum():,}")
        c[3].metric("Review Required", f"{(df.predicted_class == REVIEW).sum():,}")
        c[4].metric("Duplicate occurrences", f"{impact(df)['duplicate_occurrences']:,}")
        c[5].metric("Services", df["service"].nunique())
        a, b = st.columns(2)
        with a:
            d = df["predicted_class"].value_counts().reindex(CLASSES, fill_value=0).reset_index()
            d.columns = ["Classification", "Records"]
            show_fig(px.bar(d, x="Classification", y="Records", color="Classification",
                            color_discrete_map=COLORS, title="Classification"))
        with b:
            d = df["level_clean"].value_counts().reindex(LEVEL_ORDER, fill_value=0)
            d = d[d > 0].reset_index()
            d.columns = ["Severity", "Records"]
            show_fig(px.bar(d, x="Severity", y="Records", title="Logs by severity"))
        timed = df.dropna(subset=["ts"])
        if not timed.empty:
            vol = timed.groupby([pd.Grouper(key="ts", freq="6h"), "predicted_class"]).size().reset_index(name="Records")
            show_fig(px.bar(vol, x="ts", y="Records", color="predicted_class", color_discrete_map=COLORS,
                            title="Log volume over time (6-hour buckets)"))
        err = df[df["level_clean"].isin(PROTECTED_LEVELS)]
        if not err.empty:
            e = err.groupby(["service", "level_clean"]).size().reset_index(name="Records")
            show_fig(px.bar(e, x="service", y="Records", color="level_clean", title="Errors by service"))

# ---------------------------------------------------------------- explorer
with tab_explore:
    if df.empty:
        st.info("No records match the filters.")
    else:
        view = df[["record_id", "ts", "level_clean", "service", "message", "status_clean", "predicted_class",
                   "confidence", "reason", "dup_group_id", "group_size", "data_issues"]]
        view.columns = ["Record", "Time", "Severity", "Service", "Message", "Status", "Classification",
                        "Confidence", "Why", "Group", "Group size", "Data issues"]
        st.caption("'Why' explains each classification. Duplicates stay visible: 'Group size' shows how often "
                   "the same message occurred.")
        show_df(view.head(3000))
        if len(view) > 3000:
            st.caption(f"Showing the first 3,000 of {len(view):,} rows; the download has all of them.")
        st.download_button("⬇ Download classified CSV", record_csv(df), "loglens_classified.csv", "text/csv")

# ---------------------------------------------------------------- impact
with tab_impact:
    st.caption("Operational usefulness is shown separately from model quality. Everything follows the filters.")
    secs = st.number_input("Assumed seconds to inspect one record (editable assumption)", 1.0, 600.0, 20.0, 1.0)
    if df.empty:
        st.info("No records match the filters.")
    else:
        m = impact(df, secs)
        r = st.columns(5)
        r[0].metric("Total logs processed", f"{m['total']:,}")
        r[1].metric("Classification coverage", f"{m['coverage_pct']}%", help=FORMULAS["Classification coverage"])
        r[2].metric("Potential noise reduction", f"{m['noise_reduction_pct']}%",
                    delta=f"{m['eligible_records']:,} records", delta_color="off", help=FORMULAS["Potential noise reduction"])
        r[3].metric("Duplicate rate", f"{m['duplicate_rate_pct']}%", help=FORMULAS["Duplicate rate"])
        r[4].metric("Est. review time saved", f"{m['minutes_saved']} min", delta="estimate", delta_color="off",
                    help=FORMULAS["Estimated time saved"])
        r = st.columns(4)
        r[0].metric("Unique message groups", f"{m['unique_groups']:,}")
        r[1].metric("Review queue", f"{m['review']:,}")
        r[2].metric("Actionable (Keep)", f"{m['keep']:,}")
        r[3].metric("Errors collapsed", m["errors_collapsed"], help="Must always be 0.")

        st.subheader("Before vs after")
        st.caption("Different views of the same records: the bars overlap and are not additive. "
                   "Nothing is deleted; collapsing only folds repeated informational messages into one row.")
        ba = pd.DataFrame({"Measure": ["Raw records", "Unique message groups", "Eligible for grouping",
                                       "Rows when grouped", "Important events kept (Keep)", "Need review"],
                           "Records": [m["total"], m["unique_groups"], m["eligible_records"], m["display_rows"],
                                       m["keep"], m["review"]]})
        fig = px.bar(ba, y="Measure", x="Records", orientation="h", text="Records")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        show_fig(fig)
        mask = collapsible_mask(df)
        if mask.any():
            g = df[mask].groupby("service").size().reset_index(name="Records")
            show_fig(px.bar(g, x="service", y="Records", title="Groupable repetitive entries by service"))

        st.subheader("Model quality (held-out labeled data)")
        q = model_quality(threshold)
        st.caption(f"Evaluated on {q['n']} labeled synthetic records that share no message wording with the training "
                   "data. Small synthetic data: shows the pipeline works, not real-world accuracy.")
        k = st.columns(3)
        k[0].metric("Accuracy", f"{q['accuracy']:.1%}")
        k[1].metric("Macro F1", f"{q['macro_f1']:.3f}")
        k[2].metric("False suppressions", f"{q['false_suppressions']} of {q['keep_records']}",
                    help="Important (Keep) events wrongly marked Suppress Candidate.")
        left, right = st.columns(2)
        with left:
            show_df(q["per_class"].round(3))
        with right:
            show_fig(px.imshow(q["confusion"], text_auto=True, color_continuous_scale="Blues",
                               labels=dict(x="Predicted", y="Actual"), title="Confusion matrix"))

        st.subheader("Download reports")
        d = st.columns(2)
        d[0].download_button("⬇ impact_report.csv", record_csv(df), "impact_report.csv", "text/csv")
        d[1].download_button("⬇ impact_summary.md", summary_markdown(df, name, secs, q, synthetic),
                             "impact_summary.md", "text/markdown")
